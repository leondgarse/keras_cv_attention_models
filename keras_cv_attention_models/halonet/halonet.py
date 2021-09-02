import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

# from einops import rearrange    # Currently einops 0.3.0 is broken for tf.keras 2.6.0...
from keras_cv_attention_models.attention_layers import batchnorm_with_activation, conv2d_no_bias, RelativePositionalEmbedding


BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


def halo_attention(
    inputs, num_heads=4, key_dim=128, block_size=2, halo_size=1, strides=1, out_shape=None, out_weight=True, out_bias=False, attn_dropout=0, name=None
):
    _, hh, ww, cc = inputs.shape
    qk_scale = 1.0 / tf.math.sqrt(float(key_dim))
    out_shape = cc if out_shape is None or not out_weight else out_shape
    emb_dim = num_heads * key_dim
    final_out_shape = (None, hh // strides, ww // strides, out_shape)
    query_block = block_size // strides
    kv_kernel = block_size + halo_size * 2

    # attn_query = [batch, num_heads, hh, ww, query_block * query_block, key_dim]
    # pos_query = [batch, num_heads * hh * ww, query_block, query_block, key_dim]
    query = inputs if strides == 1 else inputs[:, ::strides, ::strides, :]
    query = keras.layers.Dense(emb_dim, use_bias=False, name=name and name + "query")(query) * qk_scale
    # print(f">>>> {inputs.shape = }, {query.shape = }, {final_out_shape = }, {strides = }")
    # attn_query = rearrange(query, "B (h hb) (w wb) (hd c) -> B hd h w (hb wb) c", hb=query_block, wb=query_block, hd=num_heads)
    # pos_query = rearrange(attn_query, "B hd h w (hb wb) c -> B (hd h w) hb wb c", hb=query_block, wb=query_block)
    _, hh, ww, cc = query.shape
    hh_qq, ww_qq, cc_qq = hh // query_block, ww // query_block, cc // num_heads
    query = tf.reshape(query, [-1, hh_qq, query_block, ww_qq, query_block, num_heads, cc_qq])
    query = tf.transpose(query, [0, 5, 1, 3, 2, 4, 6])
    attn_query = tf.reshape(query, [-1, num_heads, hh_qq, ww_qq, query_block * query_block, cc_qq])
    pos_query = tf.reshape(query, [-1, num_heads * hh_qq * ww_qq, query_block, query_block, cc_qq])

    # key_value = [batch, num_heads, hh, ww, kv_kernel * kv_kernel, key_dim * 2]
    key_value = keras.layers.Dense(emb_dim * 2, use_bias=False, name=name and name + "key_value")(inputs)
    kv_padded = tf.pad(key_value, [[0, 0], [halo_size, halo_size], [halo_size, halo_size], [0, 0]])
    sizes, strides = [1, kv_kernel, kv_kernel, 1], [1, block_size, block_size, 1]
    kv_inp = tf.image.extract_patches(kv_padded, sizes=sizes, strides=strides, rates=[1, 1, 1, 1], padding="VALID")
    # kv_inp = rearrange(kv_inp, "B h w (hb wb hd c) -> B hd h w (hb wb) c", hb=kv_kernel, wb=kv_kernel, hd=num_heads)
    _, hh_kk, ww_kk, cc = kv_inp.shape
    cc_kk = cc // num_heads // kv_kernel // kv_kernel
    kv_inp = tf.reshape(kv_inp, [-1, hh_kk, ww_kk, kv_kernel, kv_kernel, num_heads, cc_kk])
    kv_inp = tf.transpose(kv_inp, [0, 5, 1, 2, 3, 4, 6])
    kv_inp = tf.reshape(kv_inp, [-1, num_heads, hh_kk, ww_kk, kv_kernel * kv_kernel, cc_kk])

    # key = value = [batch, num_heads, hh, ww, kv_kernel * kv_kernel, key_dim]
    key, value = tf.split(kv_inp, 2, axis=-1)

    # scaled_dot_product_attention
    # print(f">>>> {attn_query.shape = }, {key.shape = }, {value.shape = }, {kv_inp.shape = }, {pos_query.shape = }")
    # [batch, num_heads, hh, ww, query_block * query_block, kv_kernel * kv_kernel]
    # attention_scores = tf.matmul(attn_query, key, transpose_b=True)
    attention_scores = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1], transpose_b=True))([attn_query, key])
    # [batch, num_heads * hh * ww, query_block, query_block, kv_kernel, kv_kernel]
    pos = RelativePositionalEmbedding(position_height=kv_kernel, name=name and name + "pos_emb")(pos_query)
    pos = tf.reshape(pos, [-1, *attention_scores.shape[1:]])
    # print(f">>>> {pos.shape = }, {attention_scores.shape = }")
    attention_scores = keras.layers.Add()([attention_scores, pos])
    attention_scores = tf.nn.softmax(attention_scores, axis=-1)

    if attn_dropout > 0:
        attention_scores = keras.layers.Dropout(attn_dropout, name=name and name + "attn_drop")(attention_scores)

    # attention_output = tf.matmul(attention_scores, value)  # [batch, num_heads, hh, ww, query_block * query_block, key_dim]
    attention_output = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([attention_scores, value])
    # attention_output = rearrange(attention_output, "B hd h w (hb wb) c -> B (h hb) (w wb) (hd c)", hb=query_block, wb=query_block)
    _, heads, hh_aa, ww_aa, patch, cc_aa = attention_output.shape
    attention_output = tf.reshape(attention_output, [-1, heads, hh_aa, ww_aa, query_block, query_block, cc_aa])
    attention_output = tf.transpose(attention_output, [0, 2, 4, 3, 5, 1, 6])
    attention_output = tf.reshape(attention_output, [-1, hh_aa * query_block, ww_aa * query_block, heads * cc_aa])
    # print(f">>>> {attention_output.shape = }, {attention_scores.shape = }")

    if out_weight:
        # [batch, hh, ww, num_heads * key_dim] * [num_heads * key_dim, out] --> [batch, hh, ww, out]
        attention_output = keras.layers.Dense(out_shape, use_bias=out_bias, name=name and name + "output")(attention_output)
    attention_output.set_shape(final_out_shape)
    return attention_output


def halo_block(inputs, filter, strides=1, shortcut=False, expansion=2, num_heads=4, halo_expansion=1, block_size=8, halo_size=4, activation="swish", name=""):
    # target_dimension = round(planes * block.expansion * self.rb)
    expanded_filter = round(filter * expansion)
    if shortcut:
        # print(">>>> Downsample")
        shortcut = conv2d_no_bias(inputs, expanded_filter, 1, strides=strides, name=name + "shorcut_")
        shortcut = batchnorm_with_activation(shortcut, activation=None, zero_gamma=False, name=name + "shorcut_")
    else:
        shortcut = inputs

    # width = planes
    nn = conv2d_no_bias(inputs, filter, 1, name=name + "1_")
    nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name=name + "1_")

    # num_heads=4, key_dim=128, block_size=2, halo_size=1, out_shape=None, out_weight=True, out_bias=False, attn_dropout=0
    # key_dim = round( dim_head * rv)
    # print(">>>>", nn.shape, num_heads, key_dim, block_size, halo_size)
    out_shape = int(filter * halo_expansion)
    key_dim = filter // num_heads
    # print(f"{filter // num_heads = }")
    # key_dim = 16
    nn = halo_attention(nn, num_heads, key_dim, block_size, halo_size, strides=strides, out_shape=out_shape, out_bias=True, name=name + "halo_")
    # print(">>>>", nn.shape)
    # nn = keras.layers.Activation(activation=activation)(nn)
    nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name=name + "halo_")
    # round(planes * self.expansion * rb), expansion = 2
    nn = conv2d_no_bias(nn, expanded_filter, 1, name=name + "2_")
    nn = batchnorm_with_activation(nn, activation=None, zero_gamma=True, name=name + "2_")

    # print(">>>>", nn.shape, shortcut.shape)
    nn = keras.layers.Add(name=name + "add")([shortcut, nn])
    return keras.layers.Activation(activation, name=name + "out")(nn)


def halo_stack(inputs, blocks, filter, strides=1, expansion=2, num_heads=4, halo_expansion=1, block_size=8, halo_size=4, activation="swish", name=""):
    shortcut = True if strides != 1 or inputs.shape[-1] != filter * 2 or expansion != 2 else False
    nn = halo_block(inputs, filter, strides, shortcut, expansion, num_heads, halo_expansion, block_size, halo_size, activation, name=name + "block1_")
    shortcut = False
    for ii in range(2, blocks + 1):
        block_name = name + "block{}_".format(ii)
        nn = halo_block(nn, filter, 1, shortcut, expansion, num_heads, halo_expansion, block_size, halo_size, activation, name=block_name)
    return nn


def HaloNet(
    halo_block_size=8,  # b
    halo_halo_size=3,  # h
    halo_expansion=1,  # rv
    expansion=1,  # rb
    output_conv_channel=-1,  # df
    num_blocks=[3, 3, 10, 3],
    num_heads=[4, 8, 8, 8],
    out_channels=[64, 128, 256, 512],
    strides=[1, 2, 2, 2],
    input_shape=(256, 256, 3),
    num_classes=1000,
    activation="swish",
    classifier_activation="softmax",
    pretrained=None,
    model_name="halonet",
    **kwargs
):
    inputs = keras.layers.Input(input_shape)
    nn = keras.layers.ZeroPadding2D(padding=3, name="stem_conv_pad")(inputs)
    nn = conv2d_no_bias(nn, 64, 7, strides=2, padding="VALID", name="stem_")
    nn = batchnorm_with_activation(nn, activation=activation, name="stem_")
    nn = keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name="stem_pool_pad")(nn)
    nn = keras.layers.MaxPooling2D(3, strides=2, name="stem_pool")(nn)

    # Input (height, width) should be dividable by `halo_block_size`, assume height == width here
    down_sample_rate = tf.reduce_prod(strides) * halo_block_size
    if nn.shape[1] % down_sample_rate != 0:
        gap = down_sample_rate - nn.shape[1] % down_sample_rate
        pad_head, pad_tail = gap // 2, gap - gap // 2
        # print(">>>> pad_head:", pad_head, "pad_tail:", pad_tail)
        nn = keras.layers.ZeroPadding2D(padding=((pad_head, pad_tail), (pad_head, pad_tail)), name="gap_pad")(nn)

    for id, (num_block, out_channel, num_head, stride) in enumerate(zip(num_blocks, out_channels, num_heads, strides)):
        name = "stack{}_".format(id + 1)
        nn = halo_stack(
            nn, num_block, out_channel, stride, expansion, num_head, halo_expansion, halo_block_size, halo_halo_size, activation=activation, name=name
        )

    if output_conv_channel > 0:
        nn = conv2d_no_bias(nn, output_conv_channel, 1, name="post_")
        nn = batchnorm_with_activation(nn, activation=activation, name="post_")

    if num_classes > 0:
        nn = keras.layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        nn = keras.layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="predictions")(nn)

    model = keras.models.Model(inputs, nn, name=model_name)
    return model


BLOCK_CONFIGS = {
    "h0": {  # rv = 1, rb = 0.5
        "halo_block_size": 8,  # b
        "halo_halo_size": 3,  # h
        "halo_expansion": 1,  # rv
        "expansion": 0.5,  # rb
        "output_conv_channel": -1,  # df
        "num_blocks": [3, 3, 7, 3],
        "num_heads": [4, 8, 8, 8],
    },
    "h1": {  # rv = 1, rb = 1
        "halo_block_size": 8,
        "halo_halo_size": 3,
        "halo_expansion": 1,  # rv
        "expansion": 1,  # rb
        "output_conv_channel": -1,  # df
        "num_blocks": [3, 3, 10, 3],
        "num_heads": [4, 8, 8, 8],
    },
    "h2": {  # rv = 1, rb = 1.25
        "halo_block_size": 8,
        "halo_halo_size": 3,
        "halo_expansion": 1,  # rv
        "expansion": 1.25,  # rb
        "output_conv_channel": -1,  # df
        "num_blocks": [3, 3, 11, 3],
        "num_heads": [4, 8, 8, 8],
    },
    "h3": {  # rv = 1, rb = 1.5
        "halo_block_size": 10,
        "halo_halo_size": 3,
        "halo_expansion": 1,  # rv
        "expansion": 1.5,  # rb
        "output_conv_channel": 1024,  # df
        "num_blocks": [3, 3, 12, 3],
        "num_heads": [4, 8, 8, 8],
    },
    "h4": {  # rv = 1, rb = 3
        "halo_block_size": 12,
        "halo_halo_size": 2,
        "halo_expansion": 1,  # rv
        "expansion": 3,  # rb
        "output_conv_channel": 1280,  # df
        "num_blocks": [3, 3, 12, 3],
        "num_heads": [4, 8, 8, 8],
    },
    "h5": {  # rv = 2.5, rb = 2
        "halo_block_size": 14,
        "halo_halo_size": 2,
        "halo_expansion": 2.5,  # rv
        "expansion": 2,  # rb
        "output_conv_channel": 1536,  # df
        "num_blocks": [3, 3, 23, 3],
        "num_heads": [4, 8, 8, 8],
    },
    "h6": {  # rv = 3, rb = 2.75
        "halo_block_size": 8,
        "halo_halo_size": 4,
        "halo_expansion": 3,  # rv
        "expansion": 2.75,  # rb
        "output_conv_channel": 1536,  # df
        "num_blocks": [3, 3, 24, 3],
        "num_heads": [4, 8, 8, 8],
    },
    "h7": {  # rv = 4, rb = 3.5
        "halo_block_size": 10,
        "halo_halo_size": 3,
        "halo_expansion": 4,  # rv
        "expansion": 3.5,  # rb
        "output_conv_channel": 2048,  # df
        "num_blocks": [3, 3, 26, 3],
        "num_heads": [4, 8, 8, 8],
    },
}


def HaloNetH0(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return HaloNet(**BLOCK_CONFIGS["h0"], model_name="haloneth0", **locals(), **kwargs)


def HaloNetH1(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return HaloNet(**BLOCK_CONFIGS["h1"], model_name="haloneth1", **locals(), **kwargs)


def HaloNetH2(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return HaloNet(**BLOCK_CONFIGS["h2"], model_name="haloneth2", **locals(), **kwargs)


def HaloNetH3(input_shape=(320, 320, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return HaloNet(**BLOCK_CONFIGS["h3"], model_name="haloneth3", **locals(), **kwargs)


def HaloNetH4(input_shape=(384, 384, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return HaloNet(**BLOCK_CONFIGS["h4"], model_name="haloneth4", **locals(), **kwargs)


def HaloNetH5(input_shape=(448, 448, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return HaloNet(**BLOCK_CONFIGS["h5"], model_name="haloneth5", **locals(), **kwargs)


def HaloNetH6(input_shape=(512, 512, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return HaloNet(**BLOCK_CONFIGS["h6"], model_name="haloneth6", **locals(), **kwargs)


def HaloNetH7(input_shape=(600, 600, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return HaloNet(**BLOCK_CONFIGS["h7"], model_name="haloneth7", **locals(), **kwargs)
