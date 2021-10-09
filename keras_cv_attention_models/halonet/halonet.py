import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from keras_cv_attention_models.attention_layers import (
    batchnorm_with_activation,
    conv2d_no_bias,
    drop_block,
    eca_module,
    RelativePositionalEmbedding,
    se_module,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

PRETRAINED_DICT = {
    "halonet26t": {"imagenet": "e9a08aa78fb0d912283834266ca179f2"},
    "halonet50t": {"imagenet": "8146970cd6443d7679aa0797581432f9"},
    "halonet_se33t": {"imagenet": "58e9382e876f4043d62154e189c065ca"},
    "halonext_eca26t": {"imagenet": "b6e140e9ea99b75de878ef173696c748"},
}


def halo_attention(
    inputs, num_heads=4, key_dim=0, block_size=2, halo_size=1, strides=1, out_shape=None, out_weight=True, out_bias=False, attn_dropout=0, name=None
):
    _, hh, ww, cc = inputs.shape
    key_dim = key_dim if key_dim > 0 else cc // num_heads
    qk_scale = 1.0 / tf.math.sqrt(tf.cast(key_dim, inputs.dtype))
    out_shape = cc if out_shape is None else out_shape
    emb_dim = num_heads * key_dim
    query_block = block_size // strides
    kv_kernel = block_size + halo_size * 2

    query = conv2d_no_bias(inputs, emb_dim, kernel_size=1, strides=strides, name=name and name + "query_")
    _, hh, ww, cc = query.shape
    # print(f">>>> {inputs.shape = }, {query.shape = }, {strides = }")
    # attn_query = rearrange(query, "B (h hb) (w wb) (hd c) -> B hd h w (hb wb) c", hb=query_block, wb=query_block, hd=num_heads)
    # pos_query = rearrange(attn_query, "B hd h w (hb wb) c -> B (hd h w) hb wb c", hb=query_block, wb=query_block)
    hh_qq, ww_qq, cc_qq = hh // query_block, ww // query_block, cc // num_heads
    query = tf.reshape(query, [-1, hh_qq, query_block, ww_qq, query_block, num_heads, cc_qq])
    query = tf.transpose(query, [0, 5, 1, 3, 2, 4, 6])  # [batch, num_heads, hh, ww, query_block, query_block, key_dim]
    # attn_query = [batch, num_heads, hh, ww, query_block * query_block, key_dim]
    attn_query = tf.reshape(query, [-1, num_heads, hh_qq, ww_qq, query_block * query_block, cc_qq]) * qk_scale  # [???] qk_scale NOT multiplied with pos_query
    # pos_query = [batch, num_heads * hh * ww, query_block, query_block, key_dim]
    pos_query = tf.reshape(query, [-1, num_heads * hh_qq * ww_qq, query_block, query_block, cc_qq])

    # key_value = [batch, height, width, key_dim + out_shape]
    key_value = conv2d_no_bias(inputs, emb_dim + out_shape, kernel_size=1, use_bias=False, name=name and name + "key_value_")
    kv_padded = tf.pad(key_value, [[0, 0], [halo_size, halo_size], [halo_size, halo_size], [0, 0]])
    sizes, strides = [1, kv_kernel, kv_kernel, 1], [1, block_size, block_size, 1]
    # kv_inp = [batch, hh, ww, kv_kernel * kv_kernel * (key_dim + out_shape)]
    kv_inp = tf.image.extract_patches(kv_padded, sizes=sizes, strides=strides, rates=[1, 1, 1, 1], padding="VALID")
    # kv_inp = rearrange(kv_inp, "B h w (hb wb hd c) -> B hd h w (hb wb) c", hb=kv_kernel, wb=kv_kernel, hd=num_heads)
    _, hh_kk, ww_kk, cc = kv_inp.shape
    cc_kk = cc // num_heads // kv_kernel // kv_kernel
    kv_inp = tf.reshape(kv_inp, [-1, hh_kk, ww_kk, kv_kernel, kv_kernel, num_heads, cc_kk])
    kv_inp = tf.transpose(kv_inp, [0, 5, 1, 2, 3, 4, 6])
    kv_inp = tf.reshape(kv_inp, [-1, num_heads, hh_kk, ww_kk, kv_kernel * kv_kernel, cc_kk])

    # key = [batch, num_heads, hh, ww, kv_kernel * kv_kernel, key_dim]
    # value = [batch, num_heads, hh, ww, kv_kernel * kv_kernel, out_dim]
    key, value = tf.split(kv_inp, [emb_dim // num_heads, out_shape // num_heads], axis=-1)

    # scaled_dot_product_attention
    # print(f">>>> {attn_query.shape = }, {key.shape = }, {value.shape = }, {kv_inp.shape = }, {pos_query.shape = }, {num_heads = }")
    # attention_scores = [batch, num_heads, hh, ww, query_block * query_block, kv_kernel * kv_kernel]
    attention_scores = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1], transpose_b=True))([attn_query, key])
    # pos = [batch, num_heads * hh * ww, query_block, query_block, kv_kernel, kv_kernel]
    pos = RelativePositionalEmbedding(position_height=kv_kernel, name=name and name + "pos_emb")(pos_query)
    # print(f">>>> {pos.shape = }, {attention_scores.shape = }")
    pos = tf.reshape(pos, [-1, *attention_scores.shape[1:]])
    attention_scores = keras.layers.Add()([attention_scores, pos])
    attention_scores = tf.nn.softmax(attention_scores, axis=-1)

    if attn_dropout > 0:
        attention_scores = keras.layers.Dropout(attn_dropout, name=name and name + "attn_drop")(attention_scores)

    # attention_output = [batch, num_heads, hh, ww, query_block * query_block, out_dim]
    attention_output = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([attention_scores, value])
    # attention_output = rearrange(attention_output, "B hd h w (hb wb) c -> B (h hb) (w wb) (hd c)", hb=query_block, wb=query_block)
    _, heads, hh_aa, ww_aa, patch, cc_aa = attention_output.shape
    attention_output = tf.reshape(attention_output, [-1, heads, hh_aa, ww_aa, query_block, query_block, cc_aa])
    attention_output = tf.transpose(attention_output, [0, 2, 4, 3, 5, 1, 6])
    attention_output = tf.reshape(attention_output, [-1, hh_aa * query_block, ww_aa * query_block, heads * cc_aa])
    # print(f">>>> {attention_output.shape = }, {attention_scores.shape = }")

    if out_weight:
        # [batch, hh, ww, num_heads * out_dim] * [out, out] --> [batch, hh, ww, out]
        attention_output = keras.layers.Dense(out_shape, use_bias=out_bias, name=name and name + "output")(attention_output)
    return attention_output


def halo_block(
    inputs,
    filter,
    strides=1,
    shortcut=False,
    expansion=2,
    attn_type="halo",
    num_heads=4,
    drop_rate=0,
    halo_expansion=1,
    block_size=8,
    halo_size=4,
    key_dim=0,
    group_size=0,
    activation="swish",
    name="",
):
    expanded_filter = round(filter * expansion)
    if shortcut:
        # print(">>>> Downsample")
        shortcut = conv2d_no_bias(inputs, expanded_filter, 1, strides=strides, name=name + "shortcut_")
        shortcut = batchnorm_with_activation(shortcut, activation=None, zero_gamma=False, name=name + "shortcut_")
    else:
        shortcut = inputs

    deep = conv2d_no_bias(inputs, filter, 1, name=name + "deep_1_")
    deep = batchnorm_with_activation(deep, activation=activation, zero_gamma=False, name=name + "deep_1_")

    if attn_type == "halo":
        out_shape = int(filter * halo_expansion)
        deep = halo_attention(deep, num_heads, key_dim, block_size, halo_size, strides=strides, out_shape=out_shape, out_weight=False, name=name + "halo_")
    else:
        groups = 1 if group_size < 1 else filter // group_size
        deep = conv2d_no_bias(deep, filter, 3, strides=strides, padding="SAME", groups=groups, name=name + "deep_2_")
    # print(">>>>", deep.shape)
    deep = batchnorm_with_activation(deep, activation=activation, zero_gamma=False, name=name + "deep_2_")

    if attn_type == "se":   # SE
        se_ratio = 1 / 16
        deep = se_module(deep, se_ratio=se_ratio, activation="relu", name=name + "se_")
    elif attn_type == "eca":  # ECA
        deep = eca_module(deep, name=name + "eca_")

    deep = conv2d_no_bias(deep, expanded_filter, 1, name=name + "deep_3_")
    deep = batchnorm_with_activation(deep, activation=None, zero_gamma=True, name=name + "deep_3_")

    # print(">>>>", deep.shape, shortcut.shape)
    deep = drop_block(deep, drop_rate)
    deep = keras.layers.Add(name=name + "add")([shortcut, deep])
    return keras.layers.Activation(activation, name=name + "out")(deep)


def halo_stack(inputs, blocks, filter, strides=1, expansion=1, attn_type="halo", num_heads=4, stack_drop=0, block_params={}, name=""):
    nn = inputs
    for id in range(blocks):
        strides = strides if id == 0 else 1
        shortcut = True if strides != 1 or nn.shape[-1] != filter * expansion else False
        block_name = name + "block{}_".format(id + 1)
        cur_attn_type = attn_type[id] if isinstance(attn_type, (list, tuple)) else attn_type
        drop_rate = stack_drop[id] if isinstance(stack_drop, (list, tuple)) else stack_drop
        nn = halo_block(nn, filter, strides, shortcut, expansion, cur_attn_type, num_heads, drop_rate, **block_params, name=block_name)
    return nn


def halo_stem(inputs, stem_width=64, stem_pool="maxpool", activation="relu", tiered_stem=False, name=""):
    if tiered_stem:
        nn = conv2d_no_bias(inputs, 3 * stem_width // 8, 3, strides=2, padding="same", name=name + "1_")
        nn = batchnorm_with_activation(nn, activation=activation, name=name + "1_")
        nn = conv2d_no_bias(nn, stem_width // 2, 3, strides=1, padding="same", name=name + "2_")
        nn = batchnorm_with_activation(nn, activation=activation, name=name + "2_")
        strides = 1 if stem_pool is not None and len(stem_pool) != 0 else 2
        nn = conv2d_no_bias(nn, stem_width, 3, strides=strides, padding="same", name=name + "3_")
    else:
        nn = conv2d_no_bias(inputs, stem_width, 7, strides=2, padding="same", name=name)

    nn = batchnorm_with_activation(nn, activation=activation, name="stem_")
    if stem_pool is not None and len(stem_pool) != 0:
        nn = keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name="stem_pool_pad")(nn)
        if "max" in stem_pool.lower():
            nn = keras.layers.MaxPooling2D(3, strides=2, name="stem_pool")(nn)
        else:
            nn = keras.layers.AveragePooling2D(3, strides=2, name="stem_pool")(nn)
    return nn


def HaloNet(
    num_blocks=[3, 3, 10, 3],
    attn_type="halo",
    group_size=0,
    stem_width=64,
    stem_pool="maxpool",
    tiered_stem=False,
    halo_block_size=8,  # b
    halo_halo_size=3,  # h
    halo_expansion=1,  # rv
    expansion=1,  # rb
    output_conv_channel=-1,  # df
    num_heads=[4, 8, 8, 8],
    key_dim=0,
    out_channels=[64, 128, 256, 512],
    strides=[1, 2, 2, 2],
    input_shape=(256, 256, 3),
    num_classes=1000,
    activation="swish",
    drop_connect_rate=0,
    classifier_activation="softmax",
    drop_rate=0,
    pretrained=None,
    model_name="halonet",
    kwargs=None,
):
    inputs = keras.layers.Input(input_shape)
    nn = halo_stem(inputs, stem_width, stem_pool=stem_pool, activation=activation, tiered_stem=tiered_stem, name="stem_")


    # Input (height, width) should be dividable by `halo_block_size`, assume height == width here
    down_sample_rate = tf.reduce_prod(strides) * halo_block_size
    if nn.shape[1] % down_sample_rate != 0:
        gap = down_sample_rate - nn.shape[1] % down_sample_rate
        pad_head, pad_tail = gap // 2, gap - gap // 2
        # print(">>>> pad_head:", pad_head, "pad_tail:", pad_tail)
        nn = keras.layers.ZeroPadding2D(padding=((pad_head, pad_tail), (pad_head, pad_tail)), name="gap_pad")(nn)

    block_params = {  # params same for all blocks
        "block_size": halo_block_size,
        "halo_size": halo_halo_size,
        "halo_expansion": halo_expansion,
        "key_dim": key_dim,
        "activation": activation,
        "group_size": group_size,
    }

    drop_connect_rates = tf.split(tf.linspace(0.0, drop_connect_rate, sum(num_blocks)), num_blocks)
    drop_connect_rates = [ii.numpy().tolist() for ii in drop_connect_rates]
    for id, (num_block, out_channel, stride, drop_connect) in enumerate(zip(num_blocks, out_channels, strides, drop_connect_rates)):
        name = "stack{}_".format(id + 1)
        cur_attn_type = attn_type[id] if isinstance(attn_type, (list, tuple)) else attn_type
        cur_expansion = expansion[id] if isinstance(expansion, (list, tuple)) else expansion
        num_head = num_heads[id] if isinstance(num_heads, (list, tuple)) else num_heads
        nn = halo_stack(nn, num_block, out_channel, stride, cur_expansion, cur_attn_type, num_head, drop_connect, block_params, name=name)

    if output_conv_channel > 0:
        nn = conv2d_no_bias(nn, output_conv_channel, 1, name="post_")
        nn = batchnorm_with_activation(nn, activation=activation, name="post_")

    if num_classes > 0:
        nn = keras.layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        if drop_rate > 0:
            nn = keras.layers.Dropout(drop_rate, name="head_drop")(nn)
        nn = keras.layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="predictions")(nn)

    model = keras.models.Model(inputs, nn, name=model_name)
    reload_model_weights(model, pretrained_dict=PRETRAINED_DICT, sub_release="halonet", input_shape=input_shape, pretrained=pretrained)
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


def HaloNet26T(input_shape=(256, 256, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 2, 2, 2]
    attn_type = [None, None, [None, 'halo'], 'halo']
    expansion = 4
    halo_block_size = 8
    halo_halo_size = 2
    num_heads = [0, 0, 8, 8]
    # key_dim = 16
    tiered_stem = True
    return HaloNet(model_name="halonet26t", **locals(), **kwargs)


def HaloNet50T(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 6, 3]
    attn_type = [None, [None, None, None, 'halo'], [None, 'halo'] * 3, [None, 'halo', None]]
    expansion = 4
    halo_block_size = 8
    halo_halo_size = 3
    num_heads = [0, 4, 8, 8]
    # key_dim = 16
    tiered_stem = True
    return HaloNet(model_name="halonet50t", **locals(), **kwargs)


def HaloNetSE33T(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 3, 3, 2]
    attn_type = ["se", ["se", "se", 'halo'], ["se", "se", 'halo'], 'halo']
    expansion = [4, 4, 4, 3]
    stem_pool = None
    halo_block_size = 8
    halo_halo_size = 3
    num_heads = 8
    tiered_stem = True
    output_conv_channel = 1280
    return HaloNet(model_name="halonet_se33t", **locals(), **kwargs)


def HaloNextECA26T(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 2, 2, 2]
    attn_type = ["eca", "eca", ["eca", 'halo'], 'halo']
    group_size = 16
    expansion = 4
    halo_block_size = 8
    halo_halo_size = 2
    num_heads = 8
    key_dim = 16
    tiered_stem = True
    return HaloNet(model_name="halonext_eca26t", **locals(), **kwargs)
