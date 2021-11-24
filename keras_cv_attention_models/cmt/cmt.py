import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from keras_cv_attention_models.attention_layers import (
    batchnorm_with_activation,
    conv2d_no_bias,
    depthwise_conv2d_no_bias,
    drop_block,
    layer_norm,
    RelativePositionalEmbedding,
)


def light_multi_head_self_attention(inputs, num_heads=4, key_dim=0, sr_ratio=1, out_shape=None, out_weight=True, out_bias=False, attn_dropout=0, name=""):
    _, hh, ww, cc = inputs.shape
    key_dim = key_dim if key_dim > 0 else cc // num_heads
    qk_scale = 1.0 / tf.math.sqrt(tf.cast(key_dim, inputs.dtype))
    out_shape = cc if out_shape is None or not out_weight else out_shape
    emb_dim = num_heads * key_dim

    query = keras.layers.Dense(emb_dim, use_bias=False, name=name and name + "query")(inputs) * qk_scale
    # print(f">>>> {inputs.shape = }, {query.shape = }, {final_out_shape = }, {strides = }")
    _, query_hh, query_ww, _ = query.shape
    query = tf.reshape(query, [-1, query_hh, query_ww, num_heads, key_dim])
    pos_query = tf.transpose(query, [0, 3, 1, 2, 4])  # [batch, num_heads, hh, ww, key_dim]
    attn_query = tf.reshape(pos_query, [-1, num_heads, query_hh * query_ww, key_dim])  # [batch, num_heads, hh * ww, key_dim]

    if sr_ratio > 1:
        key_value = depthwise_conv2d_no_bias(inputs, kernel_size=sr_ratio, strides=sr_ratio, name=name + "kv_sr_")
        # key_value = conv2d_no_bias(inputs, inputs.shape[-1], kernel_size=sr_ratio, strides=sr_ratio, use_bias=True, name=name + "kv_sr_")
        key_value = layer_norm(key_value, name=name + "kv_sr_")
        # key_value = inputs[:, ::sr_ratio, ::sr_ratio, :]
    else:
        key_value = inputs

    # key_value = [batch, num_heads, hh, ww, kv_kernel * kv_kernel, key_dim * 2]
    key_value = keras.layers.Dense(emb_dim * 2, use_bias=False, name=name and name + "key_value")(key_value)
    _, key_hh, key_ww, _ = key_value.shape
    # key_value = tf.reshape(qkv, [-1, hh * ww, 2, num_heads, key_dim])
    # key, value = tf.transpose(key_value, [2, 0, 3, 1, 4])   # [2, batch, num_heads, blocks, key_dim]
    key_value = tf.reshape(key_value, [-1, key_hh * key_ww, key_dim, num_heads, 2])
    key, value = tf.transpose(key_value, [4, 0, 3, 1, 2])  # [2, batch, num_heads, blocks, key_dim]

    # scaled_dot_product_attention
    # print(f">>>> {attn_query.shape = }, {key.shape = }, {value.shape = }, {kv_inp.shape = }, {pos_query.shape = }")
    # [batch, num_heads, hh, ww, query_block * query_block, kv_kernel * kv_kernel]
    # attention_scores = tf.matmul(attn_query, key, transpose_b=True)
    attention_scores = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1], transpose_b=True))([attn_query, key])
    # [batch, num_heads * hh * ww, query_block, query_block, kv_kernel, kv_kernel]
    pos = RelativePositionalEmbedding(name=name and name + "pos_emb")(pos_query)
    # print(f">>>> {pos.shape = }, {attention_scores.shape = }")
    pos = pos[..., -key_hh:, -key_ww:]
    pos = tf.reshape(pos, [-1, *attention_scores.shape[1:]])
    attention_scores = keras.layers.Add()([attention_scores, pos])
    # attention_scores = tf.nn.softmax(attention_scores, axis=-1)
    attention_scores = keras.layers.Softmax(axis=-1, name=name and name + "attention_scores")(attention_scores)

    if attn_dropout > 0:
        attention_scores = keras.layers.Dropout(attn_dropout, name=name and name + "attn_drop")(attention_scores)
    # value = [batch, num_heads, key_hh * key_ww, key_dim]
    # attention_output = tf.matmul(attention_scores, value)  # [batch, num_heads, query_hh * query_ww, key_dim]
    attention_output = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([attention_scores, value])
    attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
    attention_output = tf.reshape(attention_output, [-1, inputs.shape[1], inputs.shape[2], num_heads * key_dim])
    # print(f">>>> {attention_output.shape = }, {attention_scores.shape = }")

    if out_weight:
        # [batch, hh, ww, num_heads * key_dim] * [num_heads * key_dim, out] --> [batch, hh, ww, out]
        attention_output = keras.layers.Dense(out_shape, use_bias=out_bias, name=name and name + "output")(attention_output)
    # attention_output.set_shape(final_out_shape)
    return attention_output


def inverted_residual_feed_forward(inputs, expansion=4, activation="gelu", name=""):
    """ IRFFN(X) = Conv(F(Conv(X))), F(X) = DWConv(X) + X """
    in_channel = inputs.shape[-1]
    expanded = conv2d_no_bias(inputs, int(in_channel * expansion), kernel_size=1, use_bias=True, name=name + "1_")
    expanded = batchnorm_with_activation(expanded, activation=activation, act_first=True, name=name + "1_")

    dw = depthwise_conv2d_no_bias(expanded, kernel_size=3, padding="SAME", use_bias=True, name=name)
    # dw = batchnorm_with_activation(dw, activation=activation, act_first=True, name=name + "2_")
    dw_out = keras.layers.Add(name=name + "dw_out")([expanded, dw])
    dw_out = batchnorm_with_activation(dw_out, activation=activation, act_first=True, name=name + "2_")

    pw = conv2d_no_bias(dw_out, in_channel, kernel_size=1, use_bias=False, name=name + "3_")
    pw = batchnorm_with_activation(pw, activation=None, name=name + "3_")
    return pw


def cmt_block(inputs, num_heads=4, sr_ratio=1, expansion=4, activation="gelu", drop_rate=0, name=""):
    """ X0 = LPU(Xi), X1 = LMHSA(LN(X0)) + X0, X2 = IRFFN(LN(X1)) + X1 """
    """ Local Perception Unit, LPU(X) = DWConv(X) + X """
    lpu = depthwise_conv2d_no_bias(inputs, kernel_size=3, padding="SAME", use_bias=True, name=name)
    # lpu = batchnorm_with_activation(lpu, activation=activation, name=name + "lpu_", act_first=True)
    lpu_out = keras.layers.Add(name=name + "lpu_out")([inputs, lpu])

    """ light multi head self attention """
    attn = layer_norm(lpu_out, name=name + "attn_")
    attn = light_multi_head_self_attention(attn, num_heads=num_heads, sr_ratio=sr_ratio, name=name + "light_mhsa_")
    attn_out = keras.layers.Add(name=name + "attn_out")([lpu_out, attn])

    """ inverted residual feed forward """
    ffn = layer_norm(attn_out, name=name + "ffn_")
    ffn = inverted_residual_feed_forward(ffn, expansion=expansion, activation=activation, name=name + "ffn_")
    ffn = drop_block(ffn, drop_rate=drop_rate)
    ffn_out = keras.layers.Add(name=name + "ffn_out")([attn_out, ffn])

    return ffn_out


def cmt_stem(inputs, stem_width, activation="gelu", name="", **kwargs):
    nn = conv2d_no_bias(inputs, stem_width, kernel_size=3, strides=2, padding="same", use_bias=False, name=name + "1_")
    nn = batchnorm_with_activation(nn, activation=activation, act_first=False, name=name + "1_")
    nn = conv2d_no_bias(nn, stem_width, kernel_size=3, strides=1, padding="same", use_bias=False, name=name + "2_")
    nn = batchnorm_with_activation(nn, activation=activation, act_first=False, name=name + "2_")
    nn = conv2d_no_bias(nn, stem_width, kernel_size=3, strides=1, padding="same", use_bias=False, name=name + "3_")
    nn = batchnorm_with_activation(nn, activation=activation, act_first=False, name=name + "3_")
    return nn


def CMT(
    num_blocks,
    out_channels,
    stem_width=16,
    num_heads=[1, 2, 4, 8],
    sr_ratios=[8, 4, 2, 1],
    ffn_expansion=4,
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="gelu",
    drop_connect_rate=0,
    classifier_activation="softmax",
    drop_rate=0,
    pretrained=None,
    model_name="cmt",
    kwargs=None,
):
    inputs = keras.layers.Input(input_shape)
    nn = cmt_stem(inputs, stem_width=stem_width, activation=activation, name="stem_")

    """ stage [1, 2, 3, 4] """
    total_blocks = sum(num_blocks)
    global_block_id = 0
    for stack_id, (num_block, out_channel, num_head, sr_ratio) in enumerate(zip(num_blocks, out_channels, num_heads, sr_ratios)):
        stage_name = "stage_{}_".format(stack_id + 1)
        nn = conv2d_no_bias(nn, out_channel, kernel_size=2, strides=2, name=stage_name + "down_sample")
        # nn = layer_norm(nn, name=stage_name)
        for block_id in range(num_block):
            name = stage_name + "block_{}_".format(block_id + 1)
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            global_block_id += 1
            nn = cmt_block(nn, num_head, sr_ratio, ffn_expansion, activation=activation, drop_rate=block_drop_rate, name=name)

    nn = conv2d_no_bias(nn, 1280, 1, strides=1, name="post_")
    nn = batchnorm_with_activation(nn, activation=activation, name="post_")

    if num_classes > 0:
        nn = keras.layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        # nn = keras.layers.Dense(1280, name="post_dense")(nn)
        if drop_rate > 0:
            nn = keras.layers.Dropout(drop_rate)(nn)
        nn = keras.layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="predictions")(nn)

    model = keras.models.Model(inputs, nn, name=model_name)
    return model


BLOCK_CONFIGS = {
    "tiny": {
        "num_blocks": [2, 2, 10, 2],
        "out_channels": [46, 92, 184, 368],
        "stem_width": 16,
        "num_heads": [1, 2, 4, 8],
        "sr_ratios": [8, 4, 2, 1],
        "ffn_expansion": 3.6,
    },
    "xs": {
        "num_blocks": [3, 3, 12, 3],
        "out_channels": [52, 104, 208, 416],
        "stem_width": 16,
        "num_heads": [1, 2, 4, 8],
        "sr_ratios": [8, 4, 2, 1],
        "ffn_expansion": 3.8,
    },
    "small": {
        "num_blocks": [3, 3, 16, 3],
        "out_channels": [64, 128, 256, 512],
        "stem_width": 32,
        "num_heads": [1, 2, 4, 8],
        "sr_ratios": [8, 4, 2, 1],
        "ffn_expansion": 4,
    },
    "big": {
        "num_blocks": [4, 4, 20, 4],
        "out_channels": [76, 152, 304, 608],
        "stem_width": 38,
        "num_heads": [1, 2, 4, 8],
        "sr_ratios": [8, 4, 2, 1],
        "ffn_expansion": 4,
    },
}


def CMTTiny(input_shape=(160, 160, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained=None, **kwargs):
    return CMT(**BLOCK_CONFIGS["tiny"], model_name="cmt_tiny", **locals(), **kwargs)


def CMTXS(input_shape=(192, 192, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained=None, **kwargs):
    return CMT(**BLOCK_CONFIGS["xs"], model_name="cmt_xs", **locals(), **kwargs)


def CMTSmall(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained=None, **kwargs):
    return CMT(**BLOCK_CONFIGS["small"], model_name="cmt_small", **locals(), **kwargs)


def CMTBig(input_shape=(256, 256, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained=None, **kwargs):
    return CMT(**BLOCK_CONFIGS["big"], model_name="cmt_big", **locals(), **kwargs)
