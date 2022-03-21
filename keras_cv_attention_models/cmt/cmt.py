import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from keras_cv_attention_models.attention_layers import (
    batchnorm_with_activation,
    conv2d_no_bias,
    depthwise_conv2d_no_bias,
    drop_block,
    layer_norm,
    output_block,
    MultiHeadRelativePositionalEmbedding,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

PRETRAINED_DICT = {"cmt_tiny": {"imagenet": {160: "72402495cb42314cedd1e2714f2de893"}}}


def light_mhsa_with_multi_head_relative_position_embedding(
    inputs, num_heads=4, key_dim=0, sr_ratio=1, out_shape=None, out_weight=True, out_bias=False, attn_dropout=0, name=""
):
    _, hh, ww, cc = inputs.shape
    key_dim = key_dim if key_dim > 0 else cc // num_heads
    qk_scale = 1.0 / tf.math.sqrt(tf.cast(key_dim, inputs.dtype))
    out_shape = cc if out_shape is None or not out_weight else out_shape
    emb_dim = num_heads * key_dim

    query = keras.layers.Dense(emb_dim, use_bias=False, name=name and name + "query")(inputs) * qk_scale
    # print(f">>>> {inputs.shape = }, {query.shape = }, {sr_ratio = }")
    # query = [batch, num_heads, hh * ww, key_dim]
    query = tf.transpose(tf.reshape(query, [-1, inputs.shape[1] * inputs.shape[2], num_heads, key_dim]), [0, 2, 1, 3])

    if sr_ratio > 1:
        key_value = depthwise_conv2d_no_bias(inputs, kernel_size=sr_ratio, strides=sr_ratio, name=name + "kv_sr_")
        key_value = layer_norm(key_value, name=name + "kv_sr_")
        # key_value = keras.layers.AvgPool2D(sr_ratio, strides=sr_ratio, name=name + "kv_sr_")(inputs)
    else:
        key_value = inputs
    # key_value = [batch, num_heads, hh, ww, kv_kernel * kv_kernel, key_dim * 2]
    key_value = keras.layers.Dense(emb_dim * 2, use_bias=False, name=name and name + "key_value")(key_value)
    # key_value = conv2d_no_bias(inputs, emb_dim * 2, kernel_size=sr_ratio, strides=sr_ratio, use_bias=False, name=name + "key_value")
    _, kv_hh, kv_ww, _ = key_value.shape
    # print(f">>>> {key_value.shape = }")

    # dim, head, kv
    key_value = tf.reshape(key_value, [-1, kv_hh * kv_ww, key_dim, num_heads, 2])
    key = tf.transpose(key_value[:, :, :, :, 0], [0, 3, 2, 1])  # [batch, num_heads, key_dim, hh * ww]
    value = tf.transpose(key_value[:, :, :, :, 1], [0, 3, 1, 2])  # [batch, num_heads, hh * ww, key_dim]
    # kv, head, dim
    # key, value = tf.split(key_value, 2, axis=-1)
    # key = tf.transpose(tf.reshape(key, [-1, kv_hh * kv_ww, num_heads, key_dim]), [0, 2, 3, 1]) # [batch, num_heads, key_dim, hh * ww]
    # value = tf.transpose(tf.reshape(value, [-1, kv_hh * kv_ww, num_heads, key_dim]), [0, 2, 1, 3]) # [batch, num_heads, hh * ww, key_dim]

    # print(f">>>> {attn_query.shape = }, {key.shape = }, {value.shape = }, {kv_inp.shape = }, {pos_query.shape = }")
    # attention_scores = [batch, num_heads, hh * ww, kv_hh * kv_ww]
    attention_scores = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([query, key])
    attention_scores = MultiHeadRelativePositionalEmbedding(with_cls_token=False, attn_height=hh, name=name and name + "pos_emb")(attention_scores)
    attention_scores = keras.layers.Softmax(axis=-1, name=name and name + "attention_scores")(attention_scores)

    if attn_dropout > 0:
        attention_scores = keras.layers.Dropout(attn_dropout, name=name and name + "attn_drop")(attention_scores)
    # value = [batch, num_heads, key_hh * key_ww, key_dim]
    attention_output = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([attention_scores, value])
    attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
    attention_output = tf.reshape(attention_output, [-1, inputs.shape[1], inputs.shape[2], num_heads * key_dim])
    # print(f">>>> {attention_output.shape = }, {attention_scores.shape = }")

    if out_weight:
        # [batch, hh, ww, num_heads * key_dim] * [num_heads * key_dim, out] --> [batch, hh, ww, out]
        attention_output = keras.layers.Dense(out_shape, use_bias=out_bias, name=name and name + "output")(attention_output)
    return attention_output


def inverted_residual_feed_forward(inputs, expansion=4, activation="gelu", name=""):
    """ IRFFN(X) = Conv(F(Conv(X))), F(X) = DWConv(X) + X """
    in_channel = inputs.shape[-1]
    expanded = conv2d_no_bias(inputs, int(in_channel * expansion), kernel_size=1, use_bias=True, name=name + "1_")
    expanded = batchnorm_with_activation(expanded, activation=activation, act_first=True, name=name + "1_")

    dw = depthwise_conv2d_no_bias(expanded, kernel_size=3, padding="SAME", use_bias=True, name=name)
    dw = keras.layers.Add(name=name + "dw_add")([expanded, dw])
    dw = batchnorm_with_activation(dw, activation=activation, act_first=True, name=name + "2_")

    pw = conv2d_no_bias(dw, in_channel, kernel_size=1, use_bias=True, name=name + "3_")
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
    attn = light_mhsa_with_multi_head_relative_position_embedding(attn, num_heads=num_heads, sr_ratio=sr_ratio, name=name + "light_mhsa_")
    attn_out = keras.layers.Add(name=name + "attn_out")([lpu_out, attn])

    """ inverted residual feed forward """
    ffn = layer_norm(attn_out, name=name + "ffn_")
    ffn = inverted_residual_feed_forward(ffn, expansion=expansion, activation=activation, name=name + "ffn_")
    ffn = drop_block(ffn, drop_rate=drop_rate)
    ffn_out = keras.layers.Add(name=name + "ffn_output")([attn_out, ffn])

    return ffn_out


def cmt_stem(inputs, stem_width, activation="gelu", name="", **kwargs):
    nn = conv2d_no_bias(inputs, stem_width, kernel_size=3, strides=2, padding="same", use_bias=True, name=name + "1_")
    nn = batchnorm_with_activation(nn, activation=activation, act_first=True, name=name + "1_")
    nn = conv2d_no_bias(nn, stem_width, kernel_size=3, strides=1, padding="same", use_bias=True, name=name + "2_")
    nn = batchnorm_with_activation(nn, activation=activation, act_first=True, name=name + "2_")
    nn = conv2d_no_bias(nn, stem_width, kernel_size=3, strides=1, padding="same", use_bias=True, name=name + "3_")
    nn = batchnorm_with_activation(nn, activation=activation, act_first=True, name=name + "3_")
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
    output_num_features=1280,
    dropout=0,
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
        stage_name = "stack{}_".format(stack_id + 1)
        nn = conv2d_no_bias(nn, out_channel, kernel_size=2, strides=2, use_bias=True, name=stage_name + "down_sample")
        nn = layer_norm(nn, name=stage_name)
        for block_id in range(num_block):
            name = stage_name + "block{}_".format(block_id + 1)
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            global_block_id += 1
            nn = cmt_block(nn, num_head, sr_ratio, ffn_expansion, activation=activation, drop_rate=block_drop_rate, name=name)

    nn = output_block(nn, output_num_features, activation, num_classes, dropout, classifier_activation, act_first=True)
    model = keras.models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    reload_model_weights(model, PRETRAINED_DICT, "cmt", pretrained, MultiHeadRelativePositionalEmbedding)
    return model


def CMTTiny(input_shape=(160, 160, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 2, 10, 2]
    out_channels = [46, 92, 184, 368]
    stem_width = 16
    ffn_expansion = 3.6
    return CMT(**locals(), model_name="cmt_tiny", **kwargs)


def CMTXS(input_shape=(192, 192, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained=None, **kwargs):
    num_blocks = [3, 3, 12, 3]
    out_channels = [52, 104, 208, 416]
    stem_width = 16
    ffn_expansion = 3.8
    return CMT(**locals(), model_name="cmt_xs", **kwargs)


def CMTSmall(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained=None, **kwargs):
    num_blocks = [3, 3, 16, 3]
    out_channels = [64, 128, 256, 512]
    stem_width = 32
    ffn_expansion = 4
    return CMT(**locals(), model_name="cmt_small", **kwargs)


def CMTBig(input_shape=(256, 256, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained=None, **kwargs):
    num_blocks = [4, 4, 20, 4]
    out_channels = [76, 152, 304, 608]
    stem_width = 38
    ffn_expansion = 4
    return CMT(**locals(), model_name="cmt_big", **kwargs)
