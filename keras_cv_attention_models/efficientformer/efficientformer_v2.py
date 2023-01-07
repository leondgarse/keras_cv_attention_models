import tensorflow as tf
from tensorflow import keras
from keras_cv_attention_models.attention_layers import (
    activation_by_name,
    batchnorm_with_activation,
    conv2d_no_bias,
    ChannelAffine,
    depthwise_conv2d_no_bias,
    drop_block,
    MultiHeadPositionalEmbedding,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

PRETRAINED_DICT = {
    "": {"": {224: ""}},
}


def conv_mhsa_with_multi_head_position(
    inputs,
    num_heads=8,
    key_dim=32,
    strides=1,
    out_shape=None,
    attn_ratio=4,
    use_local_global_query=False,
    use_talking_head=True,
    qkv_bias=True,
    activation="relu",
    name=None,
):
    input_channel = inputs.shape[-1]
    key_dim = key_dim if key_dim > 0 else input_channel // num_heads
    qk_scale = float(1.0 / tf.math.sqrt(tf.cast(key_dim, "float32")))
    out_shape = input_channel if out_shape is None else out_shape
    qk_out = num_heads * key_dim
    value_out = attn_ratio * qk_out
    value_dim = attn_ratio * key_dim

    if strides > 1:
        inputs = depthwise_conv2d_no_bias(inputs, use_bias=True, kernel_size=3, strides=strides, padding="same", name=name and name + "down_sample_")
        inputs = batchnorm_with_activation(inputs, activation=None, name=name and name + "down_sample_")

    kv_blocks = inputs.shape[1] * inputs.shape[2]

    if use_local_global_query:
        pool_query = inputs[:, ::2, ::2]  # nn.AvgPool2d(kernel_size=1, stride=2, padding=0)
        local_query = depthwise_conv2d_no_bias(inputs, use_bias=qkv_bias, kernel_size=3, strides=2, padding="same", name=name and name + "local_query_")
        pre_query = pool_query + local_query
        vv_local_strides = 2
    else:
        pre_query = inputs
        vv_local_strides = 1
    _, query_height, query_width, _ = pre_query.shape

    query = conv2d_no_bias(pre_query, qk_out, use_bias=qkv_bias, kernel_size=1, name=name and name + "query_")
    query = batchnorm_with_activation(query, activation=None, name=name and name + "query_")
    query = tf.transpose(tf.reshape(query, [-1, query_height * query_width, num_heads, key_dim]), [0, 2, 1, 3])  #  [batch, num_heads, hh * ww, key_dim]

    key = conv2d_no_bias(inputs, qk_out, use_bias=qkv_bias, kernel_size=1, name=name and name + "key_")
    key = batchnorm_with_activation(key, activation=None, name=name and name + "key_")
    key = tf.transpose(tf.reshape(key, [-1, kv_blocks, num_heads, key_dim]), [0, 2, 3, 1])  #  [batch, num_heads, key_dim, hh * ww]

    value = conv2d_no_bias(inputs, value_out, use_bias=qkv_bias, kernel_size=1, name=name and name + "value_")
    value = batchnorm_with_activation(value, activation=None, name=name and name + "value_")
    vv_local = depthwise_conv2d_no_bias(value, use_bias=qkv_bias, kernel_size=3, strides=vv_local_strides, padding="same", name=name and name + "value_local_")
    vv_local = batchnorm_with_activation(vv_local, activation=None, name=name and name + "value_local_")
    value = tf.transpose(tf.reshape(value, [-1, kv_blocks, num_heads, value_dim]), [0, 2, 1, 3])  #  [batch, num_heads, hh * ww, value_dim]

    attention_scores = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([query, key]) * qk_scale  # [batch, num_heads, hh * ww, hh * ww]
    # print(f"{query.shape = }, {key.shape = }, {value.shape = }, {attention_scores.shape = }, {query_height = }")
    attention_scores = MultiHeadPositionalEmbedding(query_height=query_height, name=name and name + "pos_emb")(attention_scores)

    if use_talking_head:
        attention_scores = tf.transpose(attention_scores, [0, 2, 3, 1])  # [batch, hh * ww, hh * ww, num_heads]
        attention_scores = conv2d_no_bias(attention_scores, num_heads, use_bias=True, name=name and name + "talking_head_1_")
        attention_scores = keras.layers.Softmax(axis=2, name=name and name + "attention_scores")(attention_scores)  # On previous last dimension
        attention_scores = conv2d_no_bias(attention_scores, num_heads, use_bias=True, name=name and name + "talking_head_2_")
        # attention_scores = keras.layers.Dropout(attn_dropout, name=name and name + "attn_drop")(attention_scores) if attn_dropout > 0 else attention_scores
        attention_scores = tf.transpose(attention_scores, [0, 3, 1, 2])  # [batch, num_heads, hh * ww, hh * ww]
    else:
        attention_scores = keras.layers.Softmax(axis=-1, name=name and name + "attention_scores")(attention_scores)
        # attention_scores = keras.layers.Dropout(attn_dropout, name=name and name + "attn_drop")(attention_scores) if attn_dropout > 0 else attention_scores

    # value = [batch, num_heads, hh * ww, value_dim], attention_output = [batch, num_heads, hh * ww, value_dim]
    attention_output = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([attention_scores, value])
    attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
    attention_output = tf.reshape(attention_output, [-1, query_height, query_width, num_heads * value_dim])
    attention_output += vv_local
    # print(f">>>> {attention_output.shape = }, {attention_scores.shape = }")

    if strides > 1:
        attention_output = keras.layers.UpSampling2D(size=(strides, strides), interpolation="bilinear")(attention_output)

    # [batch, hh, ww, num_heads * value_dim] * [num_heads * value_dim, out] --> [batch, hh, ww, out]
    attention_output = activation_by_name(attention_output, activation=activation, name=name)
    attention_output = conv2d_no_bias(attention_output, out_shape, use_bias=True, kernel_size=1, name=name and name + "output_")
    attention_output = batchnorm_with_activation(attention_output, activation=None, name=name and name + "output_")

    return attention_output


def mlp_block_with_additional_depthwise_conv(inputs, hidden_dim, output_channel=-1, drop_rate=0, activation="gelu", name=""):
    output_channel = output_channel if output_channel > 0 else inputs.shape[-1]
    nn = conv2d_no_bias(inputs, hidden_dim, kernel_size=1, use_bias=True, name=name + "1_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "1_")

    nn = depthwise_conv2d_no_bias(nn, use_bias=True, kernel_size=3, strides=1, padding="same", name=name + "mid_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "mid_")

    nn = conv2d_no_bias(nn, output_channel, kernel_size=1, use_bias=True, name=name + "2_")
    nn = batchnorm_with_activation(nn, activation=None, name=name + "2_")
    nn = keras.layers.Dropout(drop_rate)(nn) if drop_rate > 0 else nn
    return nn


def res_layer_scale_drop_block(short, deep, layer_scale=0, drop_rate=0, name=""):
    deep = ChannelAffine(use_bias=False, weight_init_value=layer_scale, name=name + "gamma")(deep) if layer_scale >= 0 else deep
    deep = drop_block(deep, drop_rate=drop_rate, name=name)
    return keras.layers.Add(name=name + "output")([short, deep])


def down_sample_block(inputs, out_channel, use_attn=False, activation="relu", name=""):
    conv_branch = conv2d_no_bias(inputs, out_channel, kernel_size=3, strides=2, use_bias=True, padding="SAME", name=name)
    conv_branch = batchnorm_with_activation(conv_branch, activation=None, name=name)
    if use_attn:
        fixed_kwargs = {"num_heads": 8, "key_dim": 16, "strides": 1, "attn_ratio": 4, "use_local_global_query": True, "use_talking_head": False}
        attn_branch = conv_mhsa_with_multi_head_position(inputs, **fixed_kwargs, out_shape=out_channel, activation=activation, name=name + "attn_")
        nn = attn_branch + conv_branch
    else:
        nn = conv_branch
    return nn


def EfficientFormerV2(
    num_blocks=[2, 2, 6, 4],
    out_channels=[32, 48, 96, 176],
    mlp_ratios=[4, 4, [4, 3, 3, 3, 4, 4], [4, 3, 3, 4]],
    num_attn_blocks_each_stack=[0, 0, 2, 2],
    stem_width=-1,
    stem_activation=None,
    layer_scale=1e-5,
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="gelu",
    drop_connect_rate=0,
    classifier_activation=None,
    use_distillation=True,
    dropout=0,
    pretrained=None,
    model_name="efficientformer_v2",
    kwargs=None,
):
    inputs = keras.layers.Input(input_shape)
    stem_width = stem_width if stem_width > 0 else out_channels[0]
    stem_activation = stem_activation if stem_activation is not None else activation
    nn = conv2d_no_bias(inputs, stem_width // 2, 3, strides=2, use_bias=True, padding="same", name="stem_1_")
    nn = batchnorm_with_activation(nn, activation=stem_activation, name="stem_1_")
    nn = conv2d_no_bias(nn, stem_width, 3, strides=2, use_bias=True, padding="same", name="stem_2_")
    nn = batchnorm_with_activation(nn, activation=stem_activation, name="stem_2_")

    """ stages """
    total_blocks = sum(num_blocks)
    global_block_id = 0
    for stack_id, (num_block, out_channel) in enumerate(zip(num_blocks, out_channels)):
        stack_name = "stack{}_".format(stack_id + 1)
        if stack_id > 0:
            use_attn = True if stack_id >= 3 else False  # Only the last stack
            nn = down_sample_block(nn, out_channel, use_attn=use_attn, name=stack_name + "downsample_")

        cur_num_attn_blocks = num_attn_blocks_each_stack[stack_id] if isinstance(num_attn_blocks_each_stack, (list, tuple)) else num_attn_blocks_each_stack
        cur_mlp_ratios = mlp_ratios[stack_id] if isinstance(mlp_ratios, (list, tuple)) else mlp_ratios
        for block_id in range(num_block):
            block_name = stack_name + "block{}_".format(block_id + 1)
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            cur_mlp_ratio = cur_mlp_ratios[block_id] if isinstance(cur_mlp_ratios, (list, tuple)) else cur_mlp_ratios
            if block_id > num_block - cur_num_attn_blocks - 1:
                strides = 2 if stack_id == 2 else 1
                attn_out = conv_mhsa_with_multi_head_position(nn, num_heads=8, key_dim=32, strides=strides, activation=activation, name=block_name + "attn_")
                nn = res_layer_scale_drop_block(nn, attn_out, layer_scale=layer_scale, drop_rate=block_drop_rate, name=block_name + "attn_")
            mlp_out = mlp_block_with_additional_depthwise_conv(nn, nn.shape[-1] * cur_mlp_ratio, activation=activation, name=block_name + "mlp_")
            nn = res_layer_scale_drop_block(nn, mlp_out, layer_scale=layer_scale, drop_rate=block_drop_rate, name=block_name + "mlp_")
            global_block_id += 1

    """ output """
    if num_classes > 0:
        nn = batchnorm_with_activation(nn, activation=stem_activation, name="pre_output_")
        nn = keras.layers.GlobalAveragePooling2D()(nn)  # tf.reduce_mean(nn, axis=1)
        if dropout > 0 and dropout < 1:
            nn = keras.layers.Dropout(dropout)(nn)
        out = keras.layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="head")(nn)

        if use_distillation:
            distill = keras.layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="distill_head")(nn)
            out = [out, distill]
    else:
        out = nn

    model = keras.models.Model(inputs, out, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    reload_model_weights(model, PRETRAINED_DICT, "efficientformer", pretrained, MultiHeadPositionalEmbedding)
    return model


def EfficientFormerV2S0(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", use_distillation=True, pretrained="imagenet", **kwargs):
    return EfficientFormerV2(**locals(), model_name="efficientformer_v2_s0", **kwargs)


def EfficientFormerV2S1(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", use_distillation=True, pretrained="imagenet", **kwargs):
    num_blocks = [3, 3, 9, 6]
    out_channels = [32, 48, 120, 224]
    mlp_ratios = [4, 4, [4, 4, 3, 3, 3, 3, 4, 4, 4], [4, 4, 3, 3, 4, 4]]
    return EfficientFormerV2(**locals(), model_name="efficientformer_v2_s1", **kwargs)


def EfficientFormerV2S2(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", use_distillation=True, pretrained="imagenet", **kwargs):
    num_blocks = [4, 4, 12, 8]
    out_channels = [32, 64, 144, 288]
    mlp_ratios = [4, 4, [4, 4, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4], [4, 4, 3, 3, 3, 3, 4, 4]]
    num_attn_blocks_each_stack = [0, 0, 4, 4]
    return EfficientFormerV2(**locals(), model_name="efficientformer_v2_s2", **kwargs)


def EfficientFormerV2L(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", use_distillation=True, pretrained="imagenet", **kwargs):
    num_blocks = [5, 5, 15, 10]
    out_channels = [40, 80, 192, 384]
    mlp_ratios = [4, 4, [4, 4, 4, 4, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4], [4, 4, 4, 3, 3, 3, 3, 4, 4, 4]]
    num_attn_blocks_each_stack = [0, 0, 6, 6]
    return EfficientFormerV2(**locals(), model_name="efficientformer_v2_l", **kwargs)
