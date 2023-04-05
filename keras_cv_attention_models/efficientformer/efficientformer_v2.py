from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, functional, models, is_channels_last
from keras_cv_attention_models.attention_layers import (
    activation_by_name,
    add_with_layer_scale_and_drop_block,
    batchnorm_with_activation,
    conv2d_no_bias,
    ChannelAffine,
    depthwise_conv2d_no_bias,
    drop_block,
    MultiHeadPositionalEmbedding,
    qkv_to_multi_head_channels_last_format,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

PRETRAINED_DICT = {
    "efficientformer_v2_l": {"imagenet": {224: "3792a3ea9eb9d1e818d4a36c00e422a5"}},
    "efficientformer_v2_s0": {"imagenet": {224: "88de5ba4a8effd887d53df3020ba8433"}},
    "efficientformer_v2_s1": {"imagenet": {224: "a0843565d0d01004604d52b0e2ddfa0a"}},
    "efficientformer_v2_s2": {"imagenet": {224: "07c358cc0f8ea8a02722673bd38bfe97"}},
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
    activation="gelu",
    name=None,
):
    height_axis, width_axis, channel_aixs = (1, 2, 3) if is_channels_last() else (2, 3, 1)
    height, width, input_channel = inputs.shape[height_axis], inputs.shape[width_axis], inputs.shape[channel_aixs]
    key_dim = key_dim if key_dim > 0 else input_channel // num_heads
    # qk_scale = float(1.0 / tf.math.sqrt(tf.cast(key_dim, "float32")))
    qk_scale = 1.0 / (float(key_dim) ** 0.5)
    out_shape = input_channel if out_shape is None else out_shape
    qk_out = num_heads * key_dim
    value_out = attn_ratio * qk_out
    value_dim = attn_ratio * key_dim

    if strides > 1:
        should_cut_height, should_cut_width = height % 2, width % 2  # keep shape same with inputs after later UpSampling2D
        inputs = depthwise_conv2d_no_bias(inputs, use_bias=True, kernel_size=3, strides=strides, padding="same", name=name and name + "down_sample_")
        inputs = batchnorm_with_activation(inputs, activation=None, name=name and name + "down_sample_")

    kv_blocks = height * width

    if use_local_global_query:
        # pool_query = layers.AvgPool2D(pool_size=1, strides=2)(inputs)
        # pool_query = inputs[:, ::2, ::2] if is_channels_last() else inputs[:, :, ::2, ::2] # nn.AvgPool2d(kernel_size=1, stride=2, padding=0)
        pool_query = layers.AvgPool2D(pool_size=1, strides=2)(inputs)
        local_query = depthwise_conv2d_no_bias(inputs, use_bias=qkv_bias, kernel_size=3, strides=2, padding="same", name=name and name + "local_query_")
        pre_query = pool_query + local_query
        vv_local_strides = 2
    else:
        pre_query = inputs
        vv_local_strides = 1
    query_height, query_width = pre_query.shape[1:-1] if is_channels_last() else pre_query.shape[2:]

    query = conv2d_no_bias(pre_query, qk_out, use_bias=qkv_bias, kernel_size=1, name=name and name + "query_")
    query = batchnorm_with_activation(query, activation=None, name=name and name + "query_")

    key = conv2d_no_bias(inputs, qk_out, use_bias=qkv_bias, kernel_size=1, name=name and name + "key_")
    key = batchnorm_with_activation(key, activation=None, name=name and name + "key_")

    value = conv2d_no_bias(inputs, value_out, use_bias=qkv_bias, kernel_size=1, name=name and name + "value_")
    value = batchnorm_with_activation(value, activation=None, name=name and name + "value_")
    vv_local = depthwise_conv2d_no_bias(value, use_bias=qkv_bias, kernel_size=3, strides=vv_local_strides, padding="same", name=name and name + "value_local_")
    vv_local = batchnorm_with_activation(vv_local, activation=None, name=name and name + "value_local_")

    # query = functional.transpose(
    #     functional.reshape(query, [-1, query_height * query_width, num_heads, key_dim]), [0, 2, 1, 3]
    # )  #  [batch, num_heads, hh * ww, key_dim]
    # key = functional.transpose(functional.reshape(key, [-1, kv_blocks, num_heads, key_dim]), [0, 2, 3, 1])  #  [batch, num_heads, key_dim, hh * ww]
    # value = functional.transpose(functional.reshape(value, [-1, kv_blocks, num_heads, value_dim]), [0, 2, 1, 3])  #  [batch, num_heads, hh * ww, value_dim]
    query, key, value = qkv_to_multi_head_channels_last_format(query, key, value, num_heads=num_heads)

    # attention_scores = layers.Lambda(lambda xx: functional.matmul(xx[0], xx[1]))([query, key]) * qk_scale  # [batch, num_heads, hh * ww, hh * ww]
    # print(f"{query.shape = }, {key.shape = }, {value.shape = }, {attention_scores.shape = }, {query_height = }")
    attention_scores = (query @ key) * qk_scale
    attention_scores = MultiHeadPositionalEmbedding(query_height=query_height, name=name and name + "pos_emb")(attention_scores)

    if use_talking_head:
        # if channels_last, attention_scores [batch, num_heads, hh * ww, hh * ww] -> [batch, hh * ww, hh * ww, num_heads]
        attention_scores = functional.transpose(attention_scores, [0, 2, 3, 1]) if is_channels_last() else attention_scores
        attention_scores = conv2d_no_bias(attention_scores, num_heads, use_bias=True, name=name and name + "talking_head_1_")
        # On previous last dimension
        attention_scores = layers.Softmax(axis=2 if is_channels_last() else -1, name=name and name + "attention_scores")(attention_scores)
        attention_scores = conv2d_no_bias(attention_scores, num_heads, use_bias=True, name=name and name + "talking_head_2_")
        # if channels_last, attention_scores [batch, hh * ww, hh * ww, num_heads] -> [batch, num_heads, hh * ww, hh * ww]
        attention_scores = functional.transpose(attention_scores, [0, 3, 1, 2]) if is_channels_last() else attention_scores
    else:
        attention_scores = layers.Softmax(axis=-1, name=name and name + "attention_scores")(attention_scores)
        # attention_scores = layers.Dropout(attn_dropout, name=name and name + "attn_drop")(attention_scores) if attn_dropout > 0 else attention_scores

    # value = [batch, num_heads, hh * ww, value_dim], attention_output = [batch, num_heads, hh * ww, value_dim]
    # attention_output = layers.Lambda(lambda xx: functional.matmul(xx[0], xx[1]))([attention_scores, value])
    attention_output = attention_scores @ value
    if is_channels_last():
        attention_output = functional.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = functional.reshape(attention_output, [-1, query_height, query_width, num_heads * value_dim])
    else:
        attention_output = functional.transpose(attention_output, perm=[0, 1, 3, 2])
        attention_output = functional.reshape(attention_output, [-1, num_heads * value_dim, query_height, query_width])
    attention_output += vv_local
    # print(f">>>> {attention_output.shape = }, {attention_scores.shape = }")

    if strides > 1:
        attention_output = layers.UpSampling2D(size=(strides, strides), interpolation="bilinear")(attention_output)
        if should_cut_height > 0 or should_cut_width > 0:
            keep_height, keep_width = attention_output.shape[height_axis] - should_cut_height, attention_output.shape[width_axis] - should_cut_width
            attention_output = attention_output[:, :keep_height, :keep_width] if is_channels_last() else attention_output[:, :, :keep_height, :keep_width]

    # [batch, hh, ww, num_heads * value_dim] * [num_heads * value_dim, out] --> [batch, hh, ww, out]
    attention_output = activation_by_name(attention_output, activation=activation, name=name)
    attention_output = conv2d_no_bias(attention_output, out_shape, use_bias=True, kernel_size=1, name=name and name + "output_")
    attention_output = batchnorm_with_activation(attention_output, activation=None, name=name and name + "output_")

    return attention_output


def mlp_block_with_additional_depthwise_conv(inputs, hidden_dim, output_channel=-1, drop_rate=0, activation="gelu", name=""):
    output_channel = output_channel if output_channel > 0 else inputs.shape[-1 if is_channels_last() else 1]
    nn = conv2d_no_bias(inputs, hidden_dim, kernel_size=1, use_bias=True, name=name + "1_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "1_")

    nn = depthwise_conv2d_no_bias(nn, use_bias=True, kernel_size=3, strides=1, padding="same", name=name + "mid_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "mid_")

    nn = conv2d_no_bias(nn, output_channel, kernel_size=1, use_bias=True, name=name + "2_")
    nn = batchnorm_with_activation(nn, activation=None, name=name + "2_")
    nn = layers.Dropout(drop_rate)(nn) if drop_rate > 0 else nn
    return nn


def down_sample_block(inputs, out_channel, use_attn=False, activation="gelu", name=""):
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
    # Regard input_shape as force using original shape if len(input_shape) == 4,
    # else assume channel dimention is the one with min value in input_shape, and put it first or last regarding image_data_format
    input_shape = backend.align_input_shape_by_image_data_format(input_shape)
    inputs = layers.Input(input_shape)
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
            nn = down_sample_block(nn, out_channel, use_attn=use_attn, activation=activation, name=stack_name + "downsample_")

        cur_num_attn_blocks = num_attn_blocks_each_stack[stack_id] if isinstance(num_attn_blocks_each_stack, (list, tuple)) else num_attn_blocks_each_stack
        cur_mlp_ratios = mlp_ratios[stack_id] if isinstance(mlp_ratios, (list, tuple)) else mlp_ratios
        for block_id in range(num_block):
            block_name = stack_name + "block{}_".format(block_id + 1)
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            cur_mlp_ratio = cur_mlp_ratios[block_id] if isinstance(cur_mlp_ratios, (list, tuple)) else cur_mlp_ratios
            if block_id > num_block - cur_num_attn_blocks - 1:
                strides = 2 if stack_id == 2 else 1
                attn_out = conv_mhsa_with_multi_head_position(nn, num_heads=8, key_dim=32, strides=strides, activation=activation, name=block_name + "attn_")
                nn = add_with_layer_scale_and_drop_block(nn, attn_out, layer_scale=layer_scale, drop_rate=block_drop_rate, name=block_name + "attn_")
            mlp_out = mlp_block_with_additional_depthwise_conv(nn, out_channel * cur_mlp_ratio, activation=activation, name=block_name + "mlp_")
            nn = add_with_layer_scale_and_drop_block(nn, mlp_out, layer_scale=layer_scale, drop_rate=block_drop_rate, name=block_name + "mlp_")
            global_block_id += 1

    """ output """
    if num_classes > 0:
        nn = batchnorm_with_activation(nn, activation=None, name="pre_output_")
        nn = layers.GlobalAveragePooling2D()(nn)  # tf.reduce_mean(nn, axis=1)
        if dropout > 0 and dropout < 1:
            nn = layers.Dropout(dropout)(nn)
        out = layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="head")(nn)

        if use_distillation:
            distill = layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="distill_head")(nn)
            out = [out, distill]
    else:
        out = nn

    model = models.Model(inputs, out, name=model_name)
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
