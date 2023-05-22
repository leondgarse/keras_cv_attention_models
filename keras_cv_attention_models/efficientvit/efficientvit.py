from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, functional, models, image_data_format
from keras_cv_attention_models.attention_layers import (
    activation_by_name,
    batchnorm_with_activation,
    conv2d_no_bias,
    depthwise_conv2d_no_bias,
    drop_block,
    se_module,
    mlp_block,
    window_attention,
    MultiHeadPositionalEmbedding,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

PRETRAINED_DICT = {}

def grouped_mhsa_with_multi_head_position(inputs, num_heads, key_dim=-1, output_dim=-1, kernel_sizes=5, activation="relu", name=None):
    height, width = inputs.shape[1:-1] if image_data_format() == "channels_last" else inputs.shape[2:]
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    key_dim = key_dim if key_dim > 0 else inputs.shape[channel_axis] // num_heads
    output_dim = output_dim if output_dim > 0 else inputs.shape[channel_axis]
    value_dim = output_dim // num_heads
    qk_scale = 1.0 / (float(key_dim) ** 0.5)
    # embed_dim = key_dim * num_heads

    qkv_dim = key_dim + key_dim + value_dim
    kernel_sizes = kernel_sizes if isinstance(kernel_sizes, (list, tuple)) else [kernel_sizes] * num_heads
    head_inputs = functional.split(inputs, num_heads, axis=channel_axis)
    output = []
    for id, (head_input, kernel_size) in enumerate(zip(head_inputs, kernel_sizes)):
        head_input = head_input if id == 0 else (head_input + output[-1])
        cur_name = name and name + "{}_".format(id + 1)
        qkv = conv2d_no_bias(head_input, qkv_dim, name=cur_name and cur_name + "qkv_")
        qkv = batchnorm_with_activation(qkv, activation=None, name=cur_name and cur_name + "qkv_")
        qq, kk, vv = functional.split(qkv, [key_dim, key_dim, value_dim], axis=channel_axis)

        qq = depthwise_conv2d_no_bias(qq, kernel_size=kernel_size, padding="SAME", name=cur_name and cur_name + "query_")
        qq = batchnorm_with_activation(qq, activation=None, name=cur_name and cur_name + "query_")

        if image_data_format() == "channels_last":
            qq = functional.reshape(qq, [-1, qq.shape[1] * qq.shape[2], qq.shape[-1]])
            kk = functional.transpose(functional.reshape(kk, [-1, kk.shape[1] * kk.shape[2], kk.shape[-1]]), [0, 2, 1])
            vv = functional.reshape(vv, [-1, vv.shape[1] * vv.shape[2], vv.shape[-1]])
        else:
            qq = functional.transpose(functional.reshape(qq, [-1, qq.shape[1], qq.shape[2] * qq.shape[3]]), [0, 2, 1])
            kk = functional.reshape(kk, [-1, kk.shape[1], kk.shape[2] * kk.shape[3]])
            vv = functional.reshape(vv, [-1, vv.shape[1], vv.shape[2] * vv.shape[3]])

        # attention_scores = layers.Lambda(lambda xx: functional.matmul(xx[0], xx[1]))([query, key]) * qk_scale  # [batch, num_heads, key_dim, key_dim]
        attention_scores = (qq @ kk) * qk_scale
        print(f"{height = }, {width = }")
        attention_scores = MultiHeadPositionalEmbedding(query_height=height, name=cur_name and cur_name + "attn_pos")(attention_scores)
        attention_scores = layers.Softmax(axis=-1, name=cur_name and cur_name + "attention_scores")(attention_scores)
        # attention_scores = layers.Dropout(attn_dropout, name=name and name + "attn_drop")(attention_scores) if attn_dropout > 0 else attention_scores

        if image_data_format() == "channels_last":
            attention_output = attention_scores @ vv
            head_output = functional.reshape(attention_output, [-1, height, width, value_dim])
        else:
            attention_output = vv @ functional.transpose(attention_scores, [0, 2, 1])
            head_output = functional.reshape(attention_output, [-1, value_dim, height, width])
        output.append(head_output)
    output = functional.concat(output, axis=channel_axis)

    output = activation_by_name(output, activation=activation, name=name)
    output = conv2d_no_bias(output, output_dim, name=name and name + "out")
    output = batchnorm_with_activation(output, activation=None, zero_gamma=True, name=name and name + "out_")
    return output

def res_depthwise_ffn(inputs, mlp_ratio=2, drop_rate=0, activation="relu", name=""):
    input_channel = inputs.shape[-1 if image_data_format() == "channels_last" else 1]
    dw = depthwise_conv2d_no_bias(inputs, kernel_size=3, strides=1, padding="SAME", name=name + "dw_")
    dw = batchnorm_with_activation(dw, activation=None, zero_gamma=True, name=name + "dw_")
    dw = inputs + drop_block(dw, drop_rate)

    ffn = conv2d_no_bias(dw, input_channel * mlp_ratio, name=name + "ffn_1_")
    ffn = batchnorm_with_activation(ffn, activation=activation, zero_gamma=True, name=name + "ffn_1_")
    ffn = conv2d_no_bias(ffn, input_channel, name=name + "ffn_2_")
    ffn = batchnorm_with_activation(ffn, activation=None, zero_gamma=True, name=name + "ffn_2_")
    ffn = drop_block(ffn, drop_rate)
    return layers.Add(name=name + "output")([dw, ffn])

def attn_block(inputs, window_size=7, num_heads=4, key_dim=16, kernel_sizes=5, mlp_ratio=2, drop_rate=0, activation="relu", name=""):
    height, width = inputs.shape[1:-1] if image_data_format() == "channels_last" else inputs.shape[2:]
    pre = res_depthwise_ffn(inputs, mlp_ratio=mlp_ratio, drop_rate=drop_rate, activation=activation, name=name + "pre_")
    # nn = inputs if image_data_format() == "channels_last" else layers.Permute([2, 3, 1])(inputs)  # channels_first -> channels_last
    attn_kw = {"key_dim": key_dim, "kernel_sizes": kernel_sizes, "activation": activation}
    if height <= window_size and width <= window_size:
        attn = grouped_mhsa_with_multi_head_position(pre, num_heads=num_heads, **attn_kw, name=name + "attn_")
    else:
        attn = window_attention(pre, window_size, num_heads=num_heads, attention_block=grouped_mhsa_with_multi_head_position, **attn_kw, name=name + "attn_")
    attn = pre + drop_block(attn, drop_rate)
    # nn = nn if image_data_format() == "channels_last" else layers.Permute([3, 1, 2])(nn)  # channels_last, channels_first

    return res_depthwise_ffn(attn, mlp_ratio=mlp_ratio, drop_rate=drop_rate, activation=activation, name=name + "post_")


def down_sample_block(inputs, out_channel, hiddem_raio=4, activation="relu", name=""):
    input_channel = inputs.shape[-1 if image_data_format() == "channels_last" else 1]
    nn = conv2d_no_bias(inputs, input_channel * hiddem_raio, kernel_size=1, strides=1, padding="SAME", name=name)
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "pw_")
    nn = depthwise_conv2d_no_bias(nn, kernel_size=3, strides=2, padding="SAME", name=name)
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "dw_")
    nn = se_module(nn, activation=activation, name=name + "se_")
    nn = conv2d_no_bias(nn, out_channel, kernel_size=1, strides=1, padding="SAME", name=name + "out_")
    nn = batchnorm_with_activation(nn, activation=None, name=name + "out_")
    return nn


def EfficientViT(
    num_blocks=[1, 2, 3],
    out_channels=[64, 128, 192],
    window_size=7,
    num_heads=4,
    key_dim=16,
    kernel_sizes=5,
    mlp_ratio=2,
    stem_width=-1,
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="relu",
    drop_connect_rate=0,
    classifier_activation="softmax",
    use_distillation=False,  # not provided in pretrained weights
    dropout=0,
    pretrained=None,
    model_name="efficient_vit",
    kwargs=None,
):
    # Regard input_shape as force using original shape if len(input_shape) == 4,
    # else assume channel dimension is the one with min value in input_shape, and put it first or last regarding image_data_format
    input_shape = backend.align_input_shape_by_image_data_format(input_shape)
    inputs = layers.Input(input_shape)
    stem_width = stem_width if stem_width > 0 else out_channels[0]
    nn = conv2d_no_bias(inputs, stem_width // 8, 3, strides=2, padding="same", name="stem_1_")
    nn = batchnorm_with_activation(nn, activation=activation, name="stem_1_")
    nn = conv2d_no_bias(nn, stem_width // 4, 3, strides=2, padding="same", name="stem_2_")
    nn = batchnorm_with_activation(nn, activation=activation, name="stem_2_")
    nn = conv2d_no_bias(nn, stem_width // 2, 3, strides=2, padding="same", name="stem_3_")
    nn = batchnorm_with_activation(nn, activation=activation, name="stem_3_")
    nn = conv2d_no_bias(nn, stem_width, 3, strides=2, padding="same", name="stem_4_")
    nn = batchnorm_with_activation(nn, activation=None, name="stem_4_")

    """ stages """
    total_blocks = sum(num_blocks)
    global_block_id = 0
    for stack_id, (num_block, out_channel) in enumerate(zip(num_blocks, out_channels)):
        stack_name = "stack{}_".format(stack_id + 1)
        if stack_id > 0:
            nn = res_depthwise_ffn(nn, mlp_ratio=mlp_ratio, activation=activation, name=stack_name + "pre_")
            nn = down_sample_block(nn, out_channel, activation=activation, name=stack_name + "downsample_")
            nn = res_depthwise_ffn(nn, mlp_ratio=mlp_ratio, activation=activation, name=stack_name + "post_")

        for block_id in range(num_block):
            block_name = stack_name + "block{}_".format(block_id + 1)
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            nn = attn_block(nn, window_size, num_heads, key_dim, kernel_sizes, mlp_ratio, drop_rate=block_drop_rate, activation=activation, name=block_name)
            global_block_id += 1

    """ output """
    if num_classes > 0:
        nn = layers.GlobalAveragePooling2D(keepdims=True)(nn)  # tf.reduce_mean(nn, axis=1)
        if dropout > 0 and dropout < 1:
            nn = layers.Dropout(dropout)(nn)
        out = batchnorm_with_activation(nn, activation=None, name="pre_output_")
        out = layers.Flatten()(out)
        out = layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="head")(out)

        if use_distillation:
            distill_out = batchnorm_with_activation(nn, activation=None, name="distill_output_")
            distill_out = layers.Flatten()(distill_out)
            distill_out = layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="distill_head")(distill_out)
            out = [out, distill_out]
    else:
        out = nn

    model = models.Model(inputs, out, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    reload_model_weights(model, PRETRAINED_DICT, "efficient_vit", pretrained, MultiHeadPositionalEmbedding)
    return model


def EfficientViT_M0(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return EfficientViT(**locals(), model_name="efficient_vit_m0", **kwargs)


def EfficientViT_M1(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    out_channels = [128, 144, 192]
    num_heads = [2, 3, 3]
    kernel_sizes = [7, 5, 3, 3]
    return EfficientViT(**locals(), model_name="efficient_vit_m1", **kwargs)


def EfficientViT_M2(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    out_channels = [128, 192, 224]
    num_heads = [4, 3, 2]
    kernel_sizes = [7, 5, 3, 3]
    return EfficientViT(**locals(), model_name="efficient_vit_m2", **kwargs)


def EfficientViT_M3(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    out_channels = [128, 240, 320]
    num_heads = [4, 3, 4]
    kernel_sizes = 5
    return EfficientViT(**locals(), model_name="efficient_vit_m3", **kwargs)


def EfficientViT_M4(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    out_channels = [128, 256, 384]
    num_heads = 4
    kernel_sizes = [7, 5, 3, 3]
    return EfficientViT(**locals(), model_name="efficient_vit_m4", **kwargs)


def EfficientViT_M5(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    out_channels = [192, 288, 384]
    num_heads = [3, 3, 4]
    kernel_sizes = [7, 5, 3, 3]
    return EfficientViT(**locals(), model_name="efficient_vit_m5", **kwargs)
