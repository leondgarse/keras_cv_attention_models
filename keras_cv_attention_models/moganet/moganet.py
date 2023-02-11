from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models, functional, image_data_format
from keras_cv_attention_models.attention_layers import (
    activation_by_name,
    add_with_layer_scale_and_drop_block,
    batchnorm_with_activation,
    conv2d_no_bias,
    ChannelAffine,
    depthwise_conv2d_no_bias,
    output_block,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

PRETRAINED_DICT = {
    "moganet_xtiny": {"imagenet": "b1a7a1b77777cd8fdc8b5b5333e49215"},
    "moganet_tiny": {"imagenet": {224: "2ecd6f4552fb5bdbd0845c6a49bb67a9", 256: "6be0f1b79d00ba535412c1d1d3c1a71f"}},
    "moganet_small": {"imagenet": "8b090a8058304bdbacdc70896ecb25cd"},
    "moganet_base": {"imagenet": "3aa0d6fccc312bb7674495b7c0c8153e"},
    "moganet_large": {"imagenet": "1714094064c837afa3c6f0304563b496"},
}


def feature_decompose(inputs, use_pool=False, activation="gelu", name=""):
    channel_axis = -1 if backend.image_data_format() == "channels_last" else 1
    if use_pool:
        decomposed = layers.GlobalAveragePooling2D(keepdims=True, name=name + "pool")(inputs)
    else:
        decomposed = conv2d_no_bias(inputs, 1, use_bias=True, name=name)
    decomposed = activation_by_name(decomposed, activation=activation, name=name)
    decomposed = ChannelAffine(use_bias=False, weight_init_value=1e-5, axis=channel_axis, name=name + "affine")(inputs - decomposed)
    return inputs + decomposed


def multi_order_depthwise_conv2d(inputs, dilations=[1, 2, 3], channel_split=[1, 3, 4], name=""):
    channel_axis = -1 if backend.image_data_format() == "channels_last" else 1
    input_channel = inputs.shape[channel_axis]
    channel_split = [-1] + [int(input_channel * ii / sum(channel_split)) for ii in channel_split[1:]]
    paddings = [(1 + 4 * dilations[0]) // 2, (1 + 4 * dilations[1]) // 2, (1 + 6 * dilations[2]) // 2]

    # print(f"{inputs.shape = }, {input_channel = }, {channel_split = }")
    first = depthwise_conv2d_no_bias(inputs, kernel_size=5, padding=paddings[0], use_bias=True, dilation_rate=dilations[0], name=name + "first_")
    first, second, third = functional.split(first, channel_split, axis=channel_axis)
    second = depthwise_conv2d_no_bias(second, kernel_size=5, padding=paddings[1], use_bias=True, dilation_rate=dilations[1], name=name + "second_")
    third = depthwise_conv2d_no_bias(third, kernel_size=7, padding=paddings[2], use_bias=True, dilation_rate=dilations[2], name=name + "third_")
    # print(f"{first.shape = }, {second.shape = }, {third.shape = }")

    out = functional.concat([first, second, third], axis=channel_axis)
    out = conv2d_no_bias(out, input_channel, use_bias=True, name=name + "out_")
    return out


def moga_block(inputs, mlp_ratio=4, layer_scale=0, drop_rate=0, attn_activation="swish", activation="gelu", name=""):
    input_channel = inputs.shape[-1 if backend.image_data_format() == "channels_last" else 1]

    """ attention swish """
    pre_norm = batchnorm_with_activation(inputs, activation=None, name=name + "pre_attn_")
    nn = conv2d_no_bias(pre_norm, input_channel, use_bias=True, name=name + "pre_attn_")  # proj_1
    nn = feature_decompose(nn, use_pool=True, activation=None, name=name + "pre_attn_decompose_")
    nn = activation_by_name(nn, activation=attn_activation, name=name + "pre_attn_")

    gate = conv2d_no_bias(nn, input_channel, use_bias=True, name=name + "attn_gate_")
    gate = activation_by_name(gate, activation=attn_activation, name=name + "attn_gate_")
    value = multi_order_depthwise_conv2d(nn, name=name + "attn_value_")  # MultiOrderDWConv
    value = activation_by_name(value, activation=attn_activation, name=name + "attn_value_")
    gate_value = layers.Multiply(name=name + "attn_gate_value")([gate, value])
    gate_value = conv2d_no_bias(gate_value, input_channel, use_bias=True, name=name + "attn_gate_value_")  # proj_2
    nn = layers.Add(name=name + "attn_out")([pre_norm, gate_value])

    attn_out = add_with_layer_scale_and_drop_block(inputs, nn, layer_scale=layer_scale, drop_rate=drop_rate, name=name + "attn_")

    """ MLP proj 1 """
    nn = batchnorm_with_activation(attn_out, activation=None, name=name + "mlp_")
    nn = conv2d_no_bias(nn, input_channel * mlp_ratio, use_bias=True, name=name + "mlp_1_")
    nn = depthwise_conv2d_no_bias(nn, kernel_size=3, padding="SAME", use_bias=True, name=name + "mlp_2_")
    nn = activation_by_name(nn, activation=activation, name=name + "mlp_")
    # drop

    """ MLP proj 2 """
    nn = feature_decompose(nn, activation=activation, name=name + "mlp_decompose_")
    nn = conv2d_no_bias(nn, input_channel, use_bias=True, name=name + "mlp_3_")
    nn = add_with_layer_scale_and_drop_block(attn_out, nn, layer_scale=layer_scale, drop_rate=drop_rate, name=name + "mlp_")
    return nn


def MogaNet(
    num_blocks=[3, 3, 10, 2],
    out_channels=[32, 64, 96, 192],
    mlp_ratios=[8, 8, 4, 4],
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="gelu",
    attn_activation="swish",
    drop_connect_rate=0,
    dropout=0,
    layer_scale=1e-5,
    classifier_activation="softmax",
    pretrained=None,
    model_name="moganet",
    kwargs=None,
):
    # Regard input_shape as force using original shape if len(input_shape) == 4,
    # else assume channel dimention is the one with min value in input_shape, and put it first or last regarding image_data_format
    input_shape = backend.align_input_shape_by_image_data_format(input_shape)
    inputs = layers.Input(input_shape)

    """ Stem """
    nn = conv2d_no_bias(inputs, out_channels[0] // 2, kernel_size=3, strides=2, padding="SAME", use_bias=True, name="stem_1_")
    nn = batchnorm_with_activation(nn, activation=activation, name="stem_1_")
    nn = conv2d_no_bias(nn, out_channels[0], kernel_size=3, strides=2, padding="SAME", use_bias=True, name="stem_2_")
    nn = batchnorm_with_activation(nn, activation=None, name="stem_2_")

    """ stacks """
    total_blocks = sum(num_blocks)
    global_block_id = 0
    for stack_id, (num_block, out_channel) in enumerate(zip(num_blocks, out_channels)):
        stack_name = "stack{}_".format(stack_id + 1)
        if stack_id > 0:
            nn = conv2d_no_bias(nn, out_channel, 3, strides=2, padding="same", use_bias=True, name=stack_name + "downsample_")
            nn = batchnorm_with_activation(nn, activation=None, name=stack_name + "downsample_")

        mlp_ratio = mlp_ratios[stack_id] if isinstance(mlp_ratios, (list, tuple)) else mlp_ratios
        for block_id in range(num_block):
            name = stack_name + "block{}_".format(block_id + 1)
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            nn = moga_block(nn, mlp_ratio, layer_scale, block_drop_rate, attn_activation=attn_activation, activation=activation, name=name)
            global_block_id += 1
        nn = batchnorm_with_activation(nn, activation=None, name=stack_name + "output_")

    nn = output_block(nn, num_classes=num_classes, drop_rate=dropout, classifier_activation=classifier_activation)
    model = models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    reload_model_weights(model, PRETRAINED_DICT, "moganet", pretrained)
    return model


def MogaNetXtiny(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return MogaNet(**locals(), model_name="moganet_xtiny", **kwargs)


def MogaNetTiny(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 3, 12, 2]
    out_channels = [32, 64, 128, 256]
    return MogaNet(**locals(), model_name="moganet_tiny", **kwargs)


def MogaNetSmall(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 3, 12, 2]
    out_channels = [64, 128, 320, 512]
    return MogaNet(**locals(), model_name="moganet_small", **kwargs)


def MogaNetBase(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [4, 6, 22, 3]
    out_channels = [64, 160, 320, 512]
    return MogaNet(**locals(), model_name="moganet_base", **kwargs)


def MogaNetLarge(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [4, 6, 44, 4]
    out_channels = [64, 160, 320, 640]
    return MogaNet(**locals(), model_name="moganet_large", **kwargs)
