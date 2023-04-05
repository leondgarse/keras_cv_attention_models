import math
from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models, functional, image_data_format
from keras_cv_attention_models.attention_layers import (
    activation_by_name,
    batchnorm_with_activation,
    conv2d_no_bias,
    depthwise_conv2d_no_bias,
    make_divisible,
    se_module,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

PRETRAINED_DICT = {
    "ghostnetv2_100": {"imagenet": "4f28597d5f72731ed4ef4f69ec9c1799"},
    "ghostnet_050": {"imagenet": "dbb5a89e19fa78f2f35f38b4c1ae4351"},
    "ghostnet_100": {"imagenet": "19a0f0f03f20e4bd6c1736102b4d979d"},
    "ghostnet_130": {"imagenet": "3a73bc721765c516a894b567674fc60b", "ssld": "62571bb90d71a7487679ae97642d13fb"},
}


def decoupled_fully_connected_attention_block(inputs, out_channel, name=""):
    # nn = layers.AvgPool2D(pool_size=2, strides=2, padding="SAME")(inputs)
    nn = layers.AvgPool2D(pool_size=2, strides=2)(inputs)
    nn = conv2d_no_bias(nn, out_channel, name=name + "1_")
    nn = batchnorm_with_activation(nn, activation=None, name=name + "1_")
    nn = depthwise_conv2d_no_bias(nn, (1, 5), padding="SAME", name=name + "2_")
    nn = batchnorm_with_activation(nn, activation=None, name=name + "2_")
    nn = depthwise_conv2d_no_bias(nn, (5, 1), padding="SAME", name=name + "3_")
    nn = batchnorm_with_activation(nn, activation=None, name=name + "3_")
    nn = activation_by_name(nn, "sigmoid", name=name)
    # print(f"{nn.shape = }, {nn.shape = }")
    size = functional.shape(inputs)[1:-1] if image_data_format() == "channels_last" else functional.shape(inputs)[2:]  # For dynamic shape
    nn = functional.resize(nn, size, antialias=False, method="bilinear")
    # nn = layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(nn)
    # if int(nn.shape[1] - nn.shape[1]) > 0 or int(nn.shape[2] - nn.shape[2]) > 0:
    #     nn = nn[:, :nn.shape[1], :nn.shape[2]]
    # nn = layers.Resizing(nn.shape[1], nn.shape[2])(nn)
    return nn


def ghost_module(inputs, out_channel, use_dfc_block=False, activation="relu", name=""):
    ratio = 2
    hidden_channel = int(math.ceil(float(out_channel) / ratio))
    # print("[ghost_module] out_channel:", out_channel, "conv_out_channel:", conv_out_channel)
    primary_conv = conv2d_no_bias(inputs, hidden_channel, name=name + "prim_")
    primary_conv = batchnorm_with_activation(primary_conv, activation=activation, name=name + "prim_")

    # hidden_channel_cheap = int(out_channel - hidden_channel_prim)
    # cheap_conv = conv2d_no_bias(primary_conv, hidden_channel_cheap, kernel_size=3, padding="SAME", groups=hidden_channel_prim, name=name + "cheap_")
    cheap_conv = depthwise_conv2d_no_bias(primary_conv, kernel_size=3, padding="SAME", name=name + "cheap_")
    cheap_conv = batchnorm_with_activation(cheap_conv, activation=activation, name=name + "cheap_")
    ghost_out = layers.Concatenate()([primary_conv, cheap_conv])

    if use_dfc_block:
        shortcut = decoupled_fully_connected_attention_block(inputs, out_channel, name=name + "short_")
        ghost_out = layers.Multiply()([shortcut, ghost_out])
    return ghost_out


def ghost_bottleneck(
    inputs, out_channel, first_ghost_channel, kernel_size=3, strides=1, se_ratio=0, shortcut=True, use_dfc_block=False, activation="relu", name=""
):
    if shortcut:
        shortcut = depthwise_conv2d_no_bias(inputs, kernel_size, strides, padding="same", name=name + "short_1_")
        shortcut = batchnorm_with_activation(shortcut, activation=None, name=name + "short_1_")
        shortcut = conv2d_no_bias(shortcut, out_channel, name=name + "short_2_")
        shortcut = batchnorm_with_activation(shortcut, activation=None, name=name + "short_2_")
    else:
        shortcut = inputs

    nn = ghost_module(inputs, first_ghost_channel, use_dfc_block=use_dfc_block, activation=activation, name=name + "ghost_1_")

    if strides > 1:
        nn = depthwise_conv2d_no_bias(nn, kernel_size, strides, padding="same", name=name + "down_")
        nn = batchnorm_with_activation(nn, activation=None, name=name + "down_")

    if se_ratio > 0:
        nn = se_module(nn, se_ratio=se_ratio, divisor=4, activation=(activation, "hard_sigmoid_torch"), name=name + "se_")

    nn = ghost_module(nn, out_channel, activation=None, name=name + "ghost_2_")
    # print(f">>>> {strides = }, {inputs.shape = }, {shortcut.shape = }, {nn.shape = }")
    return layers.Add(name=name + "output")([shortcut, nn])


def GhostNetV2(
    kernel_sizes=[3, 3, 3, 5, 5, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5],
    first_ghost_channels=[16, 48, 72, 72, 120, 240, 200, 184, 184, 480, 672, 672, 960, 960, 960, 960],
    out_channels=[16, 24, 24, 40, 40, 80, 80, 80, 80, 112, 112, 160, 160, 160, 160, 160],
    se_ratios=[0, 0, 0, 0.25, 0.25, 0, 0, 0, 0, 0.25, 0.25, 0.25, 0, 0.25, 0, 0.25],
    strides=[1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1],
    stem_width=16,
    stem_strides=2,
    width_mul=1.0,
    num_ghost_module_v1_stacks=2,  # num of `ghost_module` stcks on the head, others are `ghost_module` with `dfc_block`, set `-1` for all using `ghost_module`
    output_conv_filter=-1,  # -1 for first_ghost_channels[-1] * width_mul
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="relu",
    classifier_activation="softmax",
    dropout=0,
    pretrained=None,
    model_name="ghostnetv2",
    kwargs=None,
):
    # Regard input_shape as force using original shape if len(input_shape) == 4,
    # else assume channel dimention is the one with min value in input_shape, and put it first or last regarding image_data_format
    input_shape = backend.align_input_shape_by_image_data_format(input_shape)
    inputs = layers.Input(input_shape)
    channel_axis = -1 if image_data_format() == "channels_last" else 1

    stem_width = make_divisible(stem_width * width_mul, divisor=4)
    nn = conv2d_no_bias(inputs, stem_width, 3, strides=stem_strides, padding="same", name="stem_")
    nn = batchnorm_with_activation(nn, activation=activation, name="stem_")

    """ stages """
    for stack_id, (kernel, stride, first_ghost, out_channel, se_ratio) in enumerate(zip(kernel_sizes, strides, first_ghost_channels, out_channels, se_ratios)):
        stack_name = "stack{}_".format(stack_id + 1)
        out_channel = make_divisible(out_channel * width_mul, 4)
        first_ghost_channel = make_divisible(first_ghost * width_mul, 4)
        shortcut = False if out_channel == nn.shape[channel_axis] and stride == 1 else True
        use_dfc_block = True if num_ghost_module_v1_stacks >= 0 and stack_id >= num_ghost_module_v1_stacks else False
        nn = ghost_bottleneck(nn, out_channel, first_ghost_channel, kernel, stride, se_ratio, shortcut, use_dfc_block, activation=activation, name=stack_name)

    output_conv_filter = output_conv_filter if output_conv_filter > 0 else make_divisible(first_ghost_channels[-1] * width_mul, 4)
    nn = conv2d_no_bias(nn, output_conv_filter, 1, strides=1, name="pre_")
    nn = batchnorm_with_activation(nn, activation=activation, name="pre_")

    if num_classes > 0:
        nn = layers.GlobalAveragePooling2D(keepdims=True)(nn)
        nn = conv2d_no_bias(nn, 1280, 1, strides=1, use_bias=True, name="features_")
        nn = activation_by_name(nn, activation, name="features_")
        nn = layers.Flatten()(nn)
        if dropout > 0 and dropout < 1:
            nn = layers.Dropout(dropout)(nn)
        nn = layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="head")(nn)

    model = models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    reload_model_weights(model, PRETRAINED_DICT, "ghostnetv2", pretrained)
    return model


def GhostNetV2_100(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return GhostNetV2(**locals(), model_name="ghostnetv2_100", **kwargs)
