from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models, functional, image_data_format
from keras_cv_attention_models.models import register_model
from keras_cv_attention_models.attention_layers import (
    activation_by_name,
    batchnorm_with_activation,
    conv2d_no_bias,
    depthwise_conv2d_no_bias,
    output_block,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

PRETRAINED_DICT = {
    "cspnext_large": {"imagenet": "e5972549741e0154356c44e6b7a5fa36"},
    "cspnext_medium": {"imagenet": "b9f2a20571fadc26eac2b725402f4496"},
    "cspnext_small": {"imagenet": "ac6865ca29c87bafede707fae324263a"},
    "cspnext_tiny": {"imagenet": "edef1d5d1f42f3ee2757dcb2173fef74"},
    "cspnext_xlarge": {"imagenet": "55f4fac58a219c40efe9bb229f94d8c2"},
}


BATCH_NORM_EPSILON = 1e-3
BATCH_NORM_MOMENTUM = 0.97


def channel_attention(inputs, activation="hard_sigmoid_torch", name=None):
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    h_axis, w_axis = [1, 2] if image_data_format() == "channels_last" else [2, 3]

    filters = inputs.shape[channel_axis]
    nn = functional.reduce_mean(inputs, [h_axis, w_axis], keepdims=True)
    nn = layers.Conv2D(filters, kernel_size=1, strides=1, padding="valid", use_bias=True, name=name and name + "conv")(nn)
    nn = activation_by_name(nn, activation=activation, name=name)
    return layers.Multiply(name=name and name + "out")([inputs, nn])


def conv_dw_pw_block(inputs, filters, kernel_size=1, strides=1, use_depthwise_conv=False, activation="swish", name=""):
    nn = inputs
    if use_depthwise_conv:
        nn = depthwise_conv2d_no_bias(nn, kernel_size, strides, padding="same", name=name)
        nn = batchnorm_with_activation(nn, activation=activation, epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM, name=name + "dw_")
        kernel_size, strides = 1, 1
    nn = conv2d_no_bias(nn, filters, kernel_size, strides, padding="same", name=name)
    nn = batchnorm_with_activation(nn, activation=activation, epsilon=BATCH_NORM_EPSILON, momentum=BATCH_NORM_MOMENTUM, name=name)
    return nn


def csp_block(inputs, expansion=0.5, use_shortcut=True, use_depthwise_conv=True, activation="swish", name=""):
    input_channels = inputs.shape[-1 if image_data_format() == "channels_last" else 1]
    nn = conv_dw_pw_block(inputs, int(input_channels * expansion), kernel_size=3, activation=activation, name=name + "1_")
    nn = conv_dw_pw_block(nn, input_channels, kernel_size=5, strides=1, use_depthwise_conv=use_depthwise_conv, activation=activation, name=name + "2_")
    if use_shortcut:
        nn = layers.Add()([inputs, nn])
    return nn


def csp_stack(inputs, depth, out_channels=-1, expansion=0.5, use_shortcut=True, use_depthwise_conv=True, activation="swish", name=""):
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    out_channels = inputs.shape[channel_axis] if out_channels == -1 else out_channels
    hidden_channels = int(out_channels * expansion)
    short = conv_dw_pw_block(inputs, hidden_channels, kernel_size=1, activation=activation, name=name + "short_")

    deep = conv_dw_pw_block(inputs, hidden_channels, kernel_size=1, activation=activation, name=name + "deep_")
    for id in range(depth):
        block_name = name + "block{}_".format(id + 1)
        deep = csp_block(deep, 1, use_shortcut=use_shortcut, use_depthwise_conv=use_depthwise_conv, activation=activation, name=block_name)

    out = functional.concat([deep, short], axis=channel_axis)
    out = channel_attention(out, name=name + "channel_attention_")
    out = conv_dw_pw_block(out, out_channels, kernel_size=1, activation=activation, name=name + "output_")
    return out


def spatial_pyramid_pooling(inputs, pool_sizes=(5, 9, 13), activation="swish", name=""):
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    input_channels = inputs.shape[channel_axis]
    nn = conv_dw_pw_block(inputs, input_channels // 2, kernel_size=1, activation=activation, name=name + "1_")
    pp = [layers.MaxPool2D(pool_size=ii, strides=1, padding="same")(nn) for ii in pool_sizes]
    nn = functional.concat([nn, *pp], axis=channel_axis)
    nn = conv_dw_pw_block(nn, input_channels, kernel_size=1, activation=activation, name=name + "2_")
    return nn


def CSPNeXt(
    num_blocks=[3, 6, 6, 3],
    out_channels=[128, 256, 512, 1024],
    stem_width=64,
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="swish",
    dropout=0,
    classifier_activation="softmax",
    pretrained=None,
    model_name="cspnext",
    kwargs=None,
):
    # Regard input_shape as force using original shape if len(input_shape) == 4,
    # else assume channel dimension is the one with min value in input_shape, and put it first or last regarding image_data_format
    input_shape = backend.align_input_shape_by_image_data_format(input_shape)
    inputs = layers.Input(input_shape)

    """ Stem """
    stem_width = stem_width if stem_width > 0 else out_channels[0]
    nn = conv_dw_pw_block(inputs, stem_width // 2, kernel_size=3, strides=2, activation=activation, name="stem_1_")
    nn = conv_dw_pw_block(nn, stem_width // 2, kernel_size=3, strides=1, activation=activation, name="stem_2_")
    nn = conv_dw_pw_block(nn, stem_width, kernel_size=3, strides=1, activation=activation, name="stem_3_")

    """ Blocks """
    use_spps = [False, False, False, True]
    use_shortcuts = [True, True, True, False]

    for stack_id, (num_block, out_channel, use_spp, use_shortcut) in enumerate(zip(num_blocks, out_channels, use_spps, use_shortcuts)):
        stack_name = "stack{}_".format(stack_id + 1)
        nn = conv_dw_pw_block(nn, out_channel, kernel_size=3, strides=2, activation=activation, name=stack_name + "downsample_")
        if use_spp:
            nn = spatial_pyramid_pooling(nn, activation=activation, name=stack_name + "spp_")
        nn = csp_stack(nn, num_block, use_shortcut=use_shortcut, activation=activation, name=stack_name)

    """ Output head """
    nn = output_block(nn, num_classes=num_classes, drop_rate=dropout, classifier_activation=classifier_activation)
    model = models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    reload_model_weights(model, pretrained_dict=PRETRAINED_DICT, sub_release="cspnext", pretrained=pretrained)
    return model


@register_model
def CSPNeXtTiny(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [1, 1, 1, 1]
    out_channels = [48, 96, 192, 384]
    stem_width = 24
    return CSPNeXt(**locals(), model_name="cspnext_tiny", **kwargs)


@register_model
def CSPNeXtSmall(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [1, 2, 2, 1]
    out_channels = [64, 128, 256, 512]
    stem_width = 32
    return CSPNeXt(**locals(), model_name="cspnext_small", **kwargs)


@register_model
def CSPNeXtMedium(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 4, 4, 2]
    out_channels = [96, 192, 384, 768]
    stem_width = 48
    return CSPNeXt(**locals(), model_name="cspnext_medium", **kwargs)


@register_model
def CSPNeXtLarge(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 6, 6, 3]
    out_channels = [128, 256, 512, 1024]
    stem_width = 64
    return CSPNeXt(**locals(), model_name="cspnext_large", **kwargs)


@register_model
def CSPNeXtXLarge(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [4, 8, 8, 4]
    out_channels = [160, 320, 640, 1280]
    stem_width = 80
    return CSPNeXt(**locals(), model_name="cspnext_xlarge", **kwargs)
