import tensorflow as tf
from tensorflow import keras
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

PRETRAINED_DICT = {"ghostnetv2_1x": {"imagenet": "4f28597d5f72731ed4ef4f69ec9c1799"}}


def ghost_module(inputs, out_channel, activation="relu", name=""):
    ratio = 2
    hidden_channel = int(tf.math.ceil(float(out_channel) / ratio))
    in_channels = inputs.shape[-1]
    # print("[ghost_module] out_channel:", out_channel, "conv_out_channel:", conv_out_channel)
    primary_conv = conv2d_no_bias(inputs, hidden_channel, name=name + "prim_")
    primary_conv = batchnorm_with_activation(primary_conv, activation=activation, name=name + "prim_")

    # hidden_channel_cheap = int(out_channel - hidden_channel_prim)
    # cheap_conv = conv2d_no_bias(primary_conv, hidden_channel_cheap, kernel_size=3, padding="SAME", groups=hidden_channel_prim, name=name + "cheap_")
    cheap_conv = depthwise_conv2d_no_bias(primary_conv, kernel_size=3, padding="SAME", name=name + "cheap_")
    cheap_conv = batchnorm_with_activation(cheap_conv, activation=activation, name=name + "cheap_")
    return keras.layers.Concatenate()([primary_conv, cheap_conv])


def ghost_module_multiply(inputs, out_channel, activation="relu", name=""):
    nn = ghost_module(inputs, out_channel, activation=activation, name=name)

    shortcut = keras.layers.AvgPool2D(pool_size=2, strides=2)(inputs)
    shortcut = conv2d_no_bias(shortcut, out_channel, name=name + "short_1_")
    shortcut = batchnorm_with_activation(shortcut, activation=None, name=name + "short_1_")
    shortcut = depthwise_conv2d_no_bias(shortcut, (1, 5), padding="SAME", name=name + "short_2_")
    shortcut = batchnorm_with_activation(shortcut, activation=None, name=name + "short_2_")
    shortcut = depthwise_conv2d_no_bias(shortcut, (5, 1), padding="SAME", name=name + "short_3_")
    shortcut = batchnorm_with_activation(shortcut, activation=None, name=name + "short_3_")
    shortcut = activation_by_name(shortcut, "sigmoid", name=name + "short_")
    # print(f"{shortcut.shape = }, {nn.shape = }")
    shortcut = tf.image.resize(shortcut, tf.shape(inputs)[1:-1], antialias=False, method="bilinear")
    # shortcut = keras.layers.UpSampling2D(size=(2, 2), interpolation='bilinear')(shortcut)

    return shortcut * nn


def ghost_bottleneck(
    inputs, out_channel, first_ghost_channel, kernel_size=3, strides=1, se_ratio=0, shortcut=True, use_ghost_module_multiply=False, activation="relu", name=""
):
    if shortcut:
        shortcut = depthwise_conv2d_no_bias(inputs, kernel_size, strides, padding="same", name=name + "short_1_")
        shortcut = batchnorm_with_activation(shortcut, activation=None, name=name + "short_1_")
        shortcut = conv2d_no_bias(shortcut, out_channel, name=name + "short_2_")
        shortcut = batchnorm_with_activation(shortcut, activation=None, name=name + "short_2_")
    else:
        shortcut = inputs

    if use_ghost_module_multiply:
        nn = ghost_module_multiply(inputs, first_ghost_channel, activation=activation, name=name + "ghost_1_")
    else:
        nn = ghost_module(inputs, first_ghost_channel, activation=activation, name=name + "ghost_1_")

    if strides > 1:
        nn = depthwise_conv2d_no_bias(nn, kernel_size, strides, padding="same", name=name + "down_")
        nn = batchnorm_with_activation(nn, activation=None, name=name + "down_")

    if se_ratio > 0:
        nn = se_module(nn, se_ratio=se_ratio, divisor=4, activation=("relu", "hard_sigmoid_torch"), name=name + "se_")

    nn = ghost_module(nn, out_channel, activation=None, name=name + "ghost_2_")
    # print(f">>>> {strides = }, {inputs.shape = }, {shortcut.shape = }, {nn.shape = }")
    return keras.layers.Add(name=name + "output")([shortcut, nn])


def GhostNetV2(
    stem_width=16,
    stem_strides=2,
    width_mul=1.0,
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="relu",
    classifier_activation="softmax",
    dropout=0,
    pretrained=None,
    model_name="ghostnetv2",
    kwargs=None,
):
    inputs = keras.layers.Input(input_shape)
    stem_width = make_divisible(stem_width * width_mul, divisor=4)
    nn = conv2d_no_bias(inputs, stem_width, 3, strides=stem_strides, padding="same", name="stem_")
    nn = batchnorm_with_activation(nn, activation=activation, name="stem_")

    """ stages """
    kernel_sizes = [3, 3, 3, 5, 5, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5]
    first_ghost_channels = [16, 48, 72, 72, 120, 240, 200, 184, 184, 480, 672, 672, 960, 960, 960, 960]
    out_channels = [16, 24, 24, 40, 40, 80, 80, 80, 80, 112, 112, 160, 160, 160, 160, 160]
    se_ratios = [0, 0, 0, 0.25, 0.25, 0, 0, 0, 0, 0.25, 0.25, 0.25, 0, 0.25, 0, 0.25]
    strides = [1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1]

    for stack_id, (kernel, stride, first_ghost, out_channel, se_ratio) in enumerate(zip(kernel_sizes, strides, first_ghost_channels, out_channels, se_ratios)):
        stack_name = "stack{}_".format(stack_id + 1)
        out_channel = make_divisible(out_channel * width_mul, 4)
        first_ghost_channel = make_divisible(first_ghost * width_mul, 4)
        shortcut = False if out_channel == nn.shape[-1] and stride == 1 else True
        use_ghost_module_multiply = True if stack_id > 1 else False
        nn = ghost_bottleneck(
            nn, out_channel, first_ghost_channel, kernel, stride, se_ratio, shortcut, use_ghost_module_multiply, activation=activation, name=stack_name
        )

    nn = conv2d_no_bias(nn, make_divisible(first_ghost_channels[-1] * width_mul, 4), 1, strides=1, name="pre_")
    nn = batchnorm_with_activation(nn, activation=activation, name="pre_")

    if num_classes > 0:
        nn = keras.layers.GlobalAveragePooling2D(keepdims=True)(nn)
        nn = conv2d_no_bias(nn, 1280, 1, strides=1, use_bias=True, name="features_")
        nn = activation_by_name(nn, activation, name="features_")
        nn = keras.layers.Flatten()(nn)
        if dropout > 0 and dropout < 1:
            nn = keras.layers.Dropout(dropout)(nn)
        nn = keras.layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="head")(nn)

    model = keras.models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    reload_model_weights(model, PRETRAINED_DICT, "ghostnetv2", pretrained)
    return model


def GhostNetV2_1X(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return GhostNetV2(**locals(), model_name="ghostnetv2_1x", **kwargs)
