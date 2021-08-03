import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras import backend as K
import os


BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 0.001
CONV_KERNEL_INITIALIZER = keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")


def batchnorm_with_activation(inputs, activation="relu", zero_gamma=False, name=""):
    """Performs a batch normalization followed by an activation. """
    bn_axis = 3 if K.image_data_format() == "channels_last" else 1
    gamma_initializer = tf.zeros_initializer() if zero_gamma else tf.ones_initializer()
    nn = layers.BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        gamma_initializer=gamma_initializer,
        name=name + "bn",
    )(inputs)
    if activation:
        nn = layers.Activation(activation=activation, name=name + activation)(nn)
    return nn


def conv2d_no_bias(inputs, filters, kernel_size, strides=1, padding="VALID", name=""):
    if padding == "SAME":
        inputs = layers.ZeroPadding2D(padding=kernel_size // 2, name=name + "pad")(inputs)
    return layers.Conv2D(filters, kernel_size, strides=strides, padding="VALID", use_bias=False, name=name + "conv")(inputs)


def MBConv(inputs, output_channel, stride, expand_ratio, shortcut, survival=None, use_se=0, is_fused=False, name=""):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    input_channel = inputs.shape[channel_axis]

    if is_fused and expand_ratio != 1:
        nn = conv2d_no_bias(inputs, input_channel * expand_ratio, (3, 3), strides=stride, padding="same", name=name + "sortcut_")
        nn = batchnorm_with_activation(nn, name=name + "sortcut_")
    elif expand_ratio != 1:
        nn = conv2d_no_bias(inputs, input_channel * expand_ratio, (1, 1), strides=(1, 1), padding="same", name=name + "sortcut_")
        nn = batchnorm_with_activation(nn, name=name + "sortcut_")
    else:
        nn = inputs

    if not is_fused:
        nn = keras.layers.DepthwiseConv2D(
            (3, 3), padding="same", strides=stride, use_bias=False, depthwise_initializer=CONV_KERNEL_INITIALIZER, name=name + "MB_dw_"
        )(nn)
        nn = batchnorm_with_activation(nn, name=name + "MB_dw_")

    if use_se:
        nn = se_module(nn, se_ratio=4 * expand_ratio, name=name + "se_")

    # pw-linear
    if is_fused and expand_ratio == 1:
        nn = conv2d_no_bias(nn, output_channel, (3, 3), strides=stride, padding="same", name=name + "fu_")
        nn = batchnorm_with_activation(nn, name=name + "fu_")
    else:
        nn = conv2d_no_bias(nn, output_channel, (1, 1), strides=(1, 1), padding="same", name=name + "MB_pw_")
        nn = batchnorm_with_activation(nn, activation=None, name=name + "MB_pw_")

    if shortcut:
        if survival is not None and survival < 1:
            from tensorflow_addons.layers import StochasticDepth

            return StochasticDepth(float(survival))([inputs, nn])
        else:
            return keras.layers.Add()([inputs, nn])
    else:
        return nn


def se_module(inputs, se_ratio=4, name=""):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    h_axis, w_axis = [2, 3] if K.image_data_format() == "channels_first" else [1, 2]

    filters = inputs.shape[channel_axis]
    # reduction = _make_divisible(filters // se_ratio, 8)
    reduction = filters // se_ratio
    # se = GlobalAveragePooling2D()(inputs)
    # se = Reshape((1, 1, filters))(se)
    se = tf.reduce_mean(inputs, [h_axis, w_axis], keepdims=True)
    se = keras.layers.Conv2D(reduction, kernel_size=1, use_bias=True, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + "1_conv")(se)
    # se = PReLU(shared_axes=[1, 2])(se)
    se = keras.layers.Activation("swish")(se)
    se = keras.layers.Conv2D(filters, kernel_size=1, use_bias=True, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + "2_conv")(se)
    se = keras.layers.Activation("sigmoid")(se)
    return keras.layers.Multiply()([inputs, se])


def conv_mlp(inputs, out_channel, kernel_size=1):
    if kernel_size != 1:
        inputs = keras.layers.ZeroPadding2D(1)(inputs)
    nn = keras.layers.Conv2D(out_channel, kernel_size)(inputs)
    nn = keras.layers.ReLU()(nn)
    if kernel_size != 1:
        nn = keras.layers.ZeroPadding2D(1)(nn)
    nn = keras.layers.Conv2D(out_channel, kernel_size)(nn)
    return nn


def CoAtNet(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained=None, model_name="CoAtNet"):
    out_channels = [64, 96, 192, 384, 768]
    inputs = keras.layers.Input(input_shape)
    nn = conv_mlp(inputs, 3, 3) # s0
    nn = conv_mlp(nn, out_channels[0], 1) # mlp0
    nn = keras.layers.MaxPool2D(pool_size=2, strides=2)(nn)

    nn = MBConv(nn, out_channels[0], stride=1, expand_ratio=1, shortcut=True, use_se=0.25, is_fused=False, name="s1_") # s1
    nn = conv_mlp(nn, out_channels[1], 1) # mlp1
    nn = keras.layers.MaxPool2D(pool_size=2, strides=2)(nn)

    nn = MBConv(nn, out_channels[1], stride=1, expand_ratio=1, shortcut=True, use_se=0.25, is_fused=False, name="s2_") # s2
    nn = conv_mlp(nn, out_channels[2], 1) # mlp2
    nn = keras.layers.MaxPool2D(pool_size=2, strides=2)(nn)

    nn = keras.layers.MultiHeadAttention(num_heads=8, key_dim=nn.shape[-1] // 8)(nn, nn)  # s3
    nn = conv_mlp(nn, out_channels[3], 1) # mlp3
    nn = keras.layers.MaxPool2D(pool_size=(1, 2), strides=(1, 2))(nn)  # [batch, height, width, channel]

    nn = keras.layers.MultiHeadAttention(num_heads=8, key_dim=nn.shape[-1] // 8)(nn, nn)  # s4
    nn = conv_mlp(nn, out_channels[4], 1) # mlp4
    nn = keras.layers.MaxPool2D(pool_size=(2, 1), strides=(2, 1))(nn)

    if num_classes > 0:
        nn = keras.layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        nn = keras.layers.Dense(num_classes, activation=classifier_activation, name="predictions")(nn)

    model = keras.models.Model(inputs, nn, name=model_name)
    return model
