import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5
LAYER_NORM_EPSILON = 1e-5
CONV_KERNEL_INITIALIZER = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")
# CONV_KERNEL_INITIALIZER = 'glorot_uniform'

@tf.keras.utils.register_keras_serializable(package="common")
def hard_swish(inputs):
    """ `out = xx * relu6(xx + 3) / 6`, arxiv: https://arxiv.org/abs/1905.02244 """
    return inputs * tf.nn.relu6(inputs + 3) / 6


def activation_by_name(inputs, activation="relu", name=None):
    """ Typical Activation layer added hard_swish and prelu. """
    layer_name = name and activation and name + activation
    if activation == "hard_swish":
        return keras.layers.Activation(activation=hard_swish, name=layer_name)(inputs)
    elif activation.lower() == "prelu":
        shared_axes = list(range(1, len(inputs.shape)))
        shared_axes.pop(-1 if K.image_data_format() == "channels_last" else 0)
        # print(f"{shared_axes = }")
        return keras.layers.PReLU(shared_axes=shared_axes, alpha_initializer=tf.initializers.Constant(0.25), name=layer_name)(inputs)
    elif activation:
        return keras.layers.Activation(activation=activation, name=layer_name)(inputs)
    else:
        return inputs


def batchnorm_with_activation(inputs, activation="relu", zero_gamma=False, epsilon=BATCH_NORM_EPSILON, act_first=False, name=None):
    """ Performs a batch normalization followed by an activation. """
    bn_axis = -1 if K.image_data_format() == "channels_last" else 1
    gamma_initializer = tf.zeros_initializer() if zero_gamma else tf.ones_initializer()
    if act_first and activation:
        inputs = activation_by_name(inputs, activation=activation, name=name)
    nn = keras.layers.BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        gamma_initializer=gamma_initializer,
        name=name and name + "bn",
    )(inputs)
    if not act_first and activation:
        nn = activation_by_name(nn, activation=activation, name=name)
    return nn


def layer_norm(inputs, epsilon=LAYER_NORM_EPSILON, name=None):
    """ Typical LayerNormalization with epsilon=1e-5 """
    norm_axis = -1 if K.image_data_format() == "channels_last" else 1
    return keras.layers.LayerNormalization(axis=norm_axis, epsilon=LAYER_NORM_EPSILON, name=name and name + "ln")(inputs)


def conv2d_no_bias(inputs, filters, kernel_size, strides=1, padding="VALID", use_bias=False, groups=1, name=None, **kwargs):
    """ Typical Conv2D with `use_bias` default as `False` and fixed padding """
    pad = max(kernel_size) // 2 if isinstance(kernel_size, (list, tuple)) else kernel_size // 2
    if padding.upper() == "SAME" and pad != 0:
        inputs = keras.layers.ZeroPadding2D(padding=pad, name=name and name + "pad")(inputs)

    groups = max(1, groups)
    if groups == filters:
        return keras.layers.DepthwiseConv2D(
            kernel_size, strides=strides, padding="VALID", use_bias=use_bias, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name and name + "conv", **kwargs
        )(inputs)
    else:
        return keras.layers.Conv2D(
            filters,
            kernel_size,
            strides=strides,
            padding="VALID",
            use_bias=use_bias,
            groups=groups,
            kernel_initializer=CONV_KERNEL_INITIALIZER,
            name=name and name + "conv",
            **kwargs,
        )(inputs)


def depthwise_conv2d_no_bias(inputs, kernel_size, strides=1, padding="VALID", use_bias=False, name=None, **kwargs):
    """ Typical DepthwiseConv2D with `use_bias` default as `False` and fixed padding """
    pad = max(kernel_size) // 2 if isinstance(kernel_size, (list, tuple)) else kernel_size // 2
    if padding.upper() == "SAME" and pad != 0:
        inputs = keras.layers.ZeroPadding2D(padding=pad, name=name and name + "dw_pad")(inputs)
    return keras.layers.DepthwiseConv2D(
        kernel_size, strides=strides, padding="valid", use_bias=use_bias, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name and name + "dw_conv", **kwargs,
    )(inputs)


def se_module(inputs, se_ratio=0.25, activation="relu", use_bias=True, name=""):
    """ Squeeze-and-Excitation block, arxiv: https://arxiv.org/pdf/1709.01507.pdf """
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    h_axis, w_axis = [2, 3] if K.image_data_format() == "channels_first" else [1, 2]

    filters = inputs.shape[channel_axis]
    # reduction = _make_divisible(filters // se_ratio, 8)
    reduction = int(filters * se_ratio)
    se = tf.reduce_mean(inputs, [h_axis, w_axis], keepdims=True)
    se = keras.layers.Conv2D(reduction, kernel_size=1, use_bias=use_bias, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name and name + "1_conv")(se)
    se = activation_by_name(se, activation=activation, name=name)
    se = keras.layers.Conv2D(filters, kernel_size=1, use_bias=use_bias, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name and name + "2_conv")(se)
    se = keras.layers.Activation("sigmoid")(se)
    return keras.layers.Multiply()([inputs, se])


def drop_block(inputs, drop_rate=0, name=None):
    """ Stochastic Depth block by Dropout, arxiv: https://arxiv.org/abs/1603.09382 """
    if drop_rate > 0:
        noise_shape = [None] + [1] * (len(inputs.shape) - 1)  # [None, 1, 1, 1]
        return keras.layers.Dropout(drop_rate, noise_shape=noise_shape, name=name and name + "drop")(inputs)
    else:
        return inputs


def anti_alias_downsample(inputs, kernel_size=3, strides=2, padding="SAME", trainable=False, name=None):
    """ DepthwiseConv2D performing anti-aliasing downsample, arxiv: https://arxiv.org/pdf/1904.11486.pdf """
    def anti_alias_downsample_initializer(weight_shape, dtype="float32"):
        import numpy as np

        kernel_size, channel = weight_shape[0], weight_shape[2]
        ww = tf.cast(np.poly1d((0.5, 0.5)) ** (kernel_size - 1), dtype)
        ww = tf.expand_dims(ww, 0) * tf.expand_dims(ww, 1)
        ww = tf.repeat(ww[:, :, tf.newaxis, tf.newaxis], channel, axis=-2)
        return ww

    return keras.layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=strides,
        padding="SAME",
        use_bias=False,
        trainable=trainable,
        depthwise_initializer=anti_alias_downsample_initializer,
        name=name and name + "anti_alias_down",
    )(inputs)
