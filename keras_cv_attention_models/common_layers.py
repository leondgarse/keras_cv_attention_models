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
        kernel_size,
        strides=strides,
        padding="valid",
        use_bias=use_bias,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name and name + "dw_conv",
        **kwargs,
    )(inputs)


def deep_stem(inputs, stem_width, activation="relu", last_strides=1, name=None):
    nn = conv2d_no_bias(inputs, stem_width // 2, 3, strides=2, padding="same", name=name and name + "1_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name and name + "1_")
    nn = conv2d_no_bias(nn, stem_width // 2, 3, strides=1, padding="same", name=name and name + "2_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name and name + "2_")
    nn = conv2d_no_bias(nn, stem_width, 3, strides=last_strides, padding="same", name=name and name + "3_")
    return nn


def quad_stem(inputs, stem_width, activation="relu", stem_act=False, last_strides=2, name=None):
    nn = conv2d_no_bias(inputs, stem_width // 8, 3, strides=2, padding="same", name=name and name + "1_")
    if stem_act:
        nn = batchnorm_with_activation(nn, activation=activation, name=name and name + "1_")
    nn = conv2d_no_bias(nn, stem_width // 4, 3, strides=1, padding="same", name=name and name + "2_")
    if stem_act:
        nn = batchnorm_with_activation(nn, activation=activation, name=name and name + "2_")
    nn = conv2d_no_bias(nn, stem_width // 2, 3, strides=1, padding="same", name=name and name + "3_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name and name + "3_")
    nn = conv2d_no_bias(nn, stem_width, 3, strides=last_strides, padding="same", name=name and name + "4_")
    return nn


def tiered_stem(inputs, stem_width, activation="relu", last_strides=1, name=None):
    nn = conv2d_no_bias(inputs, 3 * stem_width // 8, 3, strides=2, padding="same", name=name and name + "1_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name and name + "1_")
    nn = conv2d_no_bias(nn, stem_width // 2, 3, strides=1, padding="same", name=name and name + "2_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name and name + "2_")
    nn = conv2d_no_bias(nn, stem_width, 3, strides=last_strides, padding="same", name=name and name + "3_")
    return nn


def output_block(inputs, num_features=0, activation="relu", num_classes=1000, drop_rate=0, classifier_activation="softmax"):
    nn = inputs
    if num_features > 0:  # efficientnet like
        nn = conv2d_no_bias(nn, num_features, 1, strides=1, name="features_")
        nn = batchnorm_with_activation(nn, activation=activation, name="features_")

    if num_classes > 0:
        nn = keras.layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        if drop_rate > 0:
            nn = keras.layers.Dropout(drop_rate, name="head_drop")(nn)
        nn = keras.layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="predictions")(nn)
    return nn


def se_module(inputs, se_ratio=0.25, divisor=8, activation="relu", use_bias=True, name=None):
    """ Squeeze-and-Excitation block, arxiv: https://arxiv.org/pdf/1709.01507.pdf """
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    h_axis, w_axis = [2, 3] if K.image_data_format() == "channels_first" else [1, 2]

    filters = inputs.shape[channel_axis]
    # reduction = int(filters * se_ratio)
    reduction = make_divisible(filters * se_ratio, divisor)
    se = tf.reduce_mean(inputs, [h_axis, w_axis], keepdims=True)
    se = keras.layers.Conv2D(reduction, kernel_size=1, use_bias=use_bias, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name and name + "1_conv")(se)
    se = activation_by_name(se, activation=activation, name=name)
    se = keras.layers.Conv2D(filters, kernel_size=1, use_bias=use_bias, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name and name + "2_conv")(se)
    se = activation_by_name(se, activation="sigmoid", name=name)
    return keras.layers.Multiply(name=name and name + "out")([inputs, se])


def eca_module(inputs, gamma=2.0, beta=1.0, name=None, **kwargs):
    """ Efficient Channel Attention block, arxiv: https://arxiv.org/pdf/1910.03151.pdf """
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    h_axis, w_axis = [2, 3] if K.image_data_format() == "channels_first" else [1, 2]

    filters = inputs.shape[channel_axis]
    beta, gamma = float(beta), float(gamma)
    tt = int((tf.math.log(float(filters)) / tf.math.log(2.0) + beta) / gamma)
    kernel_size = max(tt if tt % 2 else tt + 1, 3)
    pad = kernel_size // 2

    nn = tf.reduce_mean(inputs, [h_axis, w_axis], keepdims=False)
    nn = tf.pad(nn, [[0, 0], [pad, pad]])
    nn = tf.expand_dims(nn, channel_axis)

    nn = keras.layers.Conv1D(1, kernel_size=kernel_size, strides=1, padding="VALID", use_bias=False, name=name and name + "conv1d")(nn)
    nn = tf.squeeze(nn, axis=channel_axis)
    nn = activation_by_name(nn, activation="sigmoid", name=name)
    return keras.layers.Multiply(name=name and name + "out")([inputs, nn])


def drop_connect_rates_split(num_blocks, start=0.0, end=0.0):
    """ split drop connect rate in range `(start, end)` according to `num_blocks` """
    drop_connect_rates = tf.split(tf.linspace(start, end, sum(num_blocks)), num_blocks)
    return [ii.numpy().tolist() for ii in drop_connect_rates]


def drop_block(inputs, drop_rate=0, name=None):
    """ Stochastic Depth block by Dropout, arxiv: https://arxiv.org/abs/1603.09382 """
    if drop_rate > 0:
        noise_shape = [None] + [1] * (len(inputs.shape) - 1)  # [None, 1, 1, 1]
        return keras.layers.Dropout(drop_rate, noise_shape=noise_shape, name=name and name + "drop")(inputs)
    else:
        return inputs


@tf.keras.utils.register_keras_serializable(package="kecamCommon")
def __anti_alias_downsample_initializer__(weight_shape, dtype="float32"):
    import numpy as np

    kernel_size, channel = weight_shape[0], weight_shape[2]
    ww = tf.cast(np.poly1d((0.5, 0.5)) ** (kernel_size - 1), dtype)
    ww = tf.expand_dims(ww, 0) * tf.expand_dims(ww, 1)
    ww = tf.repeat(ww[:, :, tf.newaxis, tf.newaxis], channel, axis=-2)
    return ww


def anti_alias_downsample(inputs, kernel_size=3, strides=2, padding="SAME", trainable=False, name=None):
    """ DepthwiseConv2D performing anti-aliasing downsample, arxiv: https://arxiv.org/pdf/1904.11486.pdf """
    return keras.layers.DepthwiseConv2D(
        kernel_size=kernel_size,
        strides=strides,
        padding="SAME",
        use_bias=False,
        trainable=trainable,
        depthwise_initializer=__anti_alias_downsample_initializer__,
        name=name and name + "anti_alias_down",
    )(inputs)


def make_divisible(vv, divisor=4, min_value=None):
    """ Copied from https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(vv + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * vv:
        new_v += divisor
    return new_v


@tf.keras.utils.register_keras_serializable(package="kecamCommon")
def __unfold_filters_initializer__(weight_shape, dtype="float32"):
    kernel_size = weight_shape[0]
    kernel_out = kernel_size * kernel_size
    ww = tf.reshape(tf.eye(kernel_out), [kernel_size, kernel_size, 1, kernel_out])
    if len(weight_shape) == 5:  # Conv3D or Conv3DTranspose
        ww = tf.expand_dims(ww, 2)
    return ww


def unfold_by_conv2d(inputs, kernel_size=3, strides=2, dilation_rate=1, padding="SAME", compressed=True, name=None):
    if padding.upper() == "SAME":
        pad = kernel_size // 2
        pad_inputs = tf.pad(inputs, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
    else:
        pad_inputs = inputs

    _, hh, ww, cc = pad_inputs.shape
    merge_channel = tf.transpose(pad_inputs, [0, 3, 1, 2])
    merge_channel = tf.reshape(merge_channel, [-1, hh, ww, 1])

    conv_rr = keras.layers.Conv2D(
        filters=kernel_size * kernel_size,
        kernel_size=kernel_size,
        strides=strides,
        dilation_rate=dilation_rate,
        padding="VALID",
        use_bias=False,
        trainable=False,
        kernel_initializer=__unfold_filters_initializer__,
        name=name and name + "unfold_conv",
    )(merge_channel)

    out = tf.reshape(conv_rr, [-1, cc, conv_rr.shape[1], conv_rr.shape[2], kernel_size, kernel_size])
    out = tf.transpose(out, [0, 2, 3, 4, 5, 1])  # [batch, hh, ww, kernel, kernel, channnel]
    if compressed:
        out = tf.reshape(out, [-1, out.shape[1], out.shape[2], kernel_size * kernel_size * pad_inputs.shape[-1]])
    return out


def fold_by_conv2d_transpose(patches, output_shape=None, kernel_size=3, strides=2, dilation_rate=1, padding="SAME", compressed="auto", name=None):
    paded = kernel_size // 2 if padding else 0
    if compressed == "auto":
        compressed = True if len(patches.shape) == 4 else False

    if compressed:
        _, hh, ww, cc = patches.shape
        channel = cc // kernel_size // kernel_size
        conv_rr = tf.reshape(patches, [-1, hh, ww, kernel_size, kernel_size, channel])
    else:
        _, hh, ww, _, _, channel = patches.shape
        conv_rr = patches
    conv_rr = tf.transpose(conv_rr, [0, 5, 1, 2, 3, 4])  # [batch, channnel, hh, ww, kernel, kernel]
    conv_rr = tf.reshape(conv_rr, [-1, hh, ww, kernel_size * kernel_size])

    convtrans_rr = keras.layers.Conv2DTranspose(
        filters=1,
        kernel_size=kernel_size,
        strides=strides,
        dilation_rate=dilation_rate,
        padding="VALID",
        output_padding=paded,
        use_bias=False,
        trainable=False,
        kernel_initializer=__unfold_filters_initializer__,
        name=name and name + "fold_convtrans",
    )(conv_rr)

    out = tf.reshape(convtrans_rr[..., 0], [-1, channel, convtrans_rr.shape[1], convtrans_rr.shape[2]])
    out = tf.transpose(out, [0, 2, 3, 1])
    if output_shape is None:
        output_shape = [-paded, -paded]
    else:
        output_shape = [output_shape[0] + paded, output_shape[1] + paded]
    out = out[:, paded : output_shape[0], paded : output_shape[1]]
    return out


def tpu_compatible_extract_patches(inputs, sizes=3, strides=2, rates=1, padding="SAME", compressed=True, tpu_test=False, name=None):
    """ Use `unfold_by_conv2d` for TPU and `extract_patches` for others, also handle `padding` and `output_shape`. """
    if len(tf.config.experimental.list_logical_devices("TPU")) == 0 and not tpu_test:
        if padding.upper() == "SAME":
            kernel_size = sizes[1] if isinstance(sizes, (list, tuple)) else sizes
            pad = kernel_size // 2
            pad_inputs = tf.pad(inputs, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
            padding = "VALID"
        else:
            pad_inputs = inputs

        sizes = sizes if isinstance(sizes, (list, tuple)) else [1, sizes, sizes, 1]
        strides = strides if isinstance(strides, (list, tuple)) else [1, strides, strides, 1]
        rates = rates if isinstance(rates, (list, tuple)) else [1, rates, rates, 1]
        patches = tf.image.extract_patches(pad_inputs, sizes, strides, rates, padding, name)
        if not compressed:
            # [batch, hh, ww, kernel, kernel, channnel]
            patches = tf.reshape(patches, [-1, patches.shape[1], patches.shape[2], sizes[1], sizes[1], inputs.shape[-1]])
        return patches
    else:
        kernel_size = sizes[1] if isinstance(sizes, (list, tuple)) else sizes
        strides = strides[1] if isinstance(strides, (list, tuple)) else strides
        dilation_rate = rates[1] if isinstance(rates, (list, tuple)) else rates
        return unfold_by_conv2d(inputs, kernel_size, strides, dilation_rate, padding, compressed=compressed, name=name)
