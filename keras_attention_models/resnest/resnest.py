import tensorflow as tf
from tensorflow.python.keras import layers
from tensorflow.keras import backend as K

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5
CONV_KERNEL_INITIALIZER = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")
# CONV_KERNEL_INITIALIZER = 'glorot_uniform'


def batchnorm_with_activation(inputs, activation="relu", name=""):
    """Performs a batch normalization followed by an activation. """
    bn_axis = 3 if K.image_data_format() == "channels_last" else 1
    nn = layers.BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        name=name + "bn",
    )(inputs)
    if activation:
        nn = layers.Activation(activation=activation, name=name + activation)(nn)
    return nn


def conv2d_no_bias(inputs, filters, kernel_size, strides=1, padding="VALID", name="", **kwargs):
    if padding.upper() == "SAME":
        inputs = layers.ZeroPadding2D(padding=kernel_size // 2, name=name + "pad")(inputs)
    return layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="VALID",
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name + "conv",
        **kwargs,
    )(inputs)


def rsoftmax(inputs, filters, radix, groups):
    if radix > 1:
        nn = tf.reshape(inputs, [-1, groups, radix, filters // groups])
        nn = tf.transpose(nn, [0, 2, 1, 3])
        # nn = tf.nn.softmax(nn, axis=1)
        nn = layers.Softmax(axis=1)(nn)
        nn = tf.reshape(nn, [-1, 1, 1, radix * filters])
    else:
        nn = layers.Activation("sigmoid")(inputs)
    return nn


def split_attention_conv2d(inputs, filters, kernel_size=3, groups=1, activation="relu", radix=2, name=""):
    h_axis, w_axis = [2, 3] if K.image_data_format() == "channels_first" else [1, 2]
    in_channels = inputs.shape[-1]
    # Using groups=2 is slow in `mixed_float16` policy
    # logits = conv2d_no_bias(inputs, filters * radix, kernel_size, padding="same", groups=groups * radix, name=name + "1_")
    logits = []
    splitted_inputs = tf.split(inputs, groups * radix, axis=-1)
    for ii in range(groups * radix):
        conv_name = name + "1_g{}_".format(ii + 1)
        logits.append(conv2d_no_bias(splitted_inputs[ii], filters, kernel_size, padding="same", name=conv_name))
    logits = tf.concat(logits, axis=-1)
    logits = batchnorm_with_activation(logits, activation=activation, name=name + "1_")

    if radix > 1:
        splited = tf.split(logits, radix, axis=-1)
        gap = tf.reduce_sum(splited, axis=0)
    else:
        gap = logits
    gap = tf.reduce_mean(gap, [h_axis, w_axis], keepdims=True)

    reduction_factor = 4
    inter_channels = max(in_channels * radix // reduction_factor, 32)
    atten = layers.Conv2D(inter_channels, kernel_size=1, name=name + "2_conv")(gap)
    atten = batchnorm_with_activation(atten, activation=activation, name=name + "2_")
    atten = layers.Conv2D(filters * radix, kernel_size=1, name=name + "3_conv")(atten)
    atten = rsoftmax(atten, filters, radix, groups)
    out = layers.Multiply()([atten, logits])

    if radix > 1:
        out = tf.split(out, radix, axis=-1)
        out = tf.reduce_sum(out, axis=0)
    return out


def block(inputs, filters, strides=1, activation="relu", use_se=False, radix=2, name=""):
    if strides != 1 or inputs.shape[-1] != filters * 4:
        short_cut = layers.AveragePooling2D(pool_size=strides, strides=strides, padding="SAME", name=name + "st_pool")(inputs)
        short_cut = conv2d_no_bias(short_cut, filters * 4, kernel_size=1, name=name + "shortcut_")
        short_cut = batchnorm_with_activation(short_cut, activation=None, name=name + "shortcut_")
    else:
        short_cut = inputs

    nn = conv2d_no_bias(inputs, filters, 1, name=name + "1_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "1_")

    nn = split_attention_conv2d(nn, filters=filters, kernel_size=3, groups=1, activation=activation, radix=radix, name=name + "sa_")
    if strides > 1:
        nn = layers.ZeroPadding2D(padding=1, name=name + "pad")(nn)
        nn = layers.AveragePooling2D(pool_size=3, strides=strides, name=name + "pool")(nn)
    nn = conv2d_no_bias(nn, filters * 4, 1, name=name + "2_")
    nn = batchnorm_with_activation(nn, activation=None, name=name + "2_")

    nn = layers.Add()([short_cut, nn])
    nn = layers.Activation(activation, name=name + "out_" + activation)(nn)
    return nn


def stack(inputs, blocks, filters, strides, activation="relu", radix=2, name=""):
    nn = block(inputs, filters=filters, strides=strides, activation=activation, radix=radix, name=name + "block1_")
    for ii in range(2, blocks + 1):
        nn = block(nn, filters=filters, strides=1, activation=activation, radix=radix, name=name + "block{}_".format(ii))
    return nn


def stem(inputs, stem_width, activation="relu", deep_stem=False, name=""):
    if deep_stem:
        nn = conv2d_no_bias(inputs, stem_width, 3, strides=2, padding="same", name=name + "1_")
        nn = batchnorm_with_activation(nn, activation=activation, name=name + "1_")
        nn = conv2d_no_bias(nn, stem_width, 3, strides=1, padding="same", name=name + "2_")
        nn = batchnorm_with_activation(nn, activation=activation, name=name + "2_")
        nn = conv2d_no_bias(nn, stem_width * 2, 3, strides=1, padding="same", name=name + "3_")
    else:
        nn = conv2d_no_bias(inputs, stem_width, 7, strides=2, padding="same", name=name + "3_")

    nn = batchnorm_with_activation(nn, activation=activation, name=name + "3_")
    nn = layers.ZeroPadding2D(padding=1, name=name + "pad")(nn)
    nn = layers.MaxPool2D(pool_size=3, strides=2, name=name + "pool")(nn)
    return nn


def ResNest(
    input_shape,
    blocks_set,
    stem_width=32,
    classes=1000,
    activation="relu",
    radix=2,
    classifier_activation="softmax",
    model_name="resnest",
    **kwargs
):
    img_input = layers.Input(shape=input_shape)
    nn = stem(img_input, stem_width, activation=activation, deep_stem=True, name="stem_")

    nn = stack(nn, blocks=blocks_set[0], filters=64, strides=1, activation=activation, radix=radix, name="stack1_")
    nn = stack(nn, blocks=blocks_set[1], filters=128, strides=2, activation=activation, radix=radix, name="stack2_")
    nn = stack(nn, blocks=blocks_set[2], filters=256, strides=2, activation=activation, radix=radix, name="stack3_")
    nn = stack(nn, blocks=blocks_set[3], filters=512, strides=2, activation=activation, radix=radix, name="stack4_")

    if classes > 0:
        nn = layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        nn = layers.Dense(classes, activation=classifier_activation, name="predictions")(nn)

    model = tf.keras.models.Model(img_input, nn, name=model_name)
    return model


def ResNest50(input_shape=(224, 224, 3), stem_width=32, model_name="ResNest50", **kwargs):
    return ResNest(blocks_set=[3, 4, 6, 3], **locals(), **kwargs)


def ResNest101(input_shape=(256, 256, 3), stem_width=64, model_name="ResNest101", **kwargs):
    return ResNest(blocks_set=[3, 4, 23, 3], **locals(), **kwargs)


def ResNest200(input_shape=(320, 320, 3), stem_width=64, model_name="ResNest200", **kwargs):
    return ResNest(blocks_set=[3, 24, 36, 3], **locals(), **kwargs)


def ResNest269(input_shape=(416, 416, 3), stem_width=64, model_name="ResNest269", **kwargs):
    return ResNest(blocks_set=[3, 30, 48, 8], **locals(), **kwargs)
