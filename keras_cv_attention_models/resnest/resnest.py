import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from keras_cv_attention_models.download_and_load import reload_model_weights

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5
CONV_KERNEL_INITIALIZER = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")
# CONV_KERNEL_INITIALIZER = 'glorot_uniform'

PRETRAINED_DICT = {
    "resnest101": {"imagenet": "0e6c69ddc5aa792df75621de750f3798"},
    "resnest200": {"imagenet": "fec89e331f745d2727e17fb2e4eb0a14"},
    "resnest269": {"imagenet": "f855648d7bba0171df92e3a6bb0faec8"},
    "resnest50": {"imagenet": "04cbe66345b2b37f0c9c4c78a3b07b26"},
}

def batchnorm_with_activation(inputs, activation="relu", name=""):
    """Performs a batch normalization followed by an activation. """
    bn_axis = 3 if K.image_data_format() == "channels_last" else 1
    nn = keras.layers.BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        name=name + "bn",
    )(inputs)
    if activation:
        nn = keras.layers.Activation(activation=activation, name=name + activation)(nn)
    return nn


def conv2d_no_bias(inputs, filters, kernel_size, strides=1, padding="VALID", name="", **kwargs):
    if padding.upper() == "SAME":
        inputs = keras.layers.ZeroPadding2D(padding=kernel_size // 2, name=name + "pad")(inputs)
    return keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="VALID",
        use_bias=False,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name + "conv",
        **kwargs,
    )(inputs)


def rsoftmax(inputs, groups):
    if groups > 1:
        nn = tf.reshape(inputs, [-1, 1, groups, inputs.shape[-1] // groups])
        # nn = tf.transpose(nn, [0, 2, 1, 3])
        nn = tf.nn.softmax(nn, axis=2)
        nn = tf.reshape(nn, [-1, 1, 1, inputs.shape[-1]])
    else:
        nn = keras.layers.Activation("sigmoid")(inputs)
    return nn


def split_attention_conv2d(inputs, filters, kernel_size=3, strides=1, groups=2, activation="relu", name=""):
    h_axis, w_axis = [2, 3] if K.image_data_format() == "channels_first" else [1, 2]
    in_channels = inputs.shape[-1]
    if groups == 1:
        logits = conv2d_no_bias(inputs, filters, kernel_size, strides=strides, padding="same", name=name + "1_")
    else:
        # Using groups=2 is slow in `mixed_float16` policy
        # logits = conv2d_no_bias(inputs, filters * groups, kernel_size, padding="same", groups=groups, name=name + "1_")
        logits = []
        splitted_inputs = tf.split(inputs, groups, axis=-1)
        for ii in range(groups):
            conv_name = name + "1_g{}_".format(ii + 1)
            logits.append(conv2d_no_bias(splitted_inputs[ii], filters, kernel_size, strides=strides, padding="same", name=conv_name))
        logits = tf.concat(logits, axis=-1)
    logits = batchnorm_with_activation(logits, activation=activation, name=name + "1_")

    if groups > 1:
        splited = tf.split(logits, groups, axis=-1)
        gap = tf.reduce_sum(splited, axis=0)
    else:
        gap = logits
    gap = tf.reduce_mean(gap, [h_axis, w_axis], keepdims=True)

    reduction_factor = 4
    inter_channels = max(in_channels * groups // reduction_factor, 32)
    atten = keras.layers.Conv2D(inter_channels, kernel_size=1, name=name + "2_conv")(gap)
    atten = batchnorm_with_activation(atten, activation=activation, name=name + "2_")
    atten = keras.layers.Conv2D(filters * groups, kernel_size=1, name=name + "3_conv")(atten)
    atten = rsoftmax(atten, groups)
    out = keras.layers.Multiply()([atten, logits])

    if groups > 1:
        out = tf.split(out, groups, axis=-1)
        out = tf.reduce_sum(out, axis=0)
    return out


def block(inputs, filters, strides=1, activation="relu", use_se=False, groups=2, name=""):
    if strides != 1 or inputs.shape[-1] != filters * 4:
        short_cut = keras.layers.AveragePooling2D(pool_size=strides, strides=strides, padding="SAME", name=name + "st_pool")(inputs)
        short_cut = conv2d_no_bias(short_cut, filters * 4, kernel_size=1, name=name + "shortcut_")
        short_cut = batchnorm_with_activation(short_cut, activation=None, name=name + "shortcut_")
    else:
        short_cut = inputs

    nn = conv2d_no_bias(inputs, filters, 1, name=name + "1_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "1_")

    nn = split_attention_conv2d(nn, filters=filters, kernel_size=3, groups=groups, activation=activation, name=name + "sa_")
    if strides > 1:
        nn = keras.layers.ZeroPadding2D(padding=1, name=name + "pad")(nn)
        nn = keras.layers.AveragePooling2D(pool_size=3, strides=strides, name=name + "pool")(nn)
    nn = conv2d_no_bias(nn, filters * 4, 1, name=name + "2_")
    nn = batchnorm_with_activation(nn, activation=None, name=name + "2_")

    nn = keras.layers.Add()([short_cut, nn])
    nn = keras.layers.Activation(activation, name=name + "out_" + activation)(nn)
    return nn


def stack(inputs, blocks, filters, strides, activation="relu", groups=2, name=""):
    nn = block(inputs, filters=filters, strides=strides, activation=activation, groups=groups, name=name + "block1_")
    for ii in range(2, blocks + 1):
        nn = block(nn, filters=filters, strides=1, activation=activation, groups=groups, name=name + "block{}_".format(ii))
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
    nn = keras.layers.ZeroPadding2D(padding=1, name=name + "pad")(nn)
    nn = keras.layers.MaxPool2D(pool_size=3, strides=2, name=name + "pool")(nn)
    return nn


def ResNest(
    num_blocks,
    stem_width=32,
    groups=2,
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="relu",
    classifier_activation="softmax",
    pretrained="imagenet",
    model_name="resnest",
    **kwargs
):
    img_input = keras.layers.Input(shape=input_shape)
    nn = stem(img_input, stem_width, activation=activation, deep_stem=True, name="stem_")

    out_channels = [64, 128, 256, 512]
    stack_strides = [1, 2, 2, 2]
    for id, (num_block, out_channel, stride) in enumerate(zip(num_blocks, out_channels, stack_strides)):
        name = "stack{}_".format(id + 1)
        nn = stack(nn, blocks=num_block, filters=out_channel, strides=stride, activation=activation, groups=groups, name=name)

    if num_classes > 0:
        nn = keras.layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        nn = keras.layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="predictions")(nn)

    model = tf.keras.models.Model(img_input, nn, name=model_name)
    reload_model_weights(model, pretrained_dict=PRETRAINED_DICT, sub_release="resnest", input_shape=input_shape, pretrained=pretrained)
    return model


def ResNest50(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", groups=2, **kwargs):
    return ResNest(num_blocks=[3, 4, 6, 3], stem_width=32, model_name="resnest50", **locals(), **kwargs)


def ResNest101(input_shape=(256, 256, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", groups=2, **kwargs):
    return ResNest(num_blocks=[3, 4, 23, 3], stem_width=64, model_name="resnest101", **locals(), **kwargs)


def ResNest200(input_shape=(320, 320, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", groups=2, **kwargs):
    return ResNest(num_blocks=[3, 24, 36, 3], stem_width=64, model_name="resnest200", **locals(), **kwargs)


def ResNest269(input_shape=(416, 416, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", groups=2, **kwargs):
    return ResNest(num_blocks=[3, 30, 48, 8], stem_width=64, model_name="resnest269", **locals(), **kwargs)
