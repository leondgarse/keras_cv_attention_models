import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.keras import backend as K
import os

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


def rsoftmax(inputs, filters, groups):
    if groups > 1:
        nn = tf.reshape(inputs, [-1, 1, groups, filters])
        # nn = tf.transpose(nn, [0, 2, 1, 3])
        nn = tf.nn.softmax(nn, axis=2)
        nn = tf.reshape(nn, [-1, 1, 1, groups * filters])
    else:
        nn = layers.Activation("sigmoid")(inputs)
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
    atten = layers.Conv2D(inter_channels, kernel_size=1, name=name + "2_conv")(gap)
    atten = batchnorm_with_activation(atten, activation=activation, name=name + "2_")
    atten = layers.Conv2D(filters * groups, kernel_size=1, name=name + "3_conv")(atten)
    atten = rsoftmax(atten, filters, groups)
    out = layers.Multiply()([atten, logits])

    if groups > 1:
        out = tf.split(out, groups, axis=-1)
        out = tf.reduce_sum(out, axis=0)
    return out


def block(inputs, filters, strides=1, activation="relu", use_se=False, groups=2, name=""):
    if strides != 1 or inputs.shape[-1] != filters * 4:
        short_cut = layers.AveragePooling2D(pool_size=strides, strides=strides, padding="SAME", name=name + "st_pool")(inputs)
        short_cut = conv2d_no_bias(short_cut, filters * 4, kernel_size=1, name=name + "shortcut_")
        short_cut = batchnorm_with_activation(short_cut, activation=None, name=name + "shortcut_")
    else:
        short_cut = inputs

    nn = conv2d_no_bias(inputs, filters, 1, name=name + "1_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "1_")

    nn = split_attention_conv2d(nn, filters=filters, kernel_size=3, groups=groups, activation=activation, name=name + "sa_")
    if strides > 1:
        nn = layers.ZeroPadding2D(padding=1, name=name + "pad")(nn)
        nn = layers.AveragePooling2D(pool_size=3, strides=strides, name=name + "pool")(nn)
    nn = conv2d_no_bias(nn, filters * 4, 1, name=name + "2_")
    nn = batchnorm_with_activation(nn, activation=None, name=name + "2_")

    nn = layers.Add()([short_cut, nn])
    nn = layers.Activation(activation, name=name + "out_" + activation)(nn)
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
    nn = layers.ZeroPadding2D(padding=1, name=name + "pad")(nn)
    nn = layers.MaxPool2D(pool_size=3, strides=2, name=name + "pool")(nn)
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
    img_input = layers.Input(shape=input_shape)
    nn = stem(img_input, stem_width, activation=activation, deep_stem=True, name="stem_")

    out_channels = [64, 128, 256, 512]
    stack_strides = [1, 2, 2, 2]
    for id, (num_block, out_channel, stride) in enumerate(zip(num_blocks, out_channels, stack_strides)):
        name = "stack{}_".format(id + 1)
        nn = stack(nn, blocks=num_block, filters=out_channel, strides=stride, activation=activation, groups=groups, name=name)

    if num_classes > 0:
        nn = layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        nn = layers.Dense(num_classes, activation=classifier_activation, name="predictions")(nn)

    model = tf.keras.models.Model(img_input, nn, name=model_name)
    reload_model_weights(model, input_shape, pretrained)
    return model


def reload_model_weights(model, input_shape=(224, 224, 3), pretrained="imagenet"):
    if not pretrained in ["imagenet"]:
        print(">>>> No pretraind available, model will be random initialized")
        return

    pre_url = "https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnest/{}.h5"
    url = pre_url.format(model.name)
    file_name = os.path.basename(url)
    try:
        # print(">>>> Load pretraind from:", file_name, url)
        pretrained_model = keras.utils.get_file(file_name, url, cache_subdir="models")
    except:
        print("[Error] will not load weights, url not found or download failed:", url)
    else:
        print(">>>> Load pretraind from:", pretrained_model)
        model.load_weights(pretrained_model, by_name=True, skip_mismatch=True)


def ResNest50(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", groups=2, **kwargs):
    return ResNest(num_blocks=[3, 4, 6, 3], stem_width=32, model_name="resnest50", **locals(), **kwargs)


def ResNest101(input_shape=(256, 256, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", groups=2, **kwargs):
    return ResNest(num_blocks=[3, 4, 23, 3], stem_width=64, model_name="resnest101", **locals(), **kwargs)


def ResNest200(input_shape=(320, 320, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", groups=2, **kwargs):
    return ResNest(num_blocks=[3, 24, 36, 3], stem_width=64, model_name="resnest200", **locals(), **kwargs)


def ResNest269(input_shape=(416, 416, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", groups=2, **kwargs):
    return ResNest(num_blocks=[3, 30, 48, 8], stem_width=64, model_name="resnest269", **locals(), **kwargs)
