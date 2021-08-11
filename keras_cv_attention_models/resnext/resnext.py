import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras import backend as K
import os

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5
CONV_KERNEL_INITIALIZER = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")


def batchnorm_with_activation(inputs, activation="relu", zero_gamma=False, name=None):
    """Performs a batch normalization followed by an activation. """
    bn_axis = 3 if K.image_data_format() == "channels_last" else 1
    gamma_initializer = tf.zeros_initializer() if zero_gamma else tf.ones_initializer()
    nn = layers.BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        gamma_initializer=gamma_initializer,
        name=name and name + "bn",
    )(inputs)
    if activation:
        nn = layers.Activation(activation=activation, name=name and name + activation)(nn)
    return nn


def conv2d_no_bias(inputs, filters, kernel_size, strides=1, padding="VALID", use_bias=False, name=None, **kwargs):
    pad = max(kernel_size) // 2 if isinstance(kernel_size, (list, tuple)) else kernel_size // 2
    if padding.upper() == "SAME" and pad != 0:
        inputs = layers.ZeroPadding2D(padding=pad, name=name and name + "pad")(inputs)
    return layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="VALID",
        use_bias=use_bias,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name and name + "conv",
        **kwargs,
    )(inputs)


def groups_depthwise(inputs, groups=32, kernel_size=3, strides=1, padding="SAME", name=None):
    input_filter = inputs.shape[-1]
    cc = input_filter // groups
    nn = inputs
    if padding.upper() == "SAME":
        nn = layers.ZeroPadding2D(padding=kernel_size // 2, name=name and name + 'pad')(nn)
    nn = layers.DepthwiseConv2D(kernel_size, strides=strides, depth_multiplier=cc, use_bias=False, name=name and name + 'DC')(nn)
    nn = layers.Reshape((*nn.shape[1:-1], groups, cc, cc))(nn)
    nn = tf.reduce_sum(nn, axis=-2)
    nn = layers.Reshape((*nn.shape[1:-2], input_filter))(nn)
    return nn


def block(inputs, filters, strides=1, conv_shortcut=False, cardinality=2, activation="relu", name=""):
    expanded_filter = filters * cardinality

    if conv_shortcut:   # Set a new shortcut using conv
        shortcut = conv2d_no_bias(inputs, expanded_filter, 1, strides=strides, name=name + "shorcut_")
        shortcut = batchnorm_with_activation(shortcut, activation=None, zero_gamma=False, name=name + "shorcut_")
    else:
        shortcut = layers.MaxPooling2D(strides, strides=strides, padding="SAME")(inputs) if strides > 1 else inputs

    nn = conv2d_no_bias(inputs, filters, 1, strides=1, padding="VALID", name=name + "1_")
    nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name=name + "1_")
    nn = groups_depthwise(nn, groups=64 // cardinality, kernel_size=3, strides=strides, name=name + "GD_")
    nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name=name)

    nn = conv2d_no_bias(nn, expanded_filter, 1, strides=1, padding="VALID", name=name + "3_")

    # print(">>>> shortcut:", shortcut.shape, "nn:", nn.shape)
    nn = batchnorm_with_activation(nn, activation=None, zero_gamma=True, name=name + "3_")
    nn = layers.Add(name=name + "add")([shortcut, nn])
    return layers.Activation(activation, name=name + "out")(nn)


def stack(inputs, blocks, filters, strides=2, cardinality=2, activation="relu", name=""):
    nn = inputs
    for id in range(blocks):
        conv_shortcut = True if id == 0 and (strides != 1 or inputs.shape[-1] != filters * cardinality) else False
        cur_strides = strides if id == 0 else 1
        block_name = name + "block{}_".format(id + 1)
        nn = block(nn, filters, cur_strides, conv_shortcut, cardinality, activation, name=block_name)
    return nn


def stem(inputs, stem_width, activation="relu", deep_stem=False, name=""):
    if deep_stem:
        nn = conv2d_no_bias(inputs, stem_width, 3, strides=2, padding="same", name=name + "1_")
        nn = batchnorm_with_activation(nn, activation=activation, name=name + "1_")
        nn = conv2d_no_bias(nn, stem_width, 3, strides=1, padding="same", name=name + "2_")
        nn = batchnorm_with_activation(nn, activation=activation, name=name + "2_")
        nn = conv2d_no_bias(nn, stem_width * 2, 3, strides=1, padding="same", name=name + "3_")
    else:
        nn = conv2d_no_bias(inputs, stem_width, 7, strides=2, padding="same", name=name)
    return nn


def ResNeXt(
    num_blocks,
    strides=[1, 2, 2, 2],
    out_channels=[128, 256, 512, 1024],
    stem_width=64,
    deep_stem=False,
    stem_downsample=True,
    cardinality=2,
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="relu",
    classifier_activation="softmax",
    pretrained="imagenet",
    model_name="resnext",
    **kwargs
):
    inputs = layers.Input(shape=input_shape)
    nn = stem(inputs, stem_width, activation=activation, deep_stem=deep_stem, name="stem_")
    nn = batchnorm_with_activation(nn, activation=activation, name="stem_")
    if stem_downsample:
        nn = layers.ZeroPadding2D(padding=1, name="stem_pool_pad")(nn)
        nn = layers.MaxPooling2D(pool_size=3, strides=2, name="stem_pool")(nn)

    strides = strides if isinstance(strides, (list, tuple)) else [1, 2, 2, strides]
    for id, (num_block, out_channel, stride) in enumerate(zip(num_blocks, out_channels, strides)):
        name = "stack{}_".format(id + 1)
        nn = stack(nn, num_block, out_channel, stride, cardinality, activation, name=name)

    if num_classes > 0:
        nn = layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        nn = layers.Dense(num_classes, activation=classifier_activation, name="predictions")(nn)

    model = keras.models.Model(inputs, nn, name=model_name)
    reload_model_weights(model, input_shape, pretrained)
    return model


def reload_model_weights(model, input_shape=(224, 224, 3), pretrained="imagenet"):
    if not pretrained in ["imagenet"] or not model.name in ["resnext50", "resnext101"]:
        print(">>>> No pretraind available, model will be random initialized")
        return

    pre_url = "https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnext/{}.h5"
    url = pre_url.format(model.name)
    file_name = os.path.basename(url)
    try:
        pretrained_model = keras.utils.get_file(file_name, url, cache_subdir="models")
    except:
        print("[Error] will not load weights, url not found or download failed:", url)
        return
    else:
        print(">>>> Load pretraind from:", pretrained_model)
        model.load_weights(pretrained_model, by_name=True, skip_mismatch=True)


def ResNeXt50(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 6, 3]
    return ResNeXt(**locals(), model_name="resnext50", **kwargs)


def ResNeXt101(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 23, 3]
    return ResNeXt(**locals(), model_name="resnext101", **kwargs)
