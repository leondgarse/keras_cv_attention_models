import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import os

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5
CONV_KERNEL_INITIALIZER = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")


def batchnorm_with_activation(inputs, activation="relu", zero_gamma=False, name=None):
    """Performs a batch normalization followed by an activation. """
    bn_axis = 3 if K.image_data_format() == "channels_last" else 1
    gamma_initializer = tf.zeros_initializer() if zero_gamma else tf.ones_initializer()
    nn = keras.layers.BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        gamma_initializer=gamma_initializer,
        name=name and name + "bn",
    )(inputs)
    if activation:
        nn = keras.layers.Activation(activation=activation, name=name and name + activation)(nn)
    return nn


def conv2d_no_bias(inputs, filters, kernel_size, strides=1, padding="VALID", use_bias=False, name=None, **kwargs):
    pad = max(kernel_size) // 2 if isinstance(kernel_size, (list, tuple)) else kernel_size // 2
    if padding.upper() == "SAME" and pad != 0:
        inputs = keras.layers.ZeroPadding2D(padding=pad, name=name and name + "pad")(inputs)
    return keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="VALID",
        use_bias=use_bias,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name and name + "conv",
        **kwargs,
    )(inputs)


def block(inputs, filters, expansion=2, strides=1, conv_shortcut=False, groups=32, avg_pool_down=False, activation="relu", name=""):
    expanded_filter = filters * expansion

    if conv_shortcut:  # Set a new shortcut using conv
        if strides > 1 and avg_pool_down:
            shortcut = keras.layers.AvgPool2D(strides, strides=strides, padding="SAME", name=name + "shorcut_down")(inputs)
            shortcut = conv2d_no_bias(shortcut, expanded_filter, 1, strides=1, name=name + "shortcut_")
        else:
            shortcut = conv2d_no_bias(inputs, expanded_filter, 1, strides=strides, name=name + "shortcut_")
        shortcut = batchnorm_with_activation(shortcut, activation=None, zero_gamma=False, name=name + "shortcut_")
    else:
        shortcut = keras.layers.MaxPooling2D(strides, strides=strides, padding="SAME")(inputs) if strides > 1 else inputs

    nn = conv2d_no_bias(inputs, filters, 1, strides=1, padding="VALID", name=name + "1_")
    nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name=name + "1_")
    nn = conv2d_no_bias(nn, nn.shape[-1], 3, strides=strides, groups=groups, padding="SAME", name=name + "GC_")
    nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name=name)

    nn = conv2d_no_bias(nn, expanded_filter, 1, strides=1, padding="VALID", name=name + "3_")

    # print(">>>> shortcut:", shortcut.shape, "nn:", nn.shape)
    nn = batchnorm_with_activation(nn, activation=None, zero_gamma=True, name=name + "3_")
    nn = keras.layers.Add(name=name + "add")([shortcut, nn])
    return keras.layers.Activation(activation, name=name + "out")(nn)


def stack(inputs, blocks, filters, expansion=2, strides=2, groups=32, avg_pool_down=False, activation="relu", name=""):
    nn = inputs
    for id in range(blocks):
        conv_shortcut = True if id == 0 and (strides != 1 or inputs.shape[-1] != filters * expansion) else False
        cur_strides = strides if id == 0 else 1
        block_name = name + "block{}_".format(id + 1)
        nn = block(nn, filters, expansion, cur_strides, conv_shortcut, groups, avg_pool_down, activation, name=block_name)
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
    expansion=2,
    stem_width=64,
    deep_stem=False,
    stem_downsample=True,
    groups=32,
    avg_pool_down=False,
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="relu",
    classifier_activation="softmax",
    pretrained="imagenet",
    model_name="resnext",
    **kwargs
):
    inputs = keras.layers.Input(shape=input_shape)
    nn = stem(inputs, stem_width, activation=activation, deep_stem=deep_stem, name="stem_")
    nn = batchnorm_with_activation(nn, activation=activation, name="stem_")
    if stem_downsample:
        nn = keras.layers.ZeroPadding2D(padding=1, name="stem_pool_pad")(nn)
        nn = keras.layers.MaxPooling2D(pool_size=3, strides=2, name="stem_pool")(nn)

    strides = strides if isinstance(strides, (list, tuple)) else [1, 2, 2, strides]
    for id, (num_block, out_channel, stride) in enumerate(zip(num_blocks, out_channels, strides)):
        name = "stack{}_".format(id + 1)
        nn = stack(nn, num_block, out_channel, expansion, stride, groups, avg_pool_down, activation, name=name)

    if num_classes > 0:
        nn = keras.layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        nn = keras.layers.Dense(num_classes, activation=classifier_activation, name="predictions")(nn)

    model = keras.models.Model(inputs, nn, name=model_name)
    reload_model_weights(model, input_shape, pretrained)
    return model


def reload_model_weights(model, input_shape=(224, 224, 3), pretrained="imagenet"):
    pretrained_dd = {
        "resnext50": ["imagenet"],
        "resnext101": ["imagenet"],
        "resnext50d": ["imagenet"],
        "resnext101w": ["imagenet"],
    }
    if model.name not in pretrained_dd or pretrained not in pretrained_dd[model.name]:
        print(">>>> No pretraind available, model will be randomly initialized")
        return

    pre_url = "https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/{}_{}.h5"
    url = pre_url.format(model.name, pretrained)
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


def ResNeXt50D(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 6, 3]
    deep_stem = True
    stem_width = 32
    avg_pool_down = True
    return ResNeXt(**locals(), model_name="resnext50d", **kwargs)


def ResNeXt101W(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    out_channels = [256, 512, 1024, 2048]
    expansion = 1
    num_blocks = [3, 4, 23, 3]
    return ResNeXt(**locals(), model_name="resnext101w", **kwargs)
