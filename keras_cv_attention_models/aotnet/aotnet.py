import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import os

from keras_cv_attention_models import attention_layers

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


def se_module(inputs, se_ratio=0.25, activation="relu", use_bias=True, name=""):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    h_axis, w_axis = [2, 3] if K.image_data_format() == "channels_first" else [1, 2]

    filters = inputs.shape[channel_axis]
    # reduction = _make_divisible(filters // se_ratio, 8)
    reduction = int(filters * se_ratio)
    se = tf.reduce_mean(inputs, [h_axis, w_axis], keepdims=True)
    se = keras.layers.Conv2D(reduction, kernel_size=1, use_bias=use_bias, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + "1_conv")(se)
    se = keras.layers.Activation(activation)(se)
    se = keras.layers.Conv2D(filters, kernel_size=1, use_bias=use_bias, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + "2_conv")(se)
    se = keras.layers.Activation("sigmoid")(se)
    return keras.layers.Multiply()([inputs, se])


def anti_alias_downsample(inputs, kernel_size=3, strides=2, padding="SAME", trainable=False, name=None):
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


def attn_block(inputs, filters, strides=1, attn_type=None, se_ratio=0, halo_block_size=4, use_bn=True, activation="relu", name=""):
    nn = inputs
    if attn_type == "mhsa":  # MHSA block
        num_heads = 4
        key_dim = filters // num_heads
        nn = attention_layers.MHSAWithPositionEmbedding(num_heads=num_heads, key_dim=key_dim, relative=True, out_bias=True, name=name + "mhsa")(nn)
    elif attn_type == "halo":  # HaloAttention
        nn = attention_layers.HaloAttention(num_heads=8, key_dim=16, block_size=halo_block_size, halo_size=1, out_bias=True, name=name + "halo")(nn)
    elif attn_type == "sa":  # split_attention_conv2d
        nn = attention_layers.split_attention_conv2d(nn, filters=filters, kernel_size=3, strides=strides, groups=2, activation=activation, name=name + "sa_")
    elif attn_type == "cot":  # cot_attention
        nn = attention_layers.cot_attention(nn, 3, activation=activation, name=name + "cot_")
    elif attn_type == "outlook":  # outlook_attention
        nn = attention_layers.outlook_attention(nn, filters, num_head=6, kernel_size=3, name=name + "outlook_")
    elif attn_type == "gd":  # resnext groups_depthwise
        nn = attention_layers.groups_depthwise(nn, groups=32, kernel_size=3, strides=strides, name=name + "GD_")
    else:  # ResNet block
        nn = conv2d_no_bias(nn, filters, 3, strides=strides, padding="SAME", name=name + "conv_")

    if attn_type not in [None, "sa", "gd"] and strides != 1:  # Downsample
        nn = keras.layers.ZeroPadding2D(padding=1, name=name + "pad")(nn)
        nn = keras.layers.AveragePooling2D(pool_size=3, strides=strides, name=name + "pool")(nn)

    if use_bn:
        nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name=name)

    if se_ratio:
        nn = se_module(nn, se_ratio=se_ratio, activation=activation, name=name + "se_")
    return nn


def block(inputs, filters, preact=False, strides=1, conv_shortcut=False, expansion=4, attn_type=None, se_ratio=0, activation="relu", name=""):
    expanded_filter = filters * expansion
    halo_block_size = 4
    if attn_type == "halo" and inputs.shape[1] % halo_block_size != 0:  # HaloAttention
        gap = halo_block_size - inputs.shape[1] % halo_block_size
        pad_head, pad_tail = gap // 2, gap - gap // 2
        inputs = keras.layers.ZeroPadding2D(padding=((pad_head, pad_tail), (pad_head, pad_tail)), name=name + "gap_pad")(inputs)

    shortcut = keras.layers.MaxPooling2D(strides, strides=strides, padding="SAME")(inputs) if strides > 1 else inputs

    if preact:  # ResNetV2
        inputs = batchnorm_with_activation(inputs, activation=activation, zero_gamma=False, name=name + "preact_")

    if conv_shortcut:  # Set a new shortcut using conv
        shortcut = keras.layers.AvgPool2D(strides, strides=strides, padding="SAME", name=name + "shorcut_pool")(inputs) if strides > 1 else inputs
        # shortcut = anti_alias_downsample(inputs, kernel_size=3, strides=2, name=name + "shorcut_") if strides > 1 else inputs
        shortcut = conv2d_no_bias(shortcut, expanded_filter, 1, strides=1, name=name + "shorcut_")
        # shortcut = conv2d_no_bias(inputs, expanded_filter, 1, strides=strides, name=name + "shorcut_")
        if not preact:  # ResNet
            shortcut = batchnorm_with_activation(shortcut, activation=None, zero_gamma=False, name=name + "shorcut_")

    if expansion > 1:
        nn = conv2d_no_bias(inputs, filters, 1, strides=1, padding="VALID", name=name + "1_")
    else:  # ResNet-RS like
        nn = conv2d_no_bias(inputs, filters, 3, strides=1, padding="SAME", name=name + "1_")  # Using strides=1 for not changing input shape
        # nn = conv2d_no_bias(inputs, filters, 3, strides=strides, padding="SAME", name=name + "1_")
        # strides = 1
    nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name=name + "1_")
    nn = attn_block(nn, filters, strides, attn_type, se_ratio / expansion, halo_block_size, True, activation, name=name + "2_")

    if expansion > 1:  # not ResNet-RS like
        nn = conv2d_no_bias(nn, expanded_filter, 1, strides=1, padding="VALID", name=name + "3_")

    # print(">>>> shortcut:", shortcut.shape, "nn:", nn.shape)
    if preact:  # ResNetV2
        return keras.layers.Add(name=name + "add")([shortcut, nn])
    else:
        nn = batchnorm_with_activation(nn, activation=None, zero_gamma=True, name=name + "3_")
        nn = keras.layers.Add(name=name + "add")([shortcut, nn])
        return keras.layers.Activation(activation, name=name + "out")(nn)


def stack1(inputs, blocks, filters, preact=False, strides=2, expansion=4, attn_types=None, se_ratio=0, activation="relu", name=""):
    nn = inputs
    # print(">>>> attn_types:", attn_types)
    for id in range(blocks):
        conv_shortcut = True if id == 0 and (strides != 1 or inputs.shape[-1] != filters * expansion) else False
        cur_strides = strides if id == 0 else 1
        block_name = name + "block{}_".format(id + 1)
        attn_type = attn_types[id] if isinstance(attn_types, (list, tuple)) else attn_types
        cur_se_ratio = se_ratio[id] if isinstance(se_ratio, (list, tuple)) else se_ratio
        nn = block(nn, filters, preact, cur_strides, conv_shortcut, expansion, attn_type, cur_se_ratio, activation, name=block_name)
    return nn


def stack2(inputs, blocks, filters, preact=True, strides=2, expansion=4, attn_types=None, se_ratio=0, activation="relu", name=""):
    nn = inputs
    # print(">>>> attn_types:", attn_types)
    for id in range(blocks):
        conv_shortcut = True if id == 0 else False
        cur_strides = strides if id == blocks - 1 else 1
        block_name = name + "block{}_".format(id + 1)
        attn_type = attn_types[id] if isinstance(attn_types, (list, tuple)) else attn_types
        cur_se_ratio = se_ratio[id] if isinstance(se_ratio, (list, tuple)) else se_ratio
        nn = block(nn, filters, preact, cur_strides, conv_shortcut, expansion, attn_type, cur_se_ratio, activation, name=block_name)
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


def AotNet(
    num_blocks,
    preact=False,
    stack=stack1,
    strides=[1, 2, 2, 1],
    out_channels=[64, 128, 256, 512],
    stem_width=64,
    deep_stem=False,
    stem_downsample=True,
    attn_types=None,
    expansion=4,
    se_ratio=0,  # (0, 1)
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="relu",
    classifier_activation="softmax",
    model_name="aotnet",
    **kwargs
):
    inputs = keras.layers.Input(shape=input_shape)
    nn = stem(inputs, stem_width, activation=activation, deep_stem=deep_stem, name="stem_")

    if not preact:
        nn = batchnorm_with_activation(nn, activation=activation, name="stem_")
    if stem_downsample:
        nn = keras.layers.ZeroPadding2D(padding=1, name="stem_pool_pad")(nn)
        nn = keras.layers.MaxPooling2D(pool_size=3, strides=2, name="stem_pool")(nn)

    for id, (num_block, out_channel, stride) in enumerate(zip(num_blocks, out_channels, strides)):
        name = "stack{}_".format(id + 1)
        attn_type = attn_types[id] if isinstance(attn_types, (list, tuple)) else attn_types
        cur_se_ratio = se_ratio[id] if isinstance(se_ratio, (list, tuple)) else se_ratio
        nn = stack(nn, num_block, out_channel, preact, stride, expansion, attn_type, cur_se_ratio, activation, name=name)

    if preact:
        nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name="post_")

    if num_classes > 0:
        nn = keras.layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        nn = keras.layers.Dense(num_classes, activation=classifier_activation, name="predictions")(nn)

    model = keras.models.Model(inputs, nn, name=model_name)
    return model


def AotNet50(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", strides=1, **kwargs):
    num_blocks = [3, 4, 6, 3]
    strides = strides if isinstance(strides, (list, tuple)) else [1, 2, 2, strides]
    return AotNet(**locals(), model_name="aotnet50", **kwargs)


def AotNet101(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", strides=1, **kwargs):
    num_blocks = [3, 4, 23, 3]
    strides = strides if isinstance(strides, (list, tuple)) else [1, 2, 2, strides]
    return AotNet(**locals(), model_name="aotnet101", **kwargs)


def AotNet152(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", strides=1, **kwargs):
    num_blocks = [3, 8, 36, 3]
    strides = strides if isinstance(strides, (list, tuple)) else [1, 2, 2, strides]
    return AotNet(**locals(), model_name="aotnet152", **kwargs)


def AotNet200(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", strides=1, **kwargs):
    num_blocks = [3, 24, 36, 3]
    strides = strides if isinstance(strides, (list, tuple)) else [1, 2, 2, strides]
    return AotNet(**locals(), model_name="aotnet200", **kwargs)


def AotNetV2(num_blocks, preact=True, stack=stack2, strides=1, **kwargs):
    strides = strides if isinstance(strides, (list, tuple)) else [2, 2, 2, strides]
    return AotNet(num_blocks, preact=preact, stack=stack, strides=strides, **kwargs)


def AotNet50V2(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", strides=1, **kwargs):
    num_blocks = [3, 4, 6, 3]
    strides = strides if isinstance(strides, (list, tuple)) else [2, 2, 2, strides]
    return AotNetV2(**locals(), model_name="aotnet50v2", **kwargs)


def AotNet101V2(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", strides=1, **kwargs):
    num_blocks = [3, 4, 23, 3]
    strides = strides if isinstance(strides, (list, tuple)) else [2, 2, 2, strides]
    return AotNetV2(**locals(), model_name="aotnet101v2", **kwargs)


def AotNet152V2(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", strides=1, **kwargs):
    num_blocks = [3, 8, 36, 3]
    strides = strides if isinstance(strides, (list, tuple)) else [2, 2, 2, strides]
    return AotNetV2(**locals(), model_name="aotnet152v2", **kwargs)


def AotNet200V2(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", strides=1, **kwargs):
    num_blocks = [3, 24, 36, 3]
    strides = strides if isinstance(strides, (list, tuple)) else [2, 2, 2, strides]
    return AotNetV2(**locals(), model_name="aotnet200v2", **kwargs)
