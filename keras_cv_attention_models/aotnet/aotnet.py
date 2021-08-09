import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras import backend as K
import os

from keras_cv_attention_models import attention_layers

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5
CONV_KERNEL_INITIALIZER = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")


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


def conv2d_no_bias(inputs, filters, kernel_size, strides=1, padding="VALID", use_bias=False, name="", **kwargs):
    if padding.upper() == "SAME":
        inputs = layers.ZeroPadding2D(padding=kernel_size // 2, name=name + "pad")(inputs)
    return layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="VALID",
        use_bias=use_bias,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name + "conv",
        **kwargs,
    )(inputs)


def se_module(inputs, se_ratio=0.25, activation="relu", use_bias=True, name=""):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    h_axis, w_axis = [2, 3] if K.image_data_format() == "channels_first" else [1, 2]

    filters = inputs.shape[channel_axis]
    # reduction = _make_divisible(filters // se_ratio, 8)
    reduction = int(filters * se_ratio)
    # se = GlobalAveragePooling2D()(inputs)
    # se = Reshape((1, 1, filters))(se)
    se = tf.reduce_mean(inputs, [h_axis, w_axis], keepdims=True)
    se = keras.layers.Conv2D(reduction, kernel_size=1, use_bias=use_bias, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + "1_conv")(se)
    # se = PReLU(shared_axes=[1, 2])(se)
    se = keras.layers.Activation(activation)(se)
    se = keras.layers.Conv2D(filters, kernel_size=1, use_bias=use_bias, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + "2_conv")(se)
    se = keras.layers.Activation("sigmoid")(se)
    return keras.layers.Multiply()([inputs, se])


def attn_block(inputs, filters, strides=1, attn_type=None, se_ratio=0, halo_block_size=4, activation="relu", name=""):
    nn = inputs
    if attn_type == "mhsa":  # MHSA block
        num_heads = 4
        key_dim = filters // num_heads
        nn = attention_layers.MHSAWithPositionEmbedding(num_heads=num_heads, key_dim=key_dim, relative=True, out_bias=True, name=name + "mhsa")(nn)
    elif attn_type == "halo":  # HaloAttention
        nn = attention_layers.HaloAttention(num_heads=8, key_dim=16, block_size=halo_block_size, halo_size=1, out_bias=True, name=name + "halo")(nn)
    elif attn_type == "sa": # split_attention_conv2d
        nn = attention_layers.split_attention_conv2d(nn, filters=filters, kernel_size=3, strides=strides, groups=2, activation=activation, name=name + "sa_")
    elif attn_type == "cot": # cot_attention
        nn = attention_layers.cot_attention(nn, 3, activation=activation, name=name + "cot_")
    elif attn_type == "outlook": # outlook_attention
        nn = attention_layers.outlook_attention(nn, filters, num_head=6, kernel_size=3, name=name + "outlook_")
    else:  # ResNet block
        nn = conv2d_no_bias(nn, filters, 3, strides=strides, padding="SAME", name=name + "2_conv_")

    if attn_type not in [None, "sa"] and strides != 1:  # Downsample
        nn = layers.ZeroPadding2D(padding=1, name=name + "pad")(nn)
        nn = layers.AveragePooling2D(pool_size=3, strides=strides, name=name + "pool")(nn)
    nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name=name + "2_")

    if se_ratio:
        nn = se_module(nn, se_ratio=se_ratio, activation=activation, name=name + "se_")
    return nn

def block1(inputs, filters, strides=1, shortcut=False, expansion=4, attn_type=None, se_ratio=0, activation="relu", name=""):
    expanded_filter = filters * expansion
    halo_block_size = 4
    if attn_type == "halo" and inputs.shape[1] % halo_block_size != 0:  # HaloAttention
        gap = halo_block_size - inputs.shape[1] % halo_block_size
        pad_head, pad_tail = gap // 2, gap - gap // 2
        inputs = layers.ZeroPadding2D(padding=((pad_head, pad_tail), (pad_head, pad_tail)), name=name + "gap_pad")(inputs)

    if shortcut:
        if strides > 1:
            shortcut = layers.AveragePooling2D(pool_size=strides, strides=strides, padding="SAME", name=name + "shorcut_pool")(inputs)
        else:
            shortcut = inputs
        shortcut = conv2d_no_bias(shortcut, expanded_filter, 1, strides=1, name=name + "shorcut_")
        shortcut = batchnorm_with_activation(shortcut, activation=None, zero_gamma=False, name=name + "shorcut_")
    else:
        shortcut = inputs

    nn = conv2d_no_bias(inputs, filters, 1, strides=1, padding="VALID", name=name + "1_")
    # nn = conv2d_no_bias(inputs, filters, 3, strides=1, padding="SAME", name=name + "1_")
    nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name=name + "1_")
    nn = attn_block(nn, filters, strides, attn_type, se_ratio, halo_block_size, activation, name)

    nn = conv2d_no_bias(nn, expanded_filter, 1, strides=1, padding="VALID", name=name + "3_")
    nn = batchnorm_with_activation(nn, activation=None, zero_gamma=True, name=name + "3_")

    nn = layers.Add(name=name + "_add")([shortcut, nn])
    return layers.Activation(activation, name=name + "out")(nn)


def stack1(inputs, blocks, filters, strides=2, expansion=4, attn_types=None, se_ratio=0, activation="relu", name=""):
    nn = inputs
    # print(">>>> attn_types:", attn_types)
    for id in range(blocks):
        shortcut = True if id == 0 and (strides != 1 or inputs.shape[-1] != filters * expansion) else False
        cur_strides = strides if id == 0 else 1
        block_name = name + "block{}_".format(id + 1)
        attn_type = attn_types[id] if isinstance(attn_types, (list, tuple)) else attn_types
        cur_se_ratio = se_ratio[id] if isinstance(se_ratio, (list, tuple)) else se_ratio
        nn = block1(nn, filters, cur_strides, shortcut, expansion, attn_type, cur_se_ratio, activation, name=block_name)
    return nn


def block2(inputs, filters, strides=1, shortcut=False, expansion=4, attn_type=None, se_ratio=0, activation="relu", name=""):
    expanded_filter = filters * expansion
    halo_block_size = 4
    if attn_type == "halo" and inputs.shape[1] % halo_block_size != 0:  # HaloAttention
        gap = halo_block_size - inputs.shape[1] % halo_block_size
        pad_head, pad_tail = gap // 2, gap - gap // 2
        inputs = layers.ZeroPadding2D(padding=((pad_head, pad_tail), (pad_head, pad_tail)), name=name + "gap_pad")(inputs)

    preact = batchnorm_with_activation(inputs, activation=activation, zero_gamma=False, name=name + "preact_")
    if shortcut:
        if strides > 1:
            shortcut = layers.AveragePooling2D(pool_size=strides, strides=strides, padding="SAME", name=name + "shorcut_pool")(preact)
        else:
            shortcut = preact
        shortcut = conv2d_no_bias(shortcut, expanded_filter, 1, strides=1, name=name + "shorcut_")
    else:
        shortcut = layers.MaxPooling2D(strides, strides=strides, padding="SAME")(inputs) if strides > 1 else inputs

    nn = conv2d_no_bias(preact, filters, 1, strides=1, padding="VALID", name=name + "1_")
    # nn = conv2d_no_bias(inputs, filters, 3, strides=1, padding="SAME", name=name + "1_")
    nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name=name + "1_")
    nn = attn_block(nn, filters, strides, attn_type, se_ratio, halo_block_size, activation, name)

    nn = conv2d_no_bias(nn, expanded_filter, 1, strides=1, padding="VALID", name=name + "3_")

    nn = layers.Add(name=name + "_add")([shortcut, nn])
    return nn


def stack2(inputs, blocks, filters, strides=2, expansion=4, attn_types=None, se_ratio=0, activation="relu", name=""):
    nn = inputs
    # print(">>>> attn_types:", attn_types)
    for id in range(blocks):
        shortcut = True if id == 0 else False
        cur_strides = strides if id == blocks - 1 else 1
        block_name = name + "block{}_".format(id + 1)
        attn_type = attn_types[id] if isinstance(attn_types, (list, tuple)) else attn_types
        cur_se_ratio = se_ratio[id] if isinstance(se_ratio, (list, tuple)) else se_ratio
        nn = block2(nn, filters, cur_strides, shortcut, expansion, attn_type, cur_se_ratio, activation, name=block_name)
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
    stem_width=64,
    deep_stem=False,
    stem_down_sample=True,
    attn_types=None,
    strides=[1, 2, 2, 1],
    expansion=4,
    se_ratio=0, # (0, 1)
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="relu",
    classifier_activation="softmax",
    model_name="aotnet",
    **kwargs
):
    inputs = layers.Input(shape=input_shape)
    nn = stem(inputs, stem_width, activation=activation, deep_stem=deep_stem, name="stem_")

    if not preact:
        nn = batchnorm_with_activation(nn, activation=activation, name="stem_")
    if stem_down_sample:
        nn = layers.ZeroPadding2D(padding=1, name="stem_pool_pad")(nn)
        nn = layers.MaxPooling2D(pool_size=3, strides=2, name="stem_pool")(nn)

    out_channels = [64, 128, 256, 512]
    for id, (num_block, out_channel, stride) in enumerate(zip(num_blocks, out_channels, strides)):
        name = "stack{}_".format(id + 1)
        attn_type = attn_types[id] if isinstance(attn_types, (list, tuple)) else attn_types
        cur_se_ratio = se_ratio[id] if isinstance(se_ratio, (list, tuple)) else se_ratio
        nn = stack(nn, num_block, out_channel, stride, expansion, attn_type, cur_se_ratio, activation, name=name)

    if preact:
        nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name="post_")

    if num_classes > 0:
        nn = layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        nn = layers.Dense(num_classes, activation=classifier_activation, name="predictions")(nn)

    model = keras.models.Model(inputs, nn, name=model_name)
    return model


def AotNet50(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", strides=1, **kwargs):
    num_blocks = [3, 4, 6, 3]
    stack = stack1
    preact = False
    strides = strides if isinstance(strides, (list, tuple)) else [1, 2, 2, strides]
    return AotNet(**locals(), model_name="aotnet50", **kwargs)


def AotNet101(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", strides=1, **kwargs):
    num_blocks = [3, 4, 23, 3]
    stack = stack1
    preact = False
    strides = strides if isinstance(strides, (list, tuple)) else [1, 2, 2, strides]
    return AotNet(**locals(), model_name="aotnet101", **kwargs)


def AotNet152(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", strides=1, **kwargs):
    num_blocks = [3, 8, 36, 3]
    stack = stack1
    preact = False
    strides = strides if isinstance(strides, (list, tuple)) else [1, 2, 2, strides]
    return AotNet(**locals(), model_name="aotnet152", **kwargs)


def AotNet50V2(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", strides=1, **kwargs):
    num_blocks = [3, 4, 6, 3]
    stack = stack2
    preact = True
    strides = strides if isinstance(strides, (list, tuple)) else [2, 2, 2, strides]
    return AotNet(**locals(), model_name="aotnet50v2", **kwargs)


def AotNet101V2(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", strides=1, **kwargs):
    num_blocks = [3, 4, 23, 3]
    stack = stack2
    preact = True
    strides = strides if isinstance(strides, (list, tuple)) else [2, 2, 2, strides]
    return AotNet(**locals(), model_name="aotnet101v2", **kwargs)


def AotNet152V2(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", strides=1, **kwargs):
    num_blocks = [3, 8, 36, 3]
    stack = stack2
    preact = True
    strides = strides if isinstance(strides, (list, tuple)) else [2, 2, 2, strides]
    return AotNet(**locals(), model_name="aotnet152v2", **kwargs)
