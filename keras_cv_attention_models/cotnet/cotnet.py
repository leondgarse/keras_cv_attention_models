import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.keras import backend as K
from tensorflow_addons.layers import StochasticDepth, GroupNormalization
import os

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5
CONV_KERNEL_INITIALIZER = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")
# CONV_KERNEL_INITIALIZER = 'glorot_uniform'


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


def group_conv(inputs, filters, kernel_size, groups=4, name=""):
    # Using groups=num in `Conv2D` is slow with `mixed_float16` policy
    return conv2d_no_bias(inputs, filters, kernel_size, groups=groups, name=name)
    # splitted_inputs = tf.split(inputs, groups, axis=-1)
    # return tf.concat([conv2d_no_bias(splitted_inputs[ii], filters, kernel_size, name=name + "1_g{}_".format(ii + 1), **kwargs) for ii in range(groups)], axis=-1)


def contextual_transformer(inputs, kernel_size=3, activation="relu", name=""):
    # inputs, kernel_size, strides, activation, name = tf.ones([1, 7, 7, 512]), 3, 1, "relu", ""
    filters = inputs.shape[-1]
    randix = 2

    # key_embed
    if kernel_size // 2 != 0:
        key_input = layers.ZeroPadding2D(padding=kernel_size // 2, name=name + "conv_pad")(inputs)
    else:
        key_input = inputs
    key = group_conv(key_input, filters, kernel_size, groups=4, name=name + "key_")
    key = batchnorm_with_activation(key, activation=activation, zero_gamma=False, name=name + "key_")

    # query key
    qk = layers.Concatenate(axis=-1)([inputs, key])
    _, height, width, _ = qk.shape

    # embed weights from query and key, ignore `num_heads`, as it's set as `1`
    reduction = 8
    embed_ww = conv2d_no_bias(qk, filters // randix, 1, name=name + "embed_ww_1_")
    embed_ww = batchnorm_with_activation(embed_ww, activation=activation, zero_gamma=False, name=name + "embed_ww_1_")
    embed_ww = conv2d_no_bias(embed_ww, kernel_size * kernel_size * filters // reduction, 1, use_bias=True, name=name + "embed_ww_2_")
    embed_ww = GroupNormalization(groups=filters // reduction, epsilon=BATCH_NORM_EPSILON, name=name + "embed_ww_group_norm")(embed_ww)
    embed_ww = tf.reshape(embed_ww, (-1, height, width, filters // reduction, kernel_size * kernel_size))
    embed_ww = tf.expand_dims(tf.transpose(embed_ww, [0, 1, 2, 4, 3]), axis=-2) # expand dim on `reduction` axis

    # matmul, local_conv
    embed = conv2d_no_bias(inputs, filters, 1, name=name + "embed_1_")
    embed = batchnorm_with_activation(embed, activation=None, zero_gamma=False, name=name + "embed_1_")

    # unfold_j = torch.nn.Unfold(kernel_size=kernel_size, dilation=1, padding=1, stride=1)
    # x2 = unfold_j(bb).view(-1, reduction, filters // reduction, kernel_size * kernel_size, height, width)
    # y2 = (ww.unsqueeze(2) * x2.unsqueeze(1)).sum(-3).view(-1, filters, height, width)
    sizes, patch_strides = [1, kernel_size, kernel_size, 1], [1, 1, 1, 1]
    embed = layers.ZeroPadding2D(padding=kernel_size // 2, name=name + "embed_pad")(embed)
    embed = tf.image.extract_patches(embed, sizes=sizes, strides=patch_strides, rates=(1, 1, 1, 1), padding="VALID")
    embed = tf.reshape(embed, [-1, height, width, kernel_size * kernel_size, reduction, filters // reduction])

    embed_out = layers.Multiply(name=name + "local_conv_mul")([embed, embed_ww])
    embed_out = tf.reduce_sum(embed_out, axis=-3)    # reduce on `kernel_size * kernel_size` axis
    embed_out = tf.reshape(embed_out, [-1, height, width, filters])
    embed_out = batchnorm_with_activation(embed_out, activation="swish", zero_gamma=False, name=name + "embed_2_")

    # attention
    attn = layers.Add()([embed_out, key])
    attn = tf.reduce_mean(attn, axis=[1, 2], keepdims=True)
    # attn se module
    attn_se_filters = max(filters * randix // 4, 32)
    # attn = layers.Dense(attn_se_filters, use_bias=True, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + "attn_se_dense_1")(attn)
    attn = conv2d_no_bias(attn, attn_se_filters, 1, use_bias=True, name=name + "attn_se_1_")
    attn = batchnorm_with_activation(attn, activation=activation, zero_gamma=False, name=name + "attn_se_")
    # attn = layers.Dense(filters * randix, use_bias=True, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + "attn_se_dense_2")(attn)
    attn = conv2d_no_bias(attn, filters * randix, 1, use_bias=True, name=name + "attn_se_2_")
    attn = tf.reshape(attn, [-1, 1, 1, filters, randix])
    attn = tf.nn.softmax(attn, axis=-1)

    # value and output
    value = layers.Concatenate(axis=-1)([tf.expand_dims(embed_out, -1), tf.expand_dims(key, -1)])
    output = layers.Multiply()([value, attn])
    output = tf.reduce_sum(output, axis=-1, name=name + "out")
    return output


def cot_block(inputs, filters, strides=1, shortcut=False, expansion=4, cardinality=1, survival=None, use_se=0, activation="relu", name=""):
    # target_dimension = round(planes * block.expansion * self.rb)
    expanded_filter = round(filters * expansion)
    if shortcut:
        # print(">>>> Downsample")
        shortcut = conv2d_no_bias(inputs, expanded_filter, 1, strides=strides, name=name + "shorcut_")
        shortcut = batchnorm_with_activation(shortcut, activation=None, zero_gamma=False, name=name + "shorcut_")
    else:
        shortcut = inputs

    # width = planes
    nn = conv2d_no_bias(inputs, filters, 1, name=name + "1_")
    nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name=name + "1_")

    if strides > 1:
        nn = layers.ZeroPadding2D(padding=1, name=name + "pool_pad")(nn)
        nn = layers.AveragePooling2D(3, strides=2, name=name + "pool")(nn)

    nn = contextual_transformer(nn, 3, activation=activation, name=name + "cot_") if cardinality == 1 else coxt_layer(nn, 3)

    nn = conv2d_no_bias(nn, expanded_filter, 1, name=name + "2_")
    nn = batchnorm_with_activation(nn, activation=None, zero_gamma=True, name=name + "2_")

    if use_se:
        nn = se_module(nn, se_ratio=4 * expansion, name=name + "se_")

    # print(">>>>", nn.shape, shortcut.shape)
    if survival is not None and survival < 1:
        nn = StochasticDepth(float(survival))([shortcut, nn])
    else:
        nn = layers.Add()([shortcut, nn])
    return layers.Activation(activation, name=name + "out")(nn)


def cot_stack(inputs, blocks, filter, strides=2, expansion=4, cardinality=1, survival=None, use_se=0, activation="relu", name=""):
    shortcut = True if strides != 1 or inputs.shape[-1] != filter * expansion else False
    nn = cot_block(inputs, filter, strides, shortcut, expansion, cardinality, survival, use_se, activation, name=name + "block1_")
    shortcut = False
    for ii in range(2, blocks + 1):
        block_name = name + "block{}_".format(ii)
        nn = cot_block(nn, filter, 1, shortcut, expansion, cardinality, survival, use_se, activation, name=block_name)
    return nn


def CotNet(
    num_blocks,
    input_shape=(224, 224, 3),
    expansion=4,
    cardinality=1,
    activation="relu",
    use_se=0,
    num_classes=1000,
    pretrained=None,
    classifier_activation="softmax",
    model_name="cotnet",
    **kwargs
):
    inputs = keras.layers.Input(input_shape)
    nn = keras.layers.ZeroPadding2D(padding=3, name="stem_conv_pad")(inputs)
    nn = conv2d_no_bias(nn, 64, 7, strides=2, padding="VALID", name="stem_")
    nn = batchnorm_with_activation(nn, activation=activation, name="stem_")
    nn = keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name="stem_pool_pad")(nn)
    nn = keras.layers.MaxPooling2D(3, strides=2, name="stem_pool")(nn)

    out_channels = [64, 128, 256, 512]
    for id, (num_block, out_channel) in enumerate(zip(num_blocks, out_channels)):
        name = "stack{}_".format(id + 1)
        survival = None
        strides = 1 if id == 0 else 2
        nn = cot_stack(nn, num_block, out_channel, strides, expansion, cardinality, survival, use_se, activation=activation, name=name)

    if num_classes > 0:
        nn = keras.layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        nn = keras.layers.Dense(num_classes, activation=classifier_activation, name="predictions")(nn)

    model = keras.models.Model(inputs, nn, name=model_name)
    return model


def CotNet50(input_shape=(224, 224, 3), num_classes=1000, activation="relu", pretrained=None, **kwargs):
    return CotNet(num_blocks=[3, 4, 6, 3], model_name="cotnet50", **locals(), **kwargs)


def CotNet101(input_shape=(224, 224, 3), num_classes=1000, activation="relu", pretrained=None, **kwargs):
    return CotNet(num_blocks=[3, 4, 23, 3], model_name="cotnet101", **locals(), **kwargs)
