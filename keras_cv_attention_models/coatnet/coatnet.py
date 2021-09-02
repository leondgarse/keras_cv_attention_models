import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from keras_cv_attention_models.attention_layers import batchnorm_with_activation, conv2d_no_bias, se_module, mhsa_with_relative_position_embedding

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5
CONV_KERNEL_INITIALIZER = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")


def res_MBConv(inputs, output_channel, conv_short_cut=True, strides=1, expansion=4, se_ratio=0, activation="relu", name=None):
    # preact
    nn = batchnorm_with_activation(inputs, activation=activation, name=name + "preact_")

    if conv_short_cut:
        shortcut = keras.layers.AvgPool2D(strides, strides=strides, padding="SAME", name=name + "shorcut_pool")(nn) if strides > 1 else nn
        shortcut = conv2d_no_bias(shortcut, output_channel, 1, strides=1, name=name + "shorcut_")
    else:
        shortcut = inputs

    # MBConv
    input_channel = inputs.shape[-1]
    nn = conv2d_no_bias(nn, input_channel * expansion, 1, strides=1, padding="same", name=name + "expand_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "expand_")
    nn = keras.layers.DepthwiseConv2D(3, padding="same", strides=strides, use_bias=False, name=name + "MB_dw_")(nn)
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "MB_dw_")
    if se_ratio:
        nn = se_module(nn, se_ratio=se_ratio / expansion, name=name + "se_")
    nn = conv2d_no_bias(nn, output_channel, 1, strides=1, padding="same", name=name + "MB_pw_")

    return keras.layers.Add()([shortcut, nn])


def res_ffn(inputs, expansion=4, kernel_size=1, activation="relu", name=None):
    input_channel = inputs.shape[-1]
    nn = conv2d_no_bias(inputs, input_channel * expansion, kernel_size, name=name + "1_")
    nn = keras.layers.Activation(activation)(nn)
    nn = conv2d_no_bias(nn, input_channel, kernel_size, name=name + "2_")
    return keras.layers.Add()([inputs, nn])


def res_mhsa(inputs, output_channel, conv_short_cut=True, strides=1, num_heads=32, activation="relu", name=None):
    # preact
    nn = batchnorm_with_activation(inputs, activation=activation, name=name + "preact_")

    if conv_short_cut:
        shortcut = keras.layers.AvgPool2D(strides, strides=strides, padding="SAME", name=name + "shorcut_pool")(nn) if strides > 1 else nn
        shortcut = conv2d_no_bias(shortcut, output_channel, 1, strides=1, name=name + "shorcut_")
    else:
        shortcut = inputs

    if strides != 1:  # Downsample
        nn = keras.layers.ZeroPadding2D(padding=1, name=name + "pad")(nn)
        nn = keras.layers.AvgPool2D(pool_size=3, strides=strides, name=name + "pool")(nn)
    nn = mhsa_with_relative_position_embedding(nn, num_heads=num_heads, out_shape=output_channel, name=name + "mhsa")

    return keras.layers.Add()([shortcut, nn])


def CoAtNet(
    num_blocks,
    out_channels,
    stem_width=64,
    block_types=["conv", "conv", "transfrom", "transform"],
    expansion=4,
    se_ratio=0.25,
    num_heads=32,
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="relu",
    classifier_activation="softmax",
    pretrained=None,
    model_name="coatnet",
    **kwargs,
):
    inputs = keras.layers.Input(input_shape)

    """ stage 0, Stem_stage """
    nn = conv2d_no_bias(inputs, stem_width, 3, strides=2, padding="same", name="stem_1_")
    nn = batchnorm_with_activation(nn, activation=activation, name="stem_1_")
    nn = conv2d_no_bias(nn, stem_width, 3, strides=1, padding="same", name="stem_2_")

    """ stage [1, 2, 3, 4] """
    for stack_id, (num_block, out_channel, block_type) in enumerate(zip(num_blocks, out_channels, block_types)):
        is_conv_block = True if block_type[0].lower() == "c" else False
        stack_se_ratio = se_ratio[stack_id] if isinstance(se_ratio, (list, tuple)) else se_ratio
        for block_id in range(num_block):
            strides = 2 if block_id == 0 else 1
            conv_short_cut = True if block_id == 0 else False
            name = "stage_{}_block_{}_".format(stack_id + 1, block_id + 1)
            block_se_ratio = stack_se_ratio[block_id] if isinstance(stack_se_ratio, (list, tuple)) else stack_se_ratio
            if is_conv_block:
                nn = res_MBConv(nn, out_channel, conv_short_cut, strides=strides, expansion=expansion, se_ratio=block_se_ratio, name=name)
            else:
                nn = res_mhsa(nn, out_channel, conv_short_cut, strides=strides, num_heads=num_heads, name=name)
                nn = res_ffn(nn, expansion=expansion, activation=activation, name=name + "ffn_")

    if num_classes > 0:
        nn = keras.layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        nn = keras.layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="predictions")(nn)

    model = keras.models.Model(inputs, nn, name=model_name)
    return model


def CoAtNet0(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", **kwargs):
    num_blocks = [2, 3, 5, 2]
    out_channels = [96, 192, 384, 768]
    stem_width = 64
    return CoAtNet(**locals(), model_name="coatnet0", **kwargs)


def CoAtNet1(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", **kwargs):
    num_blocks = [2, 6, 14, 2]
    out_channels = [96, 192, 384, 768]
    stem_width = 64
    return CoAtNet(**locals(), model_name="coatnet1", **kwargs)


def CoAtNet2(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", **kwargs):
    num_blocks = [2, 6, 14, 2]
    out_channels = [128, 256, 512, 1024]
    stem_width = 128
    return CoAtNet(**locals(), model_name="coatnet2", **kwargs)


def CoAtNet3(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", **kwargs):
    num_blocks = [2, 6, 14, 2]
    out_channels = [192, 384, 768, 1536]
    stem_width = 192
    return CoAtNet(**locals(), model_name="coatnet3", **kwargs)


def CoAtNet4(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", **kwargs):
    num_blocks = [2, 12, 28, 2]
    out_channels = [192, 384, 768, 1536]
    stem_width = 192
    return CoAtNet(**locals(), model_name="coatnet4", **kwargs)


def CoAtNet5(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", **kwargs):
    num_blocks = [2, 12, 28, 2]
    out_channels = [256, 512, 1280, 2048]
    stem_width = 192
    num_heads = 64
    return CoAtNet(**locals(), model_name="coatnet5", **kwargs)
