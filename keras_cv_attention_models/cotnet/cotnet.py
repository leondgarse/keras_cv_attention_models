import numpy as np
from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models, functional, image_data_format
from keras_cv_attention_models.aotnet import AotNet
from keras_cv_attention_models.download_and_load import reload_model_weights
from keras_cv_attention_models.attention_layers import batchnorm_with_activation, conv2d_no_bias, CompatibleExtractPatches, group_norm

BATCH_NORM_EPSILON = 1e-5
PRETRAINED_DICT = {
    "cotnet101": {"imagenet": {224: "589d2c817699d96085136f2af2dd2036"}},
    "cotnet50": {"imagenet": {224: "e2b1fd313834deebb1c1f83451525254"}},
    "cotnet_se101d": {"imagenet": {224: "4e2fc28b9a92259269f1b5544e67ab50"}},
    "cotnet_se152d": {"imagenet": {224: "6a2744af16b8cc4177fef52aba7ff083", 320: "9dad11a2ec3d2c8ecac9832fcf1e9ad3"}},
    "cotnet_se50d": {"imagenet": {224: "d1e40b172d26925794f0c9dea090dba7"}},
}


def cot_attention(inputs, kernel_size=3, strides=1, downsample_first=True, activation="relu", name=None):
    if downsample_first and strides > 1:
        inputs = layers.ZeroPadding2D(padding=1, name=name and name + "pool_pad")(inputs)
        inputs = layers.AvgPool2D(3, strides=2, name=name and name + "pool")(inputs)

    # inputs, kernel_size, strides, activation, name = tf.ones([1, 7, 7, 512]), 3, 1, "relu", ""
    height_axis, width_axis, channel_axis = (1, 2, 3) if image_data_format() == "channels_last" else (2, 3, 1)
    filters = inputs.shape[channel_axis]
    randix = 2

    # key_embed
    if kernel_size // 2 != 0:
        key_input = layers.ZeroPadding2D(padding=kernel_size // 2, name=name and name + "conv_pad")(inputs)
    else:
        key_input = inputs
    key = conv2d_no_bias(key_input, filters, kernel_size, groups=4, name=name and name + "key_")
    key = batchnorm_with_activation(key, activation=activation, zero_gamma=False, name=name and name + "key_")

    # query key
    qk = layers.Concatenate(axis=channel_axis)([inputs, key])
    height, width = qk.shape[height_axis], qk.shape[width_axis]

    # embed weights from query and key, ignore `num_heads`, as it's set as `1`
    reduction = 8
    embed_ww = conv2d_no_bias(qk, filters // randix, 1, name=name and name + "embed_ww_1_")
    embed_ww = batchnorm_with_activation(embed_ww, activation=activation, zero_gamma=False, name=name and name + "embed_ww_1_")
    embed_filters = kernel_size * kernel_size * filters // reduction
    embed_ww = conv2d_no_bias(embed_ww, embed_filters, 1, use_bias=True, name=name and name + "embed_ww_2_")
    embed_ww = group_norm(embed_ww, groups=filters // reduction, epsilon=BATCH_NORM_EPSILON, name=name and name + "embed_ww_")

    # matmul, local_conv
    embed = conv2d_no_bias(inputs, filters, 1, name=name and name + "embed_1_")
    embed = batchnorm_with_activation(embed, activation=None, zero_gamma=False, name=name and name + "embed_1_")

    # unfold_j = torch.nn.Unfold(kernel_size=kernel_size, dilation=1, padding=1, stride=1)
    # x2 = unfold_j(bb).view(-1, reduction, filters // reduction, kernel_size * kernel_size, height, width)
    # y2 = (ww.unsqueeze(2) * x2.unsqueeze(1)).sum(-3).view(-1, filters, height, width)
    # sizes, patch_strides = [1, kernel_size, kernel_size, 1], [1, 1, 1, 1]
    # embed = layers.ZeroPadding2D(padding=kernel_size // 2, name=name and name + "embed_pad")(embed)
    # embed = functional.extract_patches(embed, sizes=sizes, strides=patch_strides, rates=(1, 1, 1, 1), padding="VALID")
    embed = CompatibleExtractPatches(sizes=kernel_size, strides=1, name=name and name + "patchs_")(embed)

    if image_data_format() == "channels_last":
        embed_ww = functional.reshape(embed_ww, (-1, height, width, filters // reduction, kernel_size * kernel_size))
        reduction_axis, kernel_axis = -2, -3
        embed_ww = functional.expand_dims(functional.transpose(embed_ww, [0, 1, 2, 4, 3]), axis=reduction_axis)  # expand dim on `reduction` axis
        embed = functional.reshape(embed, [-1, height, width, kernel_size * kernel_size, reduction, filters // reduction])
    else:
        embed_ww = functional.reshape(embed_ww, (-1, filters // reduction, kernel_size * kernel_size, height, width))
        reduction_axis, kernel_axis = 2, 1
        embed_ww = functional.expand_dims(functional.transpose(embed_ww, [0, 2, 1, 3, 4]), axis=reduction_axis)  # expand dim on `reduction` axis
        embed = functional.reshape(embed, [-1, kernel_size * kernel_size, reduction, filters // reduction, height, width])

    embed_out = layers.Multiply(name=name and name + "local_conv_mul")([embed, embed_ww])
    embed_out = functional.reduce_sum(embed_out, axis=kernel_axis)  # reduce on `kernel_size * kernel_size` axis
    embed_out = functional.reshape(embed_out, [-1, height, width, filters] if image_data_format() == "channels_last" else [-1, filters, height, width])
    embed_out = batchnorm_with_activation(embed_out, activation="swish", zero_gamma=False, name=name and name + "embed_2_")

    # attention
    attn = layers.Add()([embed_out, key])
    attn = functional.reduce_mean(attn, axis=[height_axis, width_axis], keepdims=True)
    # attn se module
    attn_se_filters = max(filters * randix // 4, 32)
    # attn = layers.Dense(attn_se_filters, use_bias=True, name=name and name + "attn_se_dense_1")(attn)
    attn = conv2d_no_bias(attn, attn_se_filters, 1, use_bias=True, name=name and name + "attn_se_1_")
    attn = batchnorm_with_activation(attn, activation=activation, zero_gamma=False, name=name and name + "attn_se_")
    # attn = layers.Dense(filters * randix, use_bias=True, name=name and name + "attn_se_dense_2")(attn)
    attn = conv2d_no_bias(attn, filters * randix, 1, use_bias=True, name=name and name + "attn_se_2_")
    attn = functional.reshape(attn, [-1, 1, 1, filters, randix] if image_data_format() == "channels_last" else [-1, filters, randix, 1, 1])
    # attn = functional.nn.softmax(attn, axis=-1)
    randix_axis = -1 if image_data_format() == "channels_last" else 2
    attn = layers.Softmax(axis=randix_axis, name=name and name + "attention_scores")(attn)

    # value and output
    value = layers.Concatenate(axis=randix_axis)([functional.expand_dims(embed_out, randix_axis), functional.expand_dims(key, randix_axis)])
    output = layers.Multiply()([value, attn])
    output = functional.reduce_sum(output, axis=randix_axis, name=name and name + "out")

    if not downsample_first and strides > 1:
        output = layers.ZeroPadding2D(padding=1, name=name and name + "pool_pad")(output)
        output = layers.AvgPool2D(3, strides=2, name=name and name + "pool")(output)
    return output


def CotNet(input_shape=(224, 224, 3), bn_after_attn=False, shortcut_type="avg", attn_types="cot", pretrained="imagenet", **kwargs):
    model = AotNet(input_shape=input_shape, attn_types=attn_types, bn_after_attn=bn_after_attn, shortcut_type=shortcut_type, **kwargs)
    reload_model_weights(model, pretrained_dict=PRETRAINED_DICT, sub_release="cotnet", pretrained=pretrained)
    return model


def CotNet50(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 6, 3]
    strides = [1, 2, 2, 2]
    return CotNet(**locals(), **kwargs, model_name="cotnet50")


def CotNet101(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 23, 3]
    strides = [1, 2, 2, 2]
    return CotNet(**locals(), **kwargs, model_name="cotnet101")


def CotNetSE50D(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 6, 3]
    strides = [2, 2, 2, 2]
    attn_types = ["sa", "sa", ["cot", "sa"] * 3, "cot"]
    attn_params = [
        {"downsample_first": True, "groups": 1, "activation": "swish"},  # sa, stack1
        {"downsample_first": True, "groups": 1, "activation": "swish"},  # sa, stack2
        [{}, {"downsample_first": True, "groups": 1, "activation": "swish"}] * 3,  # cot + sa, stack 3
        {},  # cot, stack 4
    ]
    stem_type = "deep"
    stem_width = 64
    stem_downsample = False
    return CotNet(**locals(), **kwargs, model_name="cotnet_se50d")


def CotNetSE101D(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 23, 3]
    strides = [2, 2, 2, 2]
    attn_types = ["sa", "sa", ["cot", "sa"] * 12, "cot"]
    attn_params = [
        {"downsample_first": True, "groups": 1, "activation": "swish"},  # sa, stack1
        {"downsample_first": True, "groups": 1, "activation": "swish"},  # sa, stack2
        [{}, {"downsample_first": True, "groups": 1, "activation": "swish"}] * 12,  # cot + sa, stack 3
        {},  # cot, stack 4
    ]
    stem_type = "deep"
    stem_width = 128
    stem_downsample = False
    return CotNet(**locals(), **kwargs, model_name="cotnet_se101d")


def CotNetSE152D(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 8, 36, 3]
    strides = [2, 2, 2, 2]
    attn_types = ["sa", "sa", ["cot", "sa"] * 18, "cot"]
    attn_params = [
        {"downsample_first": True, "groups": 1, "activation": "swish"},  # sa, stack1
        {"downsample_first": True, "groups": 1, "activation": "swish"},  # sa, stack2
        [{"downsample_first": False}, {"downsample_first": True, "groups": 1, "activation": "swish"}] * 18,  # cot + sa, stack 3
        {"downsample_first": False},  # cot, stack 4
    ]
    stem_type = "deep"
    stem_width = 128
    stem_downsample = False
    return CotNet(**locals(), **kwargs, model_name="cotnet_se152d")
