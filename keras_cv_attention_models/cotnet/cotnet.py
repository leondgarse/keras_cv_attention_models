import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from keras_cv_attention_models.aotnet import AotNet
from keras_cv_attention_models.download_and_load import reload_model_weights
from keras_cv_attention_models.attention_layers import batchnorm_with_activation, conv2d_no_bias, CompatibleExtractPatches

BATCH_NORM_EPSILON = 1e-5
PRETRAINED_DICT = {
    "cotnet101": {"imagenet": {224: "589d2c817699d96085136f2af2dd2036"}},
    "cotnet50": {"imagenet": {224: "e2b1fd313834deebb1c1f83451525254"}},
    "cotnet_se101d": {"imagenet": {224: "4e2fc28b9a92259269f1b5544e67ab50"}},
    "cotnet_se152d": {"imagenet": {224: "6a2744af16b8cc4177fef52aba7ff083", 320: "9dad11a2ec3d2c8ecac9832fcf1e9ad3"}},
    "cotnet_se50d": {"imagenet": {224: "d1e40b172d26925794f0c9dea090dba7"}},
}


def group_conv(inputs, filters, kernel_size, groups=4, name="", **kwargs):
    # Using groups=num in `Conv2D` is slow with `mixed_float16` policy
    return conv2d_no_bias(inputs, filters, kernel_size, groups=groups, name=name)
    # splitted_inputs = tf.split(inputs, groups, axis=-1)
    # return tf.concat([conv2d_no_bias(splitted_inputs[ii], filters // groups, kernel_size, name=name + "g{}_".format(ii + 1), **kwargs) for ii in range(groups)], axis=-1)


def cot_attention(inputs, kernel_size=3, strides=1, downsample_first=True, activation="relu", name=None):
    from tensorflow_addons.layers import GroupNormalization

    if downsample_first and strides > 1:
        inputs = keras.layers.ZeroPadding2D(padding=1, name=name and name + "pool_pad")(inputs)
        inputs = keras.layers.AveragePooling2D(3, strides=2, name=name and name + "pool")(inputs)

    # inputs, kernel_size, strides, activation, name = tf.ones([1, 7, 7, 512]), 3, 1, "relu", ""
    filters = inputs.shape[-1]
    randix = 2

    # key_embed
    if kernel_size // 2 != 0:
        key_input = keras.layers.ZeroPadding2D(padding=kernel_size // 2, name=name and name + "conv_pad")(inputs)
    else:
        key_input = inputs
    key = group_conv(key_input, filters, kernel_size, groups=4, name=name and name + "key_")
    key = batchnorm_with_activation(key, activation=activation, zero_gamma=False, name=name and name + "key_")

    # query key
    qk = keras.layers.Concatenate(axis=-1)([inputs, key])
    _, height, width, _ = qk.shape

    # embed weights from query and key, ignore `num_heads`, as it's set as `1`
    reduction = 8
    embed_ww = conv2d_no_bias(qk, filters // randix, 1, name=name and name + "embed_ww_1_")
    embed_ww = batchnorm_with_activation(embed_ww, activation=activation, zero_gamma=False, name=name and name + "embed_ww_1_")
    embed_filters = kernel_size * kernel_size * filters // reduction
    embed_ww = conv2d_no_bias(embed_ww, embed_filters, 1, use_bias=True, name=name and name + "embed_ww_2_")
    embed_ww = GroupNormalization(groups=filters // reduction, epsilon=BATCH_NORM_EPSILON, name=name and name + "embed_ww_group_norm")(embed_ww)
    embed_ww = tf.reshape(embed_ww, (-1, height, width, filters // reduction, kernel_size * kernel_size))
    embed_ww = tf.expand_dims(tf.transpose(embed_ww, [0, 1, 2, 4, 3]), axis=-2)  # expand dim on `reduction` axis

    # matmul, local_conv
    embed = conv2d_no_bias(inputs, filters, 1, name=name and name + "embed_1_")
    embed = batchnorm_with_activation(embed, activation=None, zero_gamma=False, name=name and name + "embed_1_")

    # unfold_j = torch.nn.Unfold(kernel_size=kernel_size, dilation=1, padding=1, stride=1)
    # x2 = unfold_j(bb).view(-1, reduction, filters // reduction, kernel_size * kernel_size, height, width)
    # y2 = (ww.unsqueeze(2) * x2.unsqueeze(1)).sum(-3).view(-1, filters, height, width)
    # sizes, patch_strides = [1, kernel_size, kernel_size, 1], [1, 1, 1, 1]
    # embed = keras.layers.ZeroPadding2D(padding=kernel_size // 2, name=name and name + "embed_pad")(embed)
    # embed = tf.image.extract_patches(embed, sizes=sizes, strides=patch_strides, rates=(1, 1, 1, 1), padding="VALID")
    embed = CompatibleExtractPatches(sizes=kernel_size, strides=1, name=name and name + "patchs_")(embed)
    embed = tf.reshape(embed, [-1, height, width, kernel_size * kernel_size, reduction, filters // reduction])

    embed_out = keras.layers.Multiply(name=name and name + "local_conv_mul")([embed, embed_ww])
    embed_out = tf.reduce_sum(embed_out, axis=-3)  # reduce on `kernel_size * kernel_size` axis
    embed_out = tf.reshape(embed_out, [-1, height, width, filters])
    embed_out = batchnorm_with_activation(embed_out, activation="swish", zero_gamma=False, name=name and name + "embed_2_")

    # attention
    attn = keras.layers.Add()([embed_out, key])
    attn = tf.reduce_mean(attn, axis=[1, 2], keepdims=True)
    # attn se module
    attn_se_filters = max(filters * randix // 4, 32)
    # attn = keras.layers.Dense(attn_se_filters, use_bias=True, name=name and name + "attn_se_dense_1")(attn)
    attn = conv2d_no_bias(attn, attn_se_filters, 1, use_bias=True, name=name and name + "attn_se_1_")
    attn = batchnorm_with_activation(attn, activation=activation, zero_gamma=False, name=name and name + "attn_se_")
    # attn = keras.layers.Dense(filters * randix, use_bias=True, name=name and name + "attn_se_dense_2")(attn)
    attn = conv2d_no_bias(attn, filters * randix, 1, use_bias=True, name=name and name + "attn_se_2_")
    attn = tf.reshape(attn, [-1, 1, 1, filters, randix])
    # attn = tf.nn.softmax(attn, axis=-1)
    attn = keras.layers.Softmax(axis=-1, name=name and name + "attention_scores")(attn)

    # value and output
    value = keras.layers.Concatenate(axis=-1)([tf.expand_dims(embed_out, -1), tf.expand_dims(key, -1)])
    output = keras.layers.Multiply()([value, attn])
    output = tf.reduce_sum(output, axis=-1, name=name and name + "out")

    if not downsample_first and strides > 1:
        output = keras.layers.ZeroPadding2D(padding=1, name=name and name + "pool_pad")(output)
        output = keras.layers.AveragePooling2D(3, strides=2, name=name and name + "pool")(output)
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
