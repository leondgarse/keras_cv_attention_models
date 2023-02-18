import numpy as np
from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models, functional, image_data_format
from keras_cv_attention_models.aotnet import AotNet
from keras_cv_attention_models.download_and_load import reload_model_weights
from keras_cv_attention_models.attention_layers import batchnorm_with_activation, conv2d_no_bias

PRETRAINED_DICT = {
    "resnest101": {"imagenet": "63f9ebdcd32529cbc4b4fbbec3d1bb2f"},
    "resnest200": {"imagenet": "8e211dcb089b588e18d36ba7cdf92ef0"},
    "resnest269": {"imagenet": "4309ed1b0a8ae92f2b1143dc3512c5c7"},
    "resnest50": {"imagenet": "eee7b20a229821f730ab205b6afeb369"},
}


def rsoftmax(inputs, groups):
    if groups > 1:
        nn = functional.reshape(inputs, [-1, groups, np.prod(inputs.shape[1:]) // groups])
        # nn = tf.transpose(nn, [0, 2, 1, 3])
        nn = functional.softmax(nn, axis=1)
        nn = functional.reshape(nn, [-1, *inputs.shape[1:]])
    else:
        nn = layers.Activation("sigmoid")(inputs)
    return nn


def split_attention_conv2d(inputs, filters, kernel_size=3, strides=1, downsample_first=False, groups=2, activation="relu", name=""):
    h_axis, w_axis = [1, 2] if image_data_format() == "channels_last" else [2, 3]
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    in_channels = inputs.shape[channel_axis]
    conv_strides = strides if downsample_first else 1
    if groups == 1:
        logits = conv2d_no_bias(inputs, filters, kernel_size, strides=conv_strides, padding="same", name=name and name + "1_")
    else:
        # Using groups=2 is slow in `mixed_float16` policy
        # logits = conv2d_no_bias(inputs, filters * groups, kernel_size, padding="same", groups=groups, name=name and name + "1_")
        logits = []
        splitted_inputs = functional.split(inputs, groups, axis=channel_axis)
        for ii in range(groups):
            conv_name = name and name + "1_g{}_".format(ii + 1)
            logits.append(conv2d_no_bias(splitted_inputs[ii], filters, kernel_size, strides=conv_strides, padding="same", name=conv_name))
        logits = functional.concat(logits, axis=channel_axis)
    logits = batchnorm_with_activation(logits, activation=activation, name=name and name + "1_")

    if groups > 1:
        splited = functional.split(logits, groups, axis=channel_axis)
        gap = functional.reduce_sum(splited, axis=0)
    else:
        gap = logits
    gap = functional.reduce_mean(gap, [h_axis, w_axis], keepdims=True)

    reduction_factor = 4
    inter_channels = max(in_channels * groups // reduction_factor, 32)
    atten = layers.Conv2D(inter_channels, kernel_size=1, name=name and name + "2_conv")(gap)
    atten = batchnorm_with_activation(atten, activation=activation, name=name and name + "2_")
    atten = layers.Conv2D(filters * groups, kernel_size=1, name=name and name + "3_conv")(atten)
    atten = rsoftmax(atten, groups)
    out = layers.Multiply()([atten, logits])

    if groups > 1:
        out = functional.split(out, groups, axis=channel_axis)
        out = functional.reduce_sum(out, axis=0)

    if not downsample_first and strides > 1:
        out = layers.ZeroPadding2D(padding=1, name=name and name + "pool_pad")(out)
        out = layers.AvgPool2D(3, strides=2, name=name and name + "pool")(out)
    return out


def ResNest(
    num_blocks=[3, 4, 6, 3],
    stem_width=64,
    stem_type="deep",
    attn_types="sa",
    bn_after_attn=False,
    shortcut_type="avg",
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="relu",
    classifier_activation="softmax",
    pretrained="imagenet",
    model_name="",
    **kwargs
):
    kwargs.pop("kwargs", None)
    model = AotNet(**locals(), **kwargs)
    reload_model_weights(model, pretrained_dict=PRETRAINED_DICT, sub_release="resnest", pretrained=pretrained)
    return model


def ResNest50(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", groups=2, **kwargs):
    return ResNest(num_blocks=[3, 4, 6, 3], stem_width=64, model_name="resnest50", **locals(), **kwargs)


def ResNest101(input_shape=(256, 256, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", groups=2, **kwargs):
    return ResNest(num_blocks=[3, 4, 23, 3], stem_width=128, model_name="resnest101", **locals(), **kwargs)


def ResNest200(input_shape=(320, 320, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", groups=2, **kwargs):
    return ResNest(num_blocks=[3, 24, 36, 3], stem_width=128, model_name="resnest200", **locals(), **kwargs)


def ResNest269(input_shape=(416, 416, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", groups=2, **kwargs):
    return ResNest(num_blocks=[3, 30, 48, 8], stem_width=128, model_name="resnest269", **locals(), **kwargs)
