"""
Creates a EfficientNetV2 Model as defined in: Mingxing Tan, Quoc V. Le. (2021). arXiv preprint arXiv:2104.00298.
EfficientNetV2: Smaller Models and Faster Training.
"""
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from keras_cv_attention_models.attention_layers import (
    batchnorm_with_activation,
    conv2d_no_bias,
    drop_block,
    make_divisible,
    output_block,
    se_module,
    add_pre_post_process,
)

TF_BATCH_NORM_EPSILON = 0.001
TORCH_BATCH_NORM_EPSILON = 1e-5


FILE_HASH_DICT = {
    "b0": {"21k-ft1k": "4e4da4eb629897e4d6271e131039fe75", "21k": "5dbb4252df24b931e74cdd94d150f25a", "imagenet": "9abdc43cb00f4cb06a8bdae881f412d6"},
    "b1": {"21k-ft1k": "5f1aee82209f4f0f20bd24460270564e", "21k": "a50ae65b50ceff7f5283be2f4506d2c2", "imagenet": "5d4223b59ff268828d5112a1630e234e"},
    "b2": {"21k-ft1k": "ec384b84441ddf6419938d1e5a0cbef2", "21k": "9f718a8bbb7b63c5313916c5e504790d", "imagenet": "1814bc08d4bb7a5e0ed3ccfe1cf18650"},
    "b3": {"21k-ft1k": "4a27827b0b2df508bed31ae231003bb1", "21k": "ade5bdbbdf1d54c4561aa41511525855", "imagenet": "cda85b8494c7ec5a68dffb335a254bab"},
    "l": {"21k-ft1k": "30327edcf1390d10e9a0de42a2d731e3", "21k": "7970f913eec1b4918e007c8580726412", "imagenet": "2b65f5789f4d2f1bf66ecd6d9c5c2d46"},
    "m": {"21k-ft1k": "0c236c3020e3857de1e5f2939abd0cc6", "21k": "3923c286366b2a5137f39d1e5b14e202", "imagenet": "ac3fd0ff91b35d18d1df8f1895efe1d5"},
    "s": {"21k-ft1k": "93046a0d601da46bfce9d4ca14224c83", "21k": "10b05d878b64f796ab984a5316a4a1c3", "imagenet": "3b91df2c50c7a56071cca428d53b8c0d"},
    "t": {"imagenet": "4a0ff9cb396665734d7ca590fa29681b"},
    "xl": {"21k-ft1k": "9aaa2bd3c9495b23357bc6593eee5bce", "21k": "c97de2770f55701f788644336181e8ee"},
    "v1-b0": {"noisy_student": "d125a518737c601f8595937219243432", "imagenet": "cc7d08887de9df8082da44ce40761986"},
    "v1-b1": {"noisy_student": "8f44bff58fc5ef99baa3f163b3f5c5e8", "imagenet": "a967f7be55a0125c898d650502c0cfd0"},
    "v1-b2": {"noisy_student": "b4ffed8b9262df4facc5e20557983ef8", "imagenet": "6c8d1d3699275c7d1867d08e219e00a7"},
    "v1-b3": {"noisy_student": "9d696365378a1ebf987d0e46a9d26ddd", "imagenet": "d78edb3dc7007721eda781c04bd4af62"},
    "v1-b4": {"noisy_student": "a0f61b977544493e6926186463d26294", "imagenet": "4c83aa5c86d58746a56675565d4f2051"},
    "v1-b5": {"noisy_student": "c3b6eb3f1f7a1e9de6d9a93e474455b1", "imagenet": "0bda50943b8e8d0fadcbad82c17c40f5"},
    "v1-b6": {"noisy_student": "20dd18b0df60cd7c0387c8af47bd96f8", "imagenet": "da13735af8209f675d7d7d03a54bfa27"},
    "v1-b7": {"noisy_student": "7f6f6dd4e8105e32432607ad28cfad0f", "imagenet": "d9c22b5b030d1e4f4c3a96dbf5f21ce6"},
    "v1-l2": {"noisy_student": "5fedc721febfca4b08b03d1f18a4a3ca"},
}


def MBConv(inputs, output_channel, stride, expand_ratio, shortcut, kernel_size=3, drop_rate=0, use_se=0, is_fused=False, is_torch_mode=False, name=""):
    channel_axis = 1 if K.image_data_format() == "channels_first" else -1
    input_channel = inputs.shape[channel_axis]
    bn_eps = TORCH_BATCH_NORM_EPSILON if is_torch_mode else TF_BATCH_NORM_EPSILON

    if is_fused and expand_ratio != 1:
        nn = conv2d_no_bias(inputs, input_channel * expand_ratio, 3, stride, padding="same", use_torch_padding=is_torch_mode, name=name + "sortcut_")
        nn = batchnorm_with_activation(nn, activation="swish", epsilon=bn_eps, name=name + "sortcut_")
    elif expand_ratio != 1:
        nn = conv2d_no_bias(inputs, input_channel * expand_ratio, 1, strides=1, padding="valid", use_torch_padding=is_torch_mode, name=name + "sortcut_")
        nn = batchnorm_with_activation(nn, activation="swish", epsilon=bn_eps, name=name + "sortcut_")
    else:
        nn = inputs

    if not is_fused:
        if is_torch_mode and kernel_size // 2 > 0:
            nn = keras.layers.ZeroPadding2D(padding=kernel_size // 2, name=name + "pad")(nn)
            padding = "VALID"
        else:
            padding = "SAME"
        nn = keras.layers.DepthwiseConv2D(kernel_size, padding=padding, strides=stride, use_bias=False, name=name + "MB_dw_")(nn)
        nn = batchnorm_with_activation(nn, activation="swish", epsilon=bn_eps, name=name + "MB_dw_")

    if use_se:
        nn = se_module(nn, se_ratio=1 / (4 * expand_ratio), divisor=1, activation="swish", name=name + "se_")

    # pw-linear
    if is_fused and expand_ratio == 1:
        nn = conv2d_no_bias(nn, output_channel, 3, strides=stride, padding="same", use_torch_padding=is_torch_mode, name=name + "fu_")
        nn = batchnorm_with_activation(nn, activation="swish", epsilon=bn_eps, name=name + "fu_")
    else:
        nn = conv2d_no_bias(nn, output_channel, 1, strides=1, padding="valid", use_torch_padding=is_torch_mode, name=name + "MB_pw_")
        nn = batchnorm_with_activation(nn, activation=None, epsilon=bn_eps, name=name + "MB_pw_")

    if shortcut:
        nn = drop_block(nn, drop_rate, name=name + "drop")
        return keras.layers.Add(name=name + "output")([inputs, nn])
    else:
        return keras.layers.Activation("linear", name=name + "output")(nn)  # Identity, Just need a name here


def EfficientNetV2(
    model_type,
    input_shape=(None, None, 3),
    num_classes=1000,
    dropout=0.2,
    first_strides=2,
    is_torch_mode=False,
    drop_connect_rate=0,
    classifier_activation="softmax",
    include_preprocessing=False,
    pretrained="imagenet",
    model_name="EfficientNetV2",
    kwargs=None,  # Not used, just recieving parameter
):
    if isinstance(model_type, dict):  # For EfficientNetV1 configures
        model_type, blocks_config = model_type.popitem()
    else:
        blocks_config = BLOCK_CONFIGS.get(model_type.lower(), BLOCK_CONFIGS["s"])
    expands = blocks_config["expands"]
    out_channels = blocks_config["out_channels"]
    depthes = blocks_config["depthes"]
    strides = blocks_config["strides"]
    use_ses = blocks_config["use_ses"]
    first_conv_filter = blocks_config.get("first_conv_filter", out_channels[0])
    output_conv_filter = blocks_config.get("output_conv_filter", 1280)
    kernel_sizes = blocks_config.get("kernel_sizes", [3] * len(depthes))
    # "torch" for all V1 models
    # for V2 models, "21k" pretrained are all "tf", "imagenet" pretrained "bx" models are all "torch", ["s", "m", "l", "xl"] are "tf"
    rescale_mode = "tf" if pretrained is not None and pretrained.startswith("imagenet21k") else blocks_config.get("rescale_mode", "torch")

    inputs = keras.layers.Input(shape=input_shape)
    bn_eps = TORCH_BATCH_NORM_EPSILON if is_torch_mode else TF_BATCH_NORM_EPSILON

    if include_preprocessing and rescale_mode == "torch":
        channel_axis = 1 if K.image_data_format() == "channels_first" else -1
        Normalization = keras.layers.Normalization if hasattr(keras.layers, "Normalization") else keras.layers.experimental.preprocessing.Normalization
        mean = tf.constant([0.485, 0.456, 0.406]) * 255.0
        std = (tf.constant([0.229, 0.224, 0.225]) * 255.0) ** 2
        nn = Normalization(mean=mean, variance=std, axis=channel_axis)(inputs)
    elif include_preprocessing and rescale_mode == "tf":
        Rescaling = keras.layers.Rescaling if hasattr(keras.layers, "Rescaling") else keras.layers.experimental.preprocessing.Rescaling
        nn = Rescaling(scale=1.0 / 128.0, offset=-1)(inputs)
    else:
        nn = inputs
    out_channel = make_divisible(first_conv_filter, 8)
    nn = conv2d_no_bias(nn, out_channel, 3, strides=first_strides, padding="same", use_torch_padding=is_torch_mode, name="stem_")
    nn = batchnorm_with_activation(nn, activation="swish", epsilon=bn_eps, name="stem_")

    pre_out = out_channel
    global_block_id = 0
    total_blocks = sum(depthes)
    for id, (expand, out_channel, depth, stride, se, kernel_size) in enumerate(zip(expands, out_channels, depthes, strides, use_ses, kernel_sizes)):
        out = make_divisible(out_channel, 8)
        is_fused = True if se == 0 else False
        for block_id in range(depth):
            stride = stride if block_id == 0 else 1
            shortcut = True if out == pre_out and stride == 1 else False
            name = "stack_{}_block{}_".format(id, block_id)
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            nn = MBConv(nn, out, stride, expand, shortcut, kernel_size, block_drop_rate, se, is_fused, is_torch_mode, name=name)
            pre_out = out
            global_block_id += 1

    output_conv_filter = make_divisible(output_conv_filter, 8)
    nn = conv2d_no_bias(nn, output_conv_filter, 1, strides=1, padding="valid", use_torch_padding=is_torch_mode, name="post_")
    nn = batchnorm_with_activation(nn, activation="swish", epsilon=bn_eps, name="post_")
    nn = output_block(nn, num_classes=num_classes, drop_rate=dropout, classifier_activation=classifier_activation)

    model = keras.models.Model(inputs=inputs, outputs=nn, name=model_name)
    add_pre_post_process(model, rescale_mode="raw" if include_preprocessing else rescale_mode)
    reload_model_weights(model, model_type, pretrained)
    return model


def reload_model_weights(model, model_type, pretrained="imagenet"):
    if isinstance(pretrained, str) and pretrained.endswith(".h5"):
        print(">>>> Load pretrained from:", pretrained)
        model.load_weights(pretrained, by_name=True, skip_mismatch=True)
        return

    pretrained_dd = {"imagenet": "imagenet", "imagenet21k": "21k", "imagenet21k-ft1k": "21k-ft1k", "noisy_student": "noisy_student"}
    if not pretrained in pretrained_dd:
        print(">>>> No pretrained available, model will be randomly initialized")
        return
    pre_tt = pretrained_dd[pretrained]
    if model_type not in FILE_HASH_DICT or pre_tt not in FILE_HASH_DICT[model_type]:
        print(">>>> No pretrained available, model will be randomly initialized")
        return

    if model_type.startswith("v1"):
        pre_url = "https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnet{}-{}.h5"
    else:
        pre_url = "https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnetv2-{}-{}.h5"
    url = pre_url.format(model_type, pre_tt)
    file_name = os.path.basename(url)
    file_hash = FILE_HASH_DICT[model_type][pre_tt]

    try:
        pretrained_model = keras.utils.get_file(file_name, url, cache_subdir="models/efficientnetv2", file_hash=file_hash)
    except:
        print("[Error] will not load weights, url not found or download failed:", url)
        return
    else:
        print(">>>> Load pretrained from:", pretrained_model)
        model.load_weights(pretrained_model, by_name=True, skip_mismatch=True)


BLOCK_CONFIGS = {
    "b0": {  # width 1.0, depth 1.0
        "first_conv_filter": 32,
        "expands": [1, 4, 4, 4, 6, 6],
        "out_channels": [16, 32, 48, 96, 112, 192],
        "depthes": [1, 2, 2, 3, 5, 8],
        "strides": [1, 2, 2, 2, 1, 2],
        "use_ses": [0, 0, 0, 1, 1, 1],
        "rescale_mode": "torch",
    },
    "b1": {  # width 1.0, depth 1.1
        "first_conv_filter": 32,
        "expands": [1, 4, 4, 4, 6, 6],
        "out_channels": [16, 32, 48, 96, 112, 192],
        "depthes": [2, 3, 3, 4, 6, 9],
        "strides": [1, 2, 2, 2, 1, 2],
        "use_ses": [0, 0, 0, 1, 1, 1],
        "rescale_mode": "torch",
    },
    "b2": {  # width 1.1, depth 1.2
        "first_conv_filter": 32,
        "output_conv_filter": 1408,
        "expands": [1, 4, 4, 4, 6, 6],
        "out_channels": [16, 32, 56, 104, 120, 208],
        "depthes": [2, 3, 3, 4, 6, 10],
        "strides": [1, 2, 2, 2, 1, 2],
        "use_ses": [0, 0, 0, 1, 1, 1],
        "rescale_mode": "torch",
    },
    "b3": {  # width 1.2, depth 1.4
        "first_conv_filter": 40,
        "output_conv_filter": 1536,
        "expands": [1, 4, 4, 4, 6, 6],
        "out_channels": [16, 40, 56, 112, 136, 232],
        "depthes": [2, 3, 3, 5, 7, 12],
        "strides": [1, 2, 2, 2, 1, 2],
        "use_ses": [0, 0, 0, 1, 1, 1],
        "rescale_mode": "torch",
    },
    "t": {  # width 1.4 * 0.8, depth 1.8 * 0.9, from timm
        "first_conv_filter": 24,
        "output_conv_filter": 1024,
        "expands": [1, 4, 4, 4, 6, 6],
        "out_channels": [24, 40, 48, 104, 128, 208],
        "depthes": [2, 4, 4, 6, 9, 14],
        "strides": [1, 2, 2, 2, 1, 2],
        "use_ses": [0, 0, 0, 1, 1, 1],
        "rescale_mode": "torch",
    },
    "s": {  # width 1.4, depth 1.8
        "first_conv_filter": 24,
        "output_conv_filter": 1280,
        "expands": [1, 4, 4, 4, 6, 6],
        "out_channels": [24, 48, 64, 128, 160, 256],
        "depthes": [2, 4, 4, 6, 9, 15],
        "strides": [1, 2, 2, 2, 1, 2],
        "use_ses": [0, 0, 0, 1, 1, 1],
        "rescale_mode": "tf",
    },
    "early": {  # S model discribed in paper early version https://arxiv.org/pdf/2104.00298v2.pdf
        "first_conv_filter": 24,
        "output_conv_filter": 1792,
        "expands": [1, 4, 4, 4, 6, 6],
        "out_channels": [24, 48, 64, 128, 160, 272],
        "depthes": [2, 4, 4, 6, 9, 15],
        "strides": [1, 2, 2, 2, 1, 2],
        "use_ses": [0, 0, 0, 1, 1, 1],
        "rescale_mode": "tf",
    },
    "m": {  # width 1.6, depth 2.2
        "first_conv_filter": 24,
        "output_conv_filter": 1280,
        "expands": [1, 4, 4, 4, 6, 6, 6],
        "out_channels": [24, 48, 80, 160, 176, 304, 512],
        "depthes": [3, 5, 5, 7, 14, 18, 5],
        "strides": [1, 2, 2, 2, 1, 2, 1],
        "use_ses": [0, 0, 0, 1, 1, 1, 1],
        "rescale_mode": "tf",
    },
    "l": {  # width 2.0, depth 3.1
        "first_conv_filter": 32,
        "output_conv_filter": 1280,
        "expands": [1, 4, 4, 4, 6, 6, 6],
        "out_channels": [32, 64, 96, 192, 224, 384, 640],
        "depthes": [4, 7, 7, 10, 19, 25, 7],
        "strides": [1, 2, 2, 2, 1, 2, 1],
        "use_ses": [0, 0, 0, 1, 1, 1, 1],
        "rescale_mode": "tf",
    },
    "xl": {
        "first_conv_filter": 32,
        "output_conv_filter": 1280,
        "expands": [1, 4, 4, 4, 6, 6, 6],
        "out_channels": [32, 64, 96, 192, 256, 512, 640],
        "depthes": [4, 8, 8, 16, 24, 32, 8],
        "strides": [1, 2, 2, 2, 1, 2, 1],
        "use_ses": [0, 0, 0, 1, 1, 1, 1],
        "rescale_mode": "tf",
    },
}


def EfficientNetV2B0(input_shape=(224, 224, 3), num_classes=1000, dropout=0.2, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return EfficientNetV2(model_type="b0", model_name="EfficientNetV2B0", **locals(), **kwargs)


def EfficientNetV2B1(input_shape=(240, 240, 3), num_classes=1000, dropout=0.2, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return EfficientNetV2(model_type="b1", model_name="EfficientNetV2B1", **locals(), **kwargs)


def EfficientNetV2B2(input_shape=(260, 260, 3), num_classes=1000, dropout=0.3, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return EfficientNetV2(model_type="b2", model_name="EfficientNetV2B2", **locals(), **kwargs)


def EfficientNetV2B3(input_shape=(300, 300, 3), num_classes=1000, dropout=0.3, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return EfficientNetV2(model_type="b3", model_name="EfficientNetV2B3", **locals(), **kwargs)


def EfficientNetV2T(input_shape=(320, 320, 3), num_classes=1000, dropout=0.2, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    is_torch_mode = True
    return EfficientNetV2(model_type="t", model_name="EfficientNetV2T", **locals(), **kwargs)


def EfficientNetV2S(input_shape=(384, 384, 3), num_classes=1000, dropout=0.2, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return EfficientNetV2(model_type="s", model_name="EfficientNetV2S", **locals(), **kwargs)


def EfficientNetV2M(input_shape=(480, 480, 3), num_classes=1000, dropout=0.3, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return EfficientNetV2(model_type="m", model_name="EfficientNetV2M", **locals(), **kwargs)


def EfficientNetV2L(input_shape=(480, 480, 3), num_classes=1000, dropout=0.4, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return EfficientNetV2(model_type="l", model_name="EfficientNetV2L", **locals(), **kwargs)


def EfficientNetV2XL(input_shape=(512, 512, 3), num_classes=1000, dropout=0.4, classifier_activation="softmax", pretrained="imagenet21k-ft1k", **kwargs):
    return EfficientNetV2(model_type="xl", model_name="EfficientNetV2XL", **locals(), **kwargs)
