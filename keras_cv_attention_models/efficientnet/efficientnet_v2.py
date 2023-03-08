"""
Creates a EfficientNetV2 Model as defined in: Mingxing Tan, Quoc V. Le. (2021). arXiv preprint arXiv:2104.00298.
EfficientNetV2: Smaller Models and Faster Training.
"""
import numpy as np
from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, functional, models, is_channels_last
from keras_cv_attention_models.attention_layers import (
    batchnorm_with_activation,
    conv2d_no_bias,
    drop_block,
    global_context_module,
    make_divisible,
    output_block,
    se_module,
    add_pre_post_process,
)

TF_BATCH_NORM_EPSILON = 0.001
TORCH_BATCH_NORM_EPSILON = 1e-5


FILE_HASH_DICT = {
    "v2-b0": {"21k-ft1k": "4e4da4eb629897e4d6271e131039fe75", "21k": "5dbb4252df24b931e74cdd94d150f25a", "imagenet": "9abdc43cb00f4cb06a8bdae881f412d6"},
    "v2-b1": {"21k-ft1k": "5f1aee82209f4f0f20bd24460270564e", "21k": "a50ae65b50ceff7f5283be2f4506d2c2", "imagenet": "5d4223b59ff268828d5112a1630e234e"},
    "v2-b2": {"21k-ft1k": "ec384b84441ddf6419938d1e5a0cbef2", "21k": "9f718a8bbb7b63c5313916c5e504790d", "imagenet": "1814bc08d4bb7a5e0ed3ccfe1cf18650"},
    "v2-b3": {"21k-ft1k": "4a27827b0b2df508bed31ae231003bb1", "21k": "ade5bdbbdf1d54c4561aa41511525855", "imagenet": "cda85b8494c7ec5a68dffb335a254bab"},
    "v2-l": {"21k-ft1k": "30327edcf1390d10e9a0de42a2d731e3", "21k": "7970f913eec1b4918e007c8580726412", "imagenet": "2b65f5789f4d2f1bf66ecd6d9c5c2d46"},
    "v2-m": {"21k-ft1k": "0c236c3020e3857de1e5f2939abd0cc6", "21k": "3923c286366b2a5137f39d1e5b14e202", "imagenet": "ac3fd0ff91b35d18d1df8f1895efe1d5"},
    "v2-s": {"21k-ft1k": "93046a0d601da46bfce9d4ca14224c83", "21k": "10b05d878b64f796ab984a5316a4a1c3", "imagenet": "3b91df2c50c7a56071cca428d53b8c0d"},
    "v2-t": {"imagenet": "4a0ff9cb396665734d7ca590fa29681b"},
    "v2-t-gc": {"imagenet": "653fc06396f9503dff61aa17c40b2443"},
    "v2-xl": {"21k-ft1k": "9aaa2bd3c9495b23357bc6593eee5bce", "21k": "c97de2770f55701f788644336181e8ee"},
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


def inverted_residual_block(
    inputs,
    output_channel,
    stride=1,
    expand=4,
    shortcut=False,
    kernel_size=3,
    drop_rate=0,
    se_ratio=0,
    is_fused=False,
    is_torch_mode=False,
    se_activation=None,  # None for same with activation
    se_divisor=1,  # 8 for mobilenetv3
    se_limit_round_down=0.9,  # 0.95 for fbnet
    use_global_context_instead_of_se=False,
    use_last_bn_zero_gamma=False,
    activation="swish",
    name=None,
):
    input_channel = inputs.shape[-1 if is_channels_last() else 1]
    bn_eps = TORCH_BATCH_NORM_EPSILON if is_torch_mode else TF_BATCH_NORM_EPSILON
    hidden_channel = make_divisible(input_channel * expand, 8)

    if is_fused and expand != 1:
        nn = conv2d_no_bias(inputs, hidden_channel, 3, stride, padding="same", use_torch_padding=is_torch_mode, name=name and name + "sortcut_")
        nn = batchnorm_with_activation(nn, activation=activation, epsilon=bn_eps, name=name and name + "sortcut_")
    elif expand != 1:
        nn = conv2d_no_bias(inputs, hidden_channel, 1, strides=1, padding="valid", use_torch_padding=is_torch_mode, name=name and name + "sortcut_")
        nn = batchnorm_with_activation(nn, activation=activation, epsilon=bn_eps, name=name and name + "sortcut_")
    else:
        nn = inputs

    if not is_fused:
        if is_torch_mode and backend.is_tensorflow_backend and kernel_size // 2 > 0:
            nn = layers.ZeroPadding2D(padding=kernel_size // 2, name=name and name + "pad")(nn)
            padding = "VALID"
        elif is_torch_mode and kernel_size // 2 > 0:
            padding = kernel_size // 2
        else:
            padding = "SAME"
        nn = layers.DepthwiseConv2D(kernel_size, padding=padding, strides=stride, use_bias=False, name=name and name + "MB_dw_")(nn)
        nn = batchnorm_with_activation(nn, activation=activation, epsilon=bn_eps, name=name and name + "MB_dw_")

    if se_ratio > 0:
        se_activation = activation if se_activation is None else se_activation
        se_ratio = se_ratio / expand
        if use_global_context_instead_of_se:
            nn = global_context_module(nn, use_attn=True, ratio=se_ratio, divisor=1, activation=se_activation, use_bias=True, name=name and name + "gc_")
        else:
            nn = se_module(nn, se_ratio, divisor=se_divisor, limit_round_down=se_limit_round_down, activation=se_activation, name=name and name + "se_")

    # pw-linear
    if is_fused and expand == 1:
        nn = conv2d_no_bias(nn, output_channel, 3, strides=stride, padding="same", use_torch_padding=is_torch_mode, name=name and name + "fu_")
        nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=use_last_bn_zero_gamma, epsilon=bn_eps, name=name and name + "fu_")
    else:
        nn = conv2d_no_bias(nn, output_channel, 1, strides=1, padding="valid", use_torch_padding=is_torch_mode, name=name and name + "MB_pw_")
        nn = batchnorm_with_activation(nn, activation=None, zero_gamma=use_last_bn_zero_gamma, epsilon=bn_eps, name=name and name + "MB_pw_")

    if shortcut:
        nn = drop_block(nn, drop_rate, name=name and name + "drop")
        return layers.Add(name=name and name + "output")([inputs, nn])
    else:
        return layers.Activation("linear", name=name and name + "output")(nn)  # Identity, Just need a name here


def EfficientNetV2(
    expands=[1, 4, 4, 4, 6, 6],
    out_channels=[16, 32, 48, 96, 112, 192],
    depthes=[1, 2, 2, 3, 5, 8],
    strides=[1, 2, 2, 2, 1, 2],
    se_ratios=[0, 0, 0, 0.25, 0.25, 0.25],
    is_fused="auto",  # True if se_ratio == 0 else False
    first_conv_filter=32,
    output_conv_filter=1280,
    kernel_sizes=3,
    input_shape=(None, None, 3),
    num_classes=1000,
    dropout=0.2,
    first_strides=2,
    is_torch_mode=False,
    use_global_context_instead_of_se=False,
    drop_connect_rate=0,
    activation="swish",
    classifier_activation="softmax",
    include_preprocessing=False,
    pretrained="imagenet",
    model_name="EfficientNetV2",
    rescale_mode="torch",
    kwargs=None,
):
    # "torch" for all V1 models
    # for V2 models, "21k" pretrained are all "tf", "imagenet" pretrained "bx" models are all "torch", ["s", "m", "l", "xl"] are "tf"
    rescale_mode = "tf" if pretrained is not None and pretrained.startswith("imagenet21k") else rescale_mode

    # Regard input_shape as force using original shape if len(input_shape) == 4,
    # else assume channel dimention is the one with min value in input_shape, and put it first or last regarding image_data_format
    input_shape = backend.align_input_shape_by_image_data_format(input_shape)
    inputs = layers.Input(shape=input_shape)
    bn_eps = TORCH_BATCH_NORM_EPSILON if is_torch_mode else TF_BATCH_NORM_EPSILON

    if include_preprocessing and rescale_mode == "torch":
        channel_axis = 1 if backend.image_data_format() == "channels_first" else -1
        Normalization = layers.Normalization if hasattr(layers, "Normalization") else layers.experimental.preprocessing.Normalization
        mean = np.array([0.485, 0.456, 0.406], dtype="float32") * 255.0
        std = (np.array([0.229, 0.224, 0.225], dtype="float32") * 255.0) ** 2
        nn = Normalization(mean=mean, variance=std, axis=channel_axis)(inputs)
    elif include_preprocessing and rescale_mode == "tf":
        Rescaling = layers.Rescaling if hasattr(layers, "Rescaling") else layers.experimental.preprocessing.Rescaling
        nn = Rescaling(scale=1.0 / 128.0, offset=-1)(inputs)
    else:
        nn = inputs
    stem_width = make_divisible(first_conv_filter, 8)
    nn = conv2d_no_bias(nn, stem_width, 3, strides=first_strides, padding="same", use_torch_padding=is_torch_mode, name="stem_")
    nn = batchnorm_with_activation(nn, activation=activation, epsilon=bn_eps, name="stem_")

    blocks_kwargs = {  # common for all blocks
        "is_torch_mode": is_torch_mode,
        "use_global_context_instead_of_se": use_global_context_instead_of_se,
    }

    pre_out = stem_width
    global_block_id = 0
    total_blocks = sum(depthes)
    kernel_sizes = kernel_sizes if isinstance(kernel_sizes, (list, tuple)) else ([kernel_sizes] * len(depthes))
    for id, (expand, out_channel, depth, stride, se_ratio, kernel_size) in enumerate(zip(expands, out_channels, depthes, strides, se_ratios, kernel_sizes)):
        out = make_divisible(out_channel, 8)
        if is_fused == "auto":
            cur_is_fused = True if se_ratio == 0 else False
        else:
            cur_is_fused = is_fused[id] if isinstance(is_fused, (list, tuple)) else is_fused
        for block_id in range(depth):
            name = "stack_{}_block{}_".format(id, block_id)
            stride = stride if block_id == 0 else 1
            shortcut = True if out == pre_out and stride == 1 else False
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            nn = inverted_residual_block(
                nn, out, stride, expand, shortcut, kernel_size, block_drop_rate, se_ratio, cur_is_fused, **blocks_kwargs, activation=activation, name=name
            )
            pre_out = out
            global_block_id += 1

    if output_conv_filter > 0:
        output_conv_filter = make_divisible(output_conv_filter, 8)
        nn = conv2d_no_bias(nn, output_conv_filter, 1, strides=1, padding="valid", use_torch_padding=is_torch_mode, name="post_")
        nn = batchnorm_with_activation(nn, activation=activation, epsilon=bn_eps, name="post_")
    nn = output_block(nn, num_classes=num_classes, drop_rate=dropout, classifier_activation=classifier_activation)

    model = models.Model(inputs=inputs, outputs=nn, name=model_name)
    add_pre_post_process(model, rescale_mode="raw" if include_preprocessing else rescale_mode)
    reload_model_weights(model, pretrained)
    return model


def reload_model_weights(model, pretrained="imagenet"):
    import os

    if pretrained is None:
        return
    if isinstance(pretrained, str) and pretrained.endswith(".h5"):
        print(">>>> Load pretrained from:", pretrained)
        model.load_weights(pretrained, by_name=True, skip_mismatch=True)
        return

    pretrained_dd = {"imagenet": "imagenet", "imagenet21k": "21k", "imagenet21k-ft1k": "21k-ft1k", "noisy_student": "noisy_student"}
    if not pretrained in pretrained_dd:
        print(">>>> No pretrained available, model will be randomly initialized")
        return
    pre_tt = pretrained_dd[pretrained]
    model_type = model.name.split("_")[-1]
    if model_type not in FILE_HASH_DICT or pre_tt not in FILE_HASH_DICT[model_type]:
        print(">>>> No pretrained available, model will be randomly initialized")
        return

    if model_type.startswith("v1"):
        pre_url = "https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv1_pretrained/efficientnet{}-{}.h5"
    else:
        pre_url = "https://github.com/leondgarse/keras_efficientnet_v2/releases/download/effnetv2_pretrained/efficientnet{}-{}.h5"
    url = pre_url.format(model_type, pre_tt)
    file_name = os.path.basename(url)
    file_hash = FILE_HASH_DICT[model_type][pre_tt]

    try:
        pretrained_model = backend.get_file(file_name, url, cache_subdir="models/efficientnetv2", file_hash=file_hash)
    except:
        print("[Error] will not load weights, url not found or download failed:", url)
        return
    else:
        print(">>>> Load pretrained from:", pretrained_model)
        model.load_weights(pretrained_model, by_name=True, skip_mismatch=True)


def EfficientNetV2B0(input_shape=(224, 224, 3), num_classes=1000, dropout=0.2, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    # width 1.0, depth 1.0
    out_channels = [16, 32, 48, 96, 112, 192]
    depthes = [1, 2, 2, 3, 5, 8]
    first_conv_filter = kwargs.pop("first_conv_filter", 32)
    output_conv_filter = kwargs.pop("output_conv_filter", 1280)
    return EfficientNetV2(**locals(), model_name="efficientnet_v2-b0", **kwargs)


def EfficientNetV2B1(input_shape=(240, 240, 3), num_classes=1000, dropout=0.2, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    # width 1.0, depth 1.1
    out_channels = [16, 32, 48, 96, 112, 192]
    depthes = [2, 3, 3, 4, 6, 9]
    first_conv_filter = kwargs.pop("first_conv_filter", 32)
    output_conv_filter = kwargs.pop("output_conv_filter", 1280)
    return EfficientNetV2(**locals(), model_name="efficientnet_v2-b1", **kwargs)


def EfficientNetV2B2(input_shape=(260, 260, 3), num_classes=1000, dropout=0.3, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    # width 1.1, depth 1.2
    out_channels = [16, 32, 56, 104, 120, 208]
    depthes = [2, 3, 3, 4, 6, 10]
    first_conv_filter = kwargs.pop("first_conv_filter", 32)
    output_conv_filter = kwargs.pop("output_conv_filter", 1408)
    return EfficientNetV2(**locals(), model_name="efficientnet_v2-b2", **kwargs)


def EfficientNetV2B3(input_shape=(300, 300, 3), num_classes=1000, dropout=0.3, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    # width 1.2, depth 1.4
    out_channels = [16, 40, 56, 112, 136, 232]
    depthes = [2, 3, 3, 5, 7, 12]
    first_conv_filter = kwargs.pop("first_conv_filter", 40)
    output_conv_filter = kwargs.pop("output_conv_filter", 1536)
    return EfficientNetV2(**locals(), model_name="efficientnet_v2-b3", **kwargs)


def EfficientNetV2T(input_shape=(288, 288, 3), num_classes=1000, dropout=0.2, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    # width 1.4 * 0.8, depth 1.8 * 0.9, from timm
    is_torch_mode = True
    out_channels = [24, 40, 48, 104, 128, 208]
    depthes = [2, 4, 4, 6, 9, 14]
    first_conv_filter = kwargs.pop("first_conv_filter", 24)
    output_conv_filter = kwargs.pop("output_conv_filter", 1024)
    return EfficientNetV2(**locals(), model_name="efficientnet_v2-t", **kwargs)


def EfficientNetV2T_GC(input_shape=(288, 288, 3), num_classes=1000, dropout=0.2, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    # width 1.4 * 0.8, depth 1.8 * 0.9, from timm
    is_torch_mode = True
    use_global_context_instead_of_se = True
    out_channels = [24, 40, 48, 104, 128, 208]
    depthes = [2, 4, 4, 6, 9, 14]
    first_conv_filter = kwargs.pop("first_conv_filter", 24)
    output_conv_filter = kwargs.pop("output_conv_filter", 1024)
    return EfficientNetV2(**locals(), model_name="efficientnet_v2-t-gc", **kwargs)


def EfficientNetV2S(input_shape=(384, 384, 3), num_classes=1000, dropout=0.2, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    # width 1.4, depth 1.8
    out_channels = [24, 48, 64, 128, 160, 256]
    depthes = [2, 4, 4, 6, 9, 15]
    first_conv_filter = kwargs.pop("first_conv_filter", 24)
    output_conv_filter = kwargs.pop("output_conv_filter", 1280)
    rescale_mode = kwargs.pop("rescale_mode", "tf")
    return EfficientNetV2(**locals(), model_name="efficientnet_v2-s", **kwargs)


def EfficientNetV2M(input_shape=(480, 480, 3), num_classes=1000, dropout=0.3, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    # width 1.6, depth 2.2
    out_channels = [24, 48, 80, 160, 176, 304, 512]
    depthes = [3, 5, 5, 7, 14, 18, 5]
    expands = [1, 4, 4, 4, 6, 6, 6]
    strides = [1, 2, 2, 2, 1, 2, 1]
    se_ratios = [0, 0, 0, 0.25, 0.25, 0.25, 0.25]
    first_conv_filter = kwargs.pop("first_conv_filter", 24)
    output_conv_filter = kwargs.pop("output_conv_filter", 1280)
    rescale_mode = kwargs.pop("rescale_mode", "tf")
    return EfficientNetV2(**locals(), model_name="efficientnet_v2-m", **kwargs)


def EfficientNetV2L(input_shape=(480, 480, 3), num_classes=1000, dropout=0.4, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    # width 1.6, depth 2.2
    out_channels = [32, 64, 96, 192, 224, 384, 640]
    depthes = [4, 7, 7, 10, 19, 25, 7]
    expands = [1, 4, 4, 4, 6, 6, 6]
    strides = [1, 2, 2, 2, 1, 2, 1]
    se_ratios = [0, 0, 0, 0.25, 0.25, 0.25, 0.25]
    first_conv_filter = kwargs.pop("first_conv_filter", 32)
    output_conv_filter = kwargs.pop("output_conv_filter", 1280)
    rescale_mode = kwargs.pop("rescale_mode", "tf")
    return EfficientNetV2(**locals(), model_name="efficientnet_v2-l", **kwargs)


def EfficientNetV2XL(input_shape=(512, 512, 3), num_classes=1000, dropout=0.4, classifier_activation="softmax", pretrained="imagenet21k-ft1k", **kwargs):
    out_channels = [32, 64, 96, 192, 256, 512, 640]
    depthes = [4, 8, 8, 16, 24, 32, 8]
    expands = [1, 4, 4, 4, 6, 6, 6]
    strides = [1, 2, 2, 2, 1, 2, 1]
    se_ratios = [0, 0, 0, 0.25, 0.25, 0.25, 0.25]
    first_conv_filter = kwargs.pop("first_conv_filter", 32)
    output_conv_filter = kwargs.pop("output_conv_filter", 1280)
    rescale_mode = kwargs.pop("rescale_mode", "tf")
    return EfficientNetV2(**locals(), model_name="efficientnet_v2-xl", **kwargs)
