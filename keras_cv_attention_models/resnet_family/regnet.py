from keras_cv_attention_models.aotnet import AotNet
from keras_cv_attention_models.download_and_load import reload_model_weights

PRETRAINED_DICT = {
    "regnety_040": {"imagenet": "35bde7c2f9e391c20d4aa20160902961"},
    "regnety_064": {"imagenet": "1b096c91cdcf5b3ea82900468cda829b"},
    "regnety_080": {"imagenet": "651b4e500556593695258dee604eecba"},
    "regnety_160": {"imagenet": "7a039bfb1e006571991a19b5e3fbd0d1"},
    "regnety_320": {"imagenet": "20fbbf6de39635b89ae284ac0574c4cf"},
    "regnetz_b16": {"imagenet": "9f03b9e885c6918db54374b74f355eb9"},
    "regnetz_c16": {"imagenet": "cfa0cf70aa724d8e79633641208d7d39"},
    "regnetz_c16_evo": {"imagenet": "cf04ad48e51d652d901b03e195dace21"},
    "regnetz_d32": {"imagenet": "b76e7c68a309a3697ffb5f30c0aeeea6"},
    "regnetz_d8": {"imagenet": "1fa2de5606f33b503cedda4689b8bddd"},
    "regnetz_d8_evo": {"imagenet": "97b8bcf98755868765b3bc62700bbb39"},
    "regnetz_e8": {"imagenet": "7d0a85b287ad041e0446623c7070fa8f"},
}


def RegNetY(num_blocks, out_channels, input_shape=(224, 224, 3), hidden_channel_ratio=1, stem_width=32, se_ratio=0.25, pretrained="imagenet", **kwargs):
    strides = [2, 2, 2, 2]
    stem_type = "kernel_3x3"
    stem_downsample = False
    # se_ratio using input_channel
    # se_ratio = [[se_ratio * (stem_width if id == 0 else out_channels[id - 1]) / out_channels[id]] + [se_ratio] * bb for id, bb in enumerate(num_blocks)]
    se_ratio = [
        [se_ratio * stem_width / out_channels[0]] + [se_ratio] * num_blocks[0],
        [se_ratio * out_channels[0] / out_channels[1]] + [se_ratio] * num_blocks[1],
        [se_ratio * out_channels[1] / out_channels[2]] + [se_ratio] * num_blocks[2],
        [se_ratio * out_channels[2] / out_channels[3]] + [se_ratio] * num_blocks[3],
    ]
    attn_params = {"se_divisor": 1}
    kwargs.pop("kwargs", None)
    model = AotNet(**locals(), **kwargs)
    reload_model_weights(model, pretrained_dict=PRETRAINED_DICT, sub_release="resnet_family", pretrained=pretrained)
    return model


def RegNetY032(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 5, 13, 1]
    out_channels = [72, 216, 576, 1512]
    group_size = 24
    model = RegNetY(**locals(), model_name="regnety_032", **kwargs)
    return model


def RegNetY040(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 6, 12, 2]
    out_channels = [128, 192, 512, 1088]
    group_size = 64
    model = RegNetY(**locals(), model_name="regnety_040", **kwargs)
    return model


def RegNetY064(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 7, 14, 2]
    out_channels = [144, 288, 576, 1296]
    group_size = 72
    model = RegNetY(**locals(), model_name="regnety_064", **kwargs)
    return model


def RegNetY080(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 4, 10, 1]
    out_channels = [168, 448, 896, 2016]
    group_size = 56
    model = RegNetY(**locals(), model_name="regnety_080", **kwargs)
    return model


def RegNetY160(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 4, 11, 1]
    out_channels = [224, 448, 1232, 3024]
    group_size = 112
    model = RegNetY(**locals(), model_name="regnety_160", **kwargs)
    return model


def RegNetY320(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 5, 12, 1]
    out_channels = [232, 696, 1392, 3712]
    group_size = 232
    model = RegNetY(**locals(), model_name="regnety_320", **kwargs)
    return model


def RegNetZB16(input_shape=(224, 224, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 6, 12, 2]
    strides = [2, 2, 2, 2]
    out_channels = [48, 96, 192, 288]
    # timm bottle_in=True mode, the first ratio in each stack is `ratio * previous_channel`
    hidden_channel_ratio = [[32 * 3 / 48, 3], [1.5] + [3] * 5, [1.5] + [3] * 11, [192 * 3 / 288, 3]]
    use_block_output_activation = False  # timm linear_out=True mode
    stem_type = "kernel_3x3"
    stem_width = 32
    stem_downsample = False
    se_ratio = 0.25
    attn_params = {"activation": "relu"}
    group_size = 16
    shortcut_type = None
    output_num_features = 1536
    kwargs.pop("kwargs", None)
    model = AotNet(**locals(), model_name=kwargs.pop("model_name", "regnetz_b16"), **kwargs)
    reload_model_weights(model, pretrained_dict=PRETRAINED_DICT, sub_release="resnet_family", pretrained=pretrained)
    return model


def RegNetZC16(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 6, 12, 2]
    strides = [2, 2, 2, 2]
    out_channels = [48, 96, 192, 288]
    # timm bottle_in=True mode, the first ratio in each stack is `ratio * previous_channel`
    hidden_channel_ratio = [[32 * 4 / 48, 4], [2] + [4] * 5, [2] + [4] * 11, [192 * 4 / 288, 4]]
    use_block_output_activation = False  # timm linear_out=True mode
    stem_type = "kernel_3x3"
    stem_width = 32
    stem_downsample = False
    se_ratio = 0.25
    attn_params = {"activation": "relu"}
    group_size = 16
    shortcut_type = None
    output_num_features = 1536
    kwargs.pop("kwargs", None)
    model = AotNet(**locals(), model_name=kwargs.pop("model_name", "regnetz_c16"), **kwargs)
    reload_model_weights(model, pretrained_dict=PRETRAINED_DICT, sub_release="resnet_family", pretrained=pretrained)
    return model


def RegNetZC16_EVO(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    use_evo_norm = True
    evo_norm_group_size = 16
    bn_epsilon = 1e-3
    return RegNetZC16(**locals(), model_name="regnetz_c16_evo", **kwargs)


def RegNetZD32(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 6, 12, 3]
    strides = [1, 2, 2, 2]
    out_channels = [64, 128, 256, 384]
    # timm bottle_in=True mode, the first ratio in each stack is `ratio * previous_channel`
    hidden_channel_ratio = [[64 * 4 / 64, 4, 4], [2] + [4] * 5, [2] + [4] * 11, [256 * 4 / 384, 4, 4]]
    use_block_output_activation = False  # timm linear_out=True mode
    stem_type = "tiered"
    stem_last_strides = 2
    stem_width = 64
    stem_downsample = False
    se_ratio = 0.25
    attn_params = {"activation": "relu"}
    group_size = 32
    shortcut_type = None
    output_num_features = 1792
    kwargs.pop("kwargs", None)
    model = AotNet(**locals(), model_name=kwargs.pop("model_name", "regnetz_d32"), **kwargs)
    reload_model_weights(model, pretrained_dict=PRETRAINED_DICT, sub_release="resnet_family", pretrained=pretrained)
    return model


def RegNetZD8(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 6, 12, 3]
    strides = [1, 2, 2, 2]
    out_channels = [64, 128, 256, 384]
    # timm bottle_in=True mode, the first ratio in each stack is `ratio * previous_channel`
    hidden_channel_ratio = [[64 * 4 / 64, 4, 4], [64 * 4 / 128] + [4] * 5, [128 * 4 / 256] + [4] * 11, [256 * 4 / 384, 4, 4]]
    use_block_output_activation = False  # timm linear_out=True mode
    stem_type = kwargs.pop("stem_type", "tiered")  # support RegNetZD8_EVO
    stem_last_strides = 2
    stem_width = 64
    stem_downsample = False
    se_ratio = 0.25
    attn_params = {"activation": "relu"}
    group_size = 8
    shortcut_type = None
    output_num_features = 1792
    kwargs.pop("kwargs", None)
    model = AotNet(**locals(), model_name=kwargs.pop("model_name", "regnetz_d8"), **kwargs)
    reload_model_weights(model, pretrained_dict=PRETRAINED_DICT, sub_release="resnet_family", pretrained=pretrained)
    return model


def RegNetZD8_EVO(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    stem_type = "deep2"
    use_evo_norm = True
    evo_norm_group_size = 16
    bn_epsilon = 1e-3
    return RegNetZD8(**locals(), model_name="regnetz_d8_evo", **kwargs)


def RegNetZE8(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 8, 16, 3]
    strides = [1, 2, 2, 2]
    out_channels = [96, 192, 384, 512]
    # timm bottle_in=True mode, the first ratio in each stack is `ratio * previous_channel`
    hidden_channel_ratio = [[64 * 4 / 96, 4, 4], [96 * 4 / 192] + [4] * 7, [192 * 4 / 384] + [4] * 15, [384 * 4 / 512, 4, 4]]
    use_block_output_activation = False  # timm linear_out=True mode
    stem_type = "tiered"
    stem_last_strides = 2
    stem_width = 64
    stem_downsample = False
    se_ratio = 0.25
    attn_params = {"activation": "relu"}
    group_size = 8
    shortcut_type = None
    output_num_features = 2048
    kwargs.pop("kwargs", None)
    model = AotNet(**locals(), model_name=kwargs.pop("model_name", "regnetz_e8"), **kwargs)
    reload_model_weights(model, pretrained_dict=PRETRAINED_DICT, sub_release="resnet_family", pretrained=pretrained)
    return model
