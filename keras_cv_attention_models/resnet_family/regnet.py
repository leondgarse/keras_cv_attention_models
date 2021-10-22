from keras_cv_attention_models.aotnet import AotNet
from keras_cv_attention_models.download_and_load import reload_model_weights

PRETRAINED_DICT = {
    "regnetz_b": {"imagenet": "9f03b9e885c6918db54374b74f355eb9"},
    "regnetz_c": {"imagenet": "cfa0cf70aa724d8e79633641208d7d39"},
    "regnetz_d": {"imagenet": "b76e7c68a309a3697ffb5f30c0aeeea6"},
}

def RegNetZB(input_shape=(224, 224, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 6, 12, 2]
    strides = [2, 2, 2, 2]
    out_channels = [48, 96, 192, 288]
    # timm bottle_in=True mode, the first ratio in each stack is `ratio * previous_channel`
    hidden_channel_ratio = [[2, 3], [1.5] + [3] * 5, [1.5] + [3] * 11, [2, 3]]
    use_block_output_activation = False # timm linear_out=True mode
    stem_type = "kernel_3x3"
    stem_width = 32
    stem_downsample = False
    se_ratio = 0.25
    attn_params = {"activation": "relu"}
    group_size = 16
    shortcut_type = None
    output_num_features = 1536
    model = AotNet(**locals(), model_name="regnetz_b", **kwargs)
    reload_model_weights(model, pretrained_dict=PRETRAINED_DICT, sub_release="regnet", input_shape=input_shape, pretrained=pretrained)
    return model

def RegNetZC(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 6, 12, 2]
    strides = [2, 2, 2, 2]
    out_channels = [48, 96, 192, 288]
    # timm bottle_in=True mode, the first ratio in each stack is `ratio * previous_channel`
    hidden_channel_ratio = [[32 * 4 / 48, 4], [2] + [4] * 5, [2] + [4] * 11, [192 * 4 / 288, 4]]
    use_block_output_activation = False # timm linear_out=True mode
    stem_type = "kernel_3x3"
    stem_width = 32
    stem_downsample = False
    se_ratio = 0.25
    attn_params = {"activation": "relu"}
    group_size = 16
    shortcut_type = None
    output_num_features = 1536
    model = AotNet(**locals(), model_name="regnetz_c", **kwargs)
    reload_model_weights(model, pretrained_dict=PRETRAINED_DICT, sub_release="regnet", input_shape=input_shape, pretrained=pretrained)
    return model

def RegNetZD(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 6, 12, 3]
    strides = [1, 2, 2, 2]
    out_channels = [64, 128, 256, 384]
    # timm bottle_in=True mode, the first ratio in each stack is `ratio * previous_channel`
    hidden_channel_ratio = [[64 * 4 / 64, 4, 4], [2] + [4] * 5, [2] + [4] * 11, [256 * 4 / 384, 4, 4]]
    use_block_output_activation = False # timm linear_out=True mode
    stem_type = "tiered"
    stem_last_strides = 2
    stem_width = 64
    stem_downsample = False
    se_ratio = 0.25
    attn_params = {"activation": "relu"}
    group_size = 32
    shortcut_type = None
    output_num_features = 1792
    model = AotNet(**locals(), model_name="regnetz_d", **kwargs)
    reload_model_weights(model, pretrained_dict=PRETRAINED_DICT, sub_release="regnet", input_shape=input_shape, pretrained=pretrained)
    return model
