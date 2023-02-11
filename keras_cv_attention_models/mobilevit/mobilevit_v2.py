from keras_cv_attention_models.mobilevit.mobilevit import MobileViT


def MobileViT_V2(
    num_blocks=[1, 2, 3, 5, 4],
    out_channels=[64, 128, 256, 384, 512],
    attn_channels=0.5,  # Can be a list matching out_channels, or a float number for expansion ratio of out_channels
    expand_ratio=2,
    stem_width=32,
    resize_first=True,  # False for V1, True for V2
    use_depthwise=True,  # False for V1, True for V2
    use_fusion=False,  # True for V1, False for V2
    num_norm_groups=1,  # -1 or 0 for V1 using layer_norm, or 1 for V2 using group_norm
    use_linear_attention=True,  # False for V1, True for V2
    output_num_features=0,
    model_name="mobilevit_v2",
    **kwargs,
):
    kwargs.pop("kwargs", None)
    return MobileViT(**locals(), **kwargs)


def get_mobilevit_v2_width(multiplier=1.0):
    return int(32 * multiplier), [int(ii * multiplier) for ii in [64, 128, 256, 384, 512]]  # stem_width, out_channels


def MobileViT_V2_050(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    stem_width, out_channels = get_mobilevit_v2_width(0.5)
    return MobileViT_V2(**locals(), model_name="mobilevit_v2_050", **kwargs)


def MobileViT_V2_075(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    stem_width, out_channels = get_mobilevit_v2_width(0.75)
    return MobileViT_V2(**locals(), model_name="mobilevit_v2_075", **kwargs)


def MobileViT_V2_100(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return MobileViT_V2(**locals(), model_name="mobilevit_v2_100", **kwargs)


def MobileViT_V2_125(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    stem_width, out_channels = get_mobilevit_v2_width(1.25)
    return MobileViT_V2(**locals(), model_name="mobilevit_v2_125", **kwargs)


def MobileViT_V2_150(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    stem_width, out_channels = get_mobilevit_v2_width(1.5)
    return MobileViT_V2(**locals(), model_name="mobilevit_v2_150", **kwargs)


def MobileViT_V2_175(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    stem_width, out_channels = get_mobilevit_v2_width(1.75)
    return MobileViT_V2(**locals(), model_name="mobilevit_v2_175", **kwargs)


def MobileViT_V2_200(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    stem_width, out_channels = get_mobilevit_v2_width(2.0)
    return MobileViT_V2(**locals(), model_name="mobilevit_v2_200", **kwargs)
