from keras_cv_attention_models.mobilenetv3_family.mobilenetv3 import MobileNetV3


def get_expanded_width_depth(width, depth):
    out_channels = [ii * width for ii in [16, 24, 40, 80, 112, 192, 320]]
    num_blocks = [int(round(ii * depth)) for ii in [1, 2, 2, 3, 3, 4, 1]]
    return out_channels, num_blocks


def TinyNet(
    num_blocks=[1, 2, 2, 3, 3, 4, 1],
    out_channels=[16, 24, 40, 80, 112, 192, 320],
    expands=[1, 6, 6, 6, 6, 6, 6],
    kernel_sizes=[3, 3, 5, 3, 5, 5, 3],
    strides=[1, 2, 2, 2, 1, 2, 1],
    activations="swish",
    stem_width=32,
    fix_stem=True,
    se_ratios=0.25,
    se_activation=None,  # None for same with activations
    use_expanded_se_ratio=False,
    se_divisor=1,
    output_num_features=1280,
    use_additional_output_conv=False,
    use_output_feature_bias=False,
    use_avg_pool_conv_output=False,
    model_name="tinynet",
    **kwargs,
):
    stem_feature_activation = activations
    kwargs.pop("kwargs", None)
    return MobileNetV3(**locals(), **kwargs)


def TinyNetA(input_shape=(192, 192, 3), num_classes=1000, activations="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    out_channels, num_blocks = get_expanded_width_depth(1.0, 1.2)
    return TinyNet(**locals(), model_name="tinynet_a", **kwargs)


def TinyNetB(input_shape=(188, 188, 3), num_classes=1000, activations="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    out_channels, num_blocks = get_expanded_width_depth(0.75, 1.1)
    return TinyNet(**locals(), model_name="tinynet_b", **kwargs)


def TinyNetC(input_shape=(184, 184, 3), num_classes=1000, activations="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    out_channels, num_blocks = get_expanded_width_depth(0.54, 0.85)
    return TinyNet(**locals(), model_name="tinynet_c", **kwargs)


def TinyNetD(input_shape=(152, 152, 3), num_classes=1000, activations="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    out_channels, num_blocks = get_expanded_width_depth(0.54, 0.695)
    return TinyNet(**locals(), model_name="tinynet_d", **kwargs)


def TinyNetE(input_shape=(106, 106, 3), num_classes=1000, activations="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    out_channels, num_blocks = get_expanded_width_depth(0.51, 0.6)
    return TinyNet(**locals(), model_name="tinynet_e", **kwargs)
