from keras_cv_attention_models.mobilenetv3_family.mobilenetv3 import MobileNetV3


def FBNetV3(
    num_blocks=[2, 4, 1, 4, 1, 4, 1, 5, 1, 5, 1],
    out_channels=[16, 24, 40, 40, 72, 72, 120, 120, 184, 184, 224],
    expands=[1, [4, 2, 2, 2], 5, 3, 5, 3, 5, 3, 6, 4, 6],
    kernel_sizes=[3, 5, 5, 5, 5, 3, 3, 5, 3, 5, 5],
    strides=[1, 2, 2, 1, 2, 1, 1, 1, 2, 1, 1],
    activations="hard_swish",
    se_ratios=[0, 0, 0.25, 0.25, 0, 0, 0.25, 0.25, 0.25, 0.25, 0.25],
    se_activation=("hard_swish", "hard_sigmoid_torch"),
    se_limit_round_down=0.95,
    use_expanded_se_ratio=False,
    output_num_features=1984,
    use_output_feature_bias=False,
    model_name="fbnetv3",
    **kwargs,
):
    kwargs.pop("kwargs", None)
    return MobileNetV3(**locals(), **kwargs)


def FBNetV3B(input_shape=(256, 256, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return FBNetV3(**locals(), model_name="fbnetv3_b", **kwargs)


def FBNetV3D(input_shape=(256, 256, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 6, 1, 4, 1, 4, 1, 6, 1, 5, 1]
    out_channels = [16, 24, 40, 40, 72, 72, 128, 128, 208, 208, 240]
    expands = [1, [5, 2, 2, 2, 2, 2], 4, 3, 5, 3, 5, 3, 6, 5, 6]
    kernel_sizes = [3, 3, 5, 3, 3, 3, 3, 5, 3, 5, 5]
    stem_width = 24
    return FBNetV3(**locals(), model_name="fbnetv3_d", **kwargs)


def FBNetV3G(input_shape=(256, 256, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 5, 1, 4, 1, 4, 1, 8, 1, 6, 2]
    out_channels = [24, 40, 56, 56, 104, 104, 160, 160, 264, 264, 288]
    expands = [1, [4, 2, 2, 2, 2], 4, 3, 5, 3, 5, 3, 6, 5, 6]
    kernel_sizes = [3, 5, 5, 5, 5, 3, 3, 5, 3, 5, 5]
    stem_width = 32
    return FBNetV3(**locals(), model_name="fbnetv3_g", **kwargs)
