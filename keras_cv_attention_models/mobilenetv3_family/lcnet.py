from keras_cv_attention_models.mobilenetv3_family.mobilenetv3 import MobileNetV3


def LCNet(
    num_blocks=[1, 2, 2, 1, 5, 2],
    out_channels=[32, 64, 128, 256, 256, 512],
    expands=1,
    kernel_sizes=[3, 3, 3, 3, 5, 5],
    strides=[1, 2, 2, 2, 1, 2],
    activations="hard_swish",
    disable_shortcut=True,
    use_blocks_output_activation=True,
    se_ratios=[0, 0, 0, 0, 0, 0.25],
    output_num_features=1280,
    use_additional_output_conv=False,
    model_name="lcnet",
    **kwargs,
):
    kwargs.pop("kwargs", None)
    return MobileNetV3(**locals(), **kwargs)


def LCNet050(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return LCNet(**locals(), width_ratio=0.5, model_name="lcnet_050", **kwargs)


def LCNet075(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return LCNet(**locals(), width_ratio=0.75, model_name="lcnet_075", **kwargs)


def LCNet100(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return LCNet(**locals(), model_name="lcnet_100", **kwargs)


def LCNet150(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    use_output_feature_bias = False
    return LCNet(**locals(), width_ratio=1.5, model_name="lcnet_150", **kwargs)


def LCNet200(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    use_output_feature_bias = False
    return LCNet(**locals(), width_ratio=2.0, model_name="lcnet_200", **kwargs)


def LCNet250(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    use_output_feature_bias = False
    return LCNet(**locals(), width_ratio=2.5, model_name="lcnet_250", **kwargs)
