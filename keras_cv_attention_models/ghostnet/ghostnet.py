from keras_cv_attention_models.ghostnet.ghostnet_v2 import GhostNetV2


def GhostNet(
    kernel_sizes=[3, 3, 3, 5, 5, 3, 3, 3, 3, 3, 3, 5, 5, 5, 5, 5],
    first_ghost_channels=[16, 48, 72, 72, 120, 240, 200, 184, 184, 480, 672, 672, 960, 960, 960, 960],
    out_channels=[16, 24, 24, 40, 40, 80, 80, 80, 80, 112, 112, 160, 160, 160, 160, 160],
    se_ratios=[0, 0, 0, 0.25, 0.25, 0, 0, 0, 0, 0.25, 0.25, 0.25, 0, 0.25, 0, 0.25],
    strides=[1, 2, 1, 2, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 1, 1],
    stem_width=16,
    stem_strides=2,
    width_mul=1.0,
    num_ghost_module_v1_stacks=-1,  # num of `ghost_module` stcks on the head, others are `ghost_module_multiply`, set `-1` for all using `ghost_module`
    output_conv_filter=-1,  # -1 for first_ghost_channels[-1] * width_mul
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="relu",
    classifier_activation="softmax",
    dropout=0,
    pretrained=None,
    model_name="ghostnet",
    kwargs=None,
):
    return GhostNetV2(**locals())


def GhostNet_050(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return GhostNet(**locals(), width_mul=0.5, model_name="ghostnet_050", **kwargs)


def GhostNet_100(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return GhostNet(**locals(), model_name="ghostnet_100", **kwargs)


def GhostNet_130(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return GhostNet(**locals(), width_mul=1.3, model_name="ghostnet_130", **kwargs)
