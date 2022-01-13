"""
Creates a EfficientNetV1 Model as defined in: EfficientNetV1: Self-training with Noisy Student improves ImageNet classification.
arXiv preprint arXiv:1911.04252.
"""
import math
import copy
from keras_cv_attention_models.efficientnet.efficientnet_v2 import EfficientNetV2

BLOCK_CONFIGS = {
    "base": {  # width 1.0, depth 1.0
        "first_conv_filter": 32,
        "output_conv_filter": 1280,
        "expands": [1, 6, 6, 6, 6, 6, 6],
        "out_channels": [16, 24, 40, 80, 112, 192, 320],
        "depthes": [1, 2, 2, 3, 3, 4, 1],
        "strides": [1, 2, 2, 2, 1, 2, 1],
        "use_ses": [1, 1, 1, 1, 1, 1, 1],
        "kernel_sizes": [3, 3, 5, 3, 5, 5, 3],
        "rescale_mode": "torch",
    },
    "b0": {"width": 1.0, "depth": 1.0},
    "b1": {"width": 1.0, "depth": 1.1},
    "b2": {"width": 1.1, "depth": 1.2},
    "b3": {"width": 1.2, "depth": 1.4},
    "b4": {"width": 1.4, "depth": 1.8},
    "b5": {"width": 1.6, "depth": 2.2},
    "b6": {"width": 1.8, "depth": 2.6},
    "b7": {"width": 2.0, "depth": 3.1},
    "l2": {"width": 4.3, "depth": 5.3},
}


def EfficientNetV1(
    model_type,
    input_shape=(None, None, 3),
    num_classes=1000,
    dropout=0.2,
    first_strides=2,
    drop_connect_rate=0.2,
    classifier_activation="softmax",
    include_preprocessing=False,
    pretrained="noisy_student",
    model_name="EfficientNetV1",
    kwargs=None,  # Not used, just recieving parameter
):
    blocks_config = copy.deepcopy(BLOCK_CONFIGS["base"])
    exp_config = BLOCK_CONFIGS.get(model_type.lower(), BLOCK_CONFIGS["b0"])
    blocks_config.update({"first_conv_filter": exp_config["width"] * blocks_config["first_conv_filter"]})
    blocks_config.update({"output_conv_filter": exp_config["width"] * blocks_config["output_conv_filter"]})
    blocks_config.update({"out_channels": [ii * exp_config["width"] for ii in blocks_config["out_channels"]]})
    blocks_config.update({"depthes": [int(math.ceil(ii * exp_config["depth"])) for ii in blocks_config["depthes"]]})

    return EfficientNetV2(
        model_type={"v1-" + model_type: blocks_config},
        input_shape=input_shape,
        num_classes=num_classes,
        dropout=dropout,
        first_strides=first_strides,
        drop_connect_rate=drop_connect_rate,
        classifier_activation=classifier_activation,
        include_preprocessing=include_preprocessing,
        pretrained=pretrained,
        model_name=model_name,
    )


def EfficientNetV1B0(input_shape=(224, 224, 3), num_classes=1000, dropout=0.2, classifier_activation="softmax", pretrained="noisy_student", **kwargs):
    return EfficientNetV1(model_type="b0", model_name="EfficientNetV1B0", **locals(), **kwargs)


def EfficientNetV1B1(input_shape=(240, 240, 3), num_classes=1000, dropout=0.2, classifier_activation="softmax", pretrained="noisy_student", **kwargs):
    return EfficientNetV1(model_type="b1", model_name="EfficientNetV1B1", **locals(), **kwargs)


def EfficientNetV1B2(input_shape=(260, 260, 3), num_classes=1000, dropout=0.3, classifier_activation="softmax", pretrained="noisy_student", **kwargs):
    return EfficientNetV1(model_type="b2", model_name="EfficientNetV1B2", **locals(), **kwargs)


def EfficientNetV1B3(input_shape=(300, 300, 3), num_classes=1000, dropout=0.3, classifier_activation="softmax", pretrained="noisy_student", **kwargs):
    return EfficientNetV1(model_type="b3", model_name="EfficientNetV1B3", **locals(), **kwargs)


def EfficientNetV1B4(input_shape=(380, 380, 3), num_classes=1000, dropout=0.4, classifier_activation="softmax", pretrained="noisy_student", **kwargs):
    return EfficientNetV1(model_type="b4", model_name="EfficientNetV1B4", **locals(), **kwargs)


def EfficientNetV1B5(input_shape=(456, 456, 3), num_classes=1000, dropout=0.4, classifier_activation="softmax", pretrained="noisy_student", **kwargs):
    return EfficientNetV1(model_type="b5", model_name="EfficientNetV1B5", **locals(), **kwargs)


def EfficientNetV1B6(input_shape=(528, 528, 3), num_classes=1000, dropout=0.5, classifier_activation="softmax", pretrained="noisy_student", **kwargs):
    return EfficientNetV1(model_type="b6", model_name="EfficientNetV1B6", **locals(), **kwargs)


def EfficientNetV1B7(input_shape=(600, 600, 3), num_classes=1000, dropout=0.5, classifier_activation="softmax", pretrained="noisy_student", **kwargs):
    return EfficientNetV1(model_type="b7", model_name="EfficientNetV1B7", **locals(), **kwargs)


def EfficientNetV1L2(input_shape=(800, 800, 3), num_classes=1000, dropout=0.5, classifier_activation="softmax", pretrained="noisy_student", **kwargs):
    return EfficientNetV1(model_type="l2", model_name="EfficientNetV1L2", **locals(), **kwargs)
