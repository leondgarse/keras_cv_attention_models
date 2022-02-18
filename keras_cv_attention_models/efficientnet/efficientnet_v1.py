"""
Creates a EfficientNetV1 Model as defined in: EfficientNetV1: Self-training with Noisy Student improves ImageNet classification.
arXiv preprint arXiv:1911.04252.
"""
import tensorflow as tf
from keras_cv_attention_models.efficientnet.efficientnet_v2 import EfficientNetV2


def get_expanded_width_depth(width, depth, fix_head_stem=False):
    out_channels = [ii * width for ii in [16, 24, 40, 80, 112, 192, 320]]
    depthes = [int(tf.math.ceil(ii * depth)) for ii in [1, 2, 2, 3, 3, 4, 1]]
    if fix_head_stem:
        depthes[0], depthes[-1] = 1, 1
        first_conv_filter, output_conv_filter = 32, 1280
    else:
        first_conv_filter = 32 * width
        output_conv_filter = 1280 * width
    return out_channels, depthes, first_conv_filter, output_conv_filter


def EfficientNetV1(
    expands=[1, 6, 6, 6, 6, 6, 6],
    out_channels=[16, 24, 40, 80, 112, 192, 320],
    depthes=[1, 2, 2, 3, 3, 4, 1],
    strides=[1, 2, 2, 2, 1, 2, 1],
    se_ratios=[0.25, 0.25, 0.25, 0.25, 0.25, 0.25, 0.25],
    first_conv_filter=32,
    output_conv_filter=1280,
    kernel_sizes=[3, 3, 5, 3, 5, 5, 3],
    drop_connect_rate=0.2,
    pretrained="noisy_student",
    model_name="EfficientNetV1",
    **kwargs,
):
    kwargs.pop("kwargs", None)
    return EfficientNetV2(**locals(), **kwargs)


def EfficientNetV1B0(input_shape=(224, 224, 3), num_classes=1000, dropout=0.2, classifier_activation="softmax", pretrained="noisy_student", **kwargs):
    out_channels, depthes, first_conv_filter, output_conv_filter = get_expanded_width_depth(1.0, 1.0)
    first_conv_filter = kwargs.pop("first_conv_filter", first_conv_filter)
    output_conv_filter = kwargs.pop("output_conv_filter", output_conv_filter)
    return EfficientNetV1(**locals(), model_name="efficientnet_v1-b0", **kwargs)


def EfficientNetV1B1(input_shape=(240, 240, 3), num_classes=1000, dropout=0.2, classifier_activation="softmax", pretrained="noisy_student", **kwargs):
    out_channels, depthes, first_conv_filter, output_conv_filter = get_expanded_width_depth(1.0, 1.1)
    first_conv_filter = kwargs.pop("first_conv_filter", first_conv_filter)
    output_conv_filter = kwargs.pop("output_conv_filter", output_conv_filter)
    return EfficientNetV1(**locals(), model_name="efficientnet_v1-b1", **kwargs)


def EfficientNetV1B2(input_shape=(260, 260, 3), num_classes=1000, dropout=0.3, classifier_activation="softmax", pretrained="noisy_student", **kwargs):
    out_channels, depthes, first_conv_filter, output_conv_filter = get_expanded_width_depth(1.1, 1.2)
    first_conv_filter = kwargs.pop("first_conv_filter", first_conv_filter)
    output_conv_filter = kwargs.pop("output_conv_filter", output_conv_filter)
    return EfficientNetV1(**locals(), model_name="efficientnet_v1-b2", **kwargs)


def EfficientNetV1B3(input_shape=(300, 300, 3), num_classes=1000, dropout=0.3, classifier_activation="softmax", pretrained="noisy_student", **kwargs):
    out_channels, depthes, first_conv_filter, output_conv_filter = get_expanded_width_depth(1.2, 1.4)
    first_conv_filter = kwargs.pop("first_conv_filter", first_conv_filter)
    output_conv_filter = kwargs.pop("output_conv_filter", output_conv_filter)
    return EfficientNetV1(**locals(), model_name="efficientnet_v1-b3", **kwargs)


def EfficientNetV1B4(input_shape=(380, 380, 3), num_classes=1000, dropout=0.4, classifier_activation="softmax", pretrained="noisy_student", **kwargs):
    out_channels, depthes, first_conv_filter, output_conv_filter = get_expanded_width_depth(1.4, 1.8)
    first_conv_filter = kwargs.pop("first_conv_filter", first_conv_filter)
    output_conv_filter = kwargs.pop("output_conv_filter", output_conv_filter)
    return EfficientNetV1(**locals(), model_name="efficientnet_v1-b4", **kwargs)


def EfficientNetV1B5(input_shape=(456, 456, 3), num_classes=1000, dropout=0.4, classifier_activation="softmax", pretrained="noisy_student", **kwargs):
    out_channels, depthes, first_conv_filter, output_conv_filter = get_expanded_width_depth(1.6, 2.2)
    first_conv_filter = kwargs.pop("first_conv_filter", first_conv_filter)
    output_conv_filter = kwargs.pop("output_conv_filter", output_conv_filter)
    return EfficientNetV1(**locals(), model_name="efficientnet_v1-b5", **kwargs)


def EfficientNetV1B6(input_shape=(528, 528, 3), num_classes=1000, dropout=0.5, classifier_activation="softmax", pretrained="noisy_student", **kwargs):
    out_channels, depthes, first_conv_filter, output_conv_filter = get_expanded_width_depth(1.8, 2.6)
    first_conv_filter = kwargs.pop("first_conv_filter", first_conv_filter)
    output_conv_filter = kwargs.pop("output_conv_filter", output_conv_filter)
    return EfficientNetV1(**locals(), model_name="efficientnet_v1-b6", **kwargs)


def EfficientNetV1B7(input_shape=(600, 600, 3), num_classes=1000, dropout=0.5, classifier_activation="softmax", pretrained="noisy_student", **kwargs):
    out_channels, depthes, first_conv_filter, output_conv_filter = get_expanded_width_depth(2.0, 3.1)
    first_conv_filter = kwargs.pop("first_conv_filter", first_conv_filter)
    output_conv_filter = kwargs.pop("output_conv_filter", output_conv_filter)
    return EfficientNetV1(**locals(), model_name="efficientnet_v1-b7", **kwargs)


def EfficientNetV1L2(input_shape=(800, 800, 3), num_classes=1000, dropout=0.5, classifier_activation="softmax", pretrained="noisy_student", **kwargs):
    out_channels, depthes, first_conv_filter, output_conv_filter = get_expanded_width_depth(4.3, 5.3)
    first_conv_filter = kwargs.pop("first_conv_filter", first_conv_filter)
    output_conv_filter = kwargs.pop("output_conv_filter", output_conv_filter)
    return EfficientNetV1(**locals(), model_name="efficientnet_v1-l2", **kwargs)


# https://github.com/google/automl/tree/master/efficientdet/backbone/efficientnet_lite_builder.py
def EfficientNetV1Lite0(input_shape=(320, 320, 3), num_classes=1000, dropout=0.2, classifier_activation="softmax", pretrained=None, **kwargs):
    out_channels, depthes, first_conv_filter, output_conv_filter = get_expanded_width_depth(1.0, 1.0, fix_head_stem=True)
    first_conv_filter, output_conv_filter = kwargs.pop("first_conv_filter", first_conv_filter), kwargs.pop("output_conv_filter", output_conv_filter)
    return EfficientNetV1(**locals(), se_ratios=[0] * len(depthes), is_fused=False, model_name="efficientnet_v1-lite0", **kwargs)


def EfficientNetV1Lite1(input_shape=(384, 384, 3), num_classes=1000, dropout=0.2, classifier_activation="softmax", pretrained=None, **kwargs):
    out_channels, depthes, first_conv_filter, output_conv_filter = get_expanded_width_depth(1.0, 1.1, fix_head_stem=True)
    first_conv_filter, output_conv_filter = kwargs.pop("first_conv_filter", first_conv_filter), kwargs.pop("output_conv_filter", output_conv_filter)
    return EfficientNetV1(**locals(), se_ratios=[0] * len(depthes), is_fused=False, model_name="efficientnet_v1-lite1", **kwargs)


def EfficientNetV1Lite2(input_shape=(448, 448, 3), num_classes=1000, dropout=0.3, classifier_activation="softmax", pretrained=None, **kwargs):
    out_channels, depthes, first_conv_filter, output_conv_filter = get_expanded_width_depth(1.1, 1.2, fix_head_stem=True)
    first_conv_filter, output_conv_filter = kwargs.pop("first_conv_filter", first_conv_filter), kwargs.pop("output_conv_filter", output_conv_filter)
    return EfficientNetV1(**locals(), se_ratios=[0] * len(depthes), is_fused=False, model_name="efficientnet_v1-lite2", **kwargs)


def EfficientNetV1Lite3(input_shape=(512, 512, 3), num_classes=1000, dropout=0.3, classifier_activation="softmax", pretrained=None, **kwargs):
    out_channels, depthes, first_conv_filter, output_conv_filter = get_expanded_width_depth(1.2, 1.4, fix_head_stem=True)
    first_conv_filter, output_conv_filter = kwargs.pop("first_conv_filter", first_conv_filter), kwargs.pop("output_conv_filter", output_conv_filter)
    return EfficientNetV1(**locals(), se_ratios=[0] * len(depthes), is_fused=False, model_name="efficientnet_v1-lite3", **kwargs)


def EfficientNetV1Lite4(input_shape=(640, 640, 3), num_classes=1000, dropout=0.4, classifier_activation="softmax", pretrained=None, **kwargs):
    out_channels, depthes, first_conv_filter, output_conv_filter = get_expanded_width_depth(1.4, 1.8, fix_head_stem=True)
    first_conv_filter, output_conv_filter = kwargs.pop("first_conv_filter", first_conv_filter), kwargs.pop("output_conv_filter", output_conv_filter)
    return EfficientNetV1(**locals(), se_ratios=[0] * len(depthes), is_fused=False, model_name="efficientnet_v1-lite4", **kwargs)
