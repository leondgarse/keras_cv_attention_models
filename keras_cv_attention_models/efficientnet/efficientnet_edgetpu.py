""" Creates an EfficientNet-EdgeTPU model
Ref impl: https://github.com/tensorflow/tpu/tree/master/models/official/efficientnet/edgetpu
"""
import math
from keras_cv_attention_models.efficientnet.efficientnet_v2 import EfficientNetV2
from keras_cv_attention_models.attention_layers import make_divisible
from keras_cv_attention_models.models import register_model


def get_expanded_width_depth(width, depth, fix_head_stem=False):
    out_channels = [ii * width for ii in [24, 32, 48, 96, 144, 192]]
    depthes = [int(math.ceil(ii * depth)) for ii in [1, 2, 4, 5, 4, 2]]
    first_conv_filter = 32 * width
    output_conv_filter = 1280 * width

    out_channels = [out_channels[0], out_channels[0], *out_channels[1:]]
    depthes = [1, depthes[0] - 1, *depthes[1:]]
    return out_channels, depthes, first_conv_filter, output_conv_filter


def EfficientNetEdgeTPU(
    expands=[-1, 4, 8, 8, 8, 8, 8],  # expands[0] = expands[1] * out_channels[0] / first_conv_filter, as timm using expand on out_channel
    out_channels=[24, 24, 32, 48, 96, 144, 192],
    depthes=[1, 0, 2, 4, 5, 4, 2],  # Add an additional block, as timm using expand on out_channel
    strides=[1, 1, 2, 2, 2, 1, 2],
    se_ratios=[0, 0, 0, 0, 0, 0, 0, 0],
    first_conv_filter=32,
    output_conv_filter=1280,
    kernel_sizes=[3, 3, 3, 3, 5, 5, 5],
    use_shortcuts=[False, False, True, True, True, True, True],
    is_fused=[True, True, True, True, False, False, False],
    is_torch_mode=True,
    drop_connect_rate=0.2,
    pretrained="imagenet",
    activation="relu",
    model_name="EfficientNetEdgeTPU",
    **kwargs,
):
    kwargs.pop("kwargs", None)
    expands[0] = make_divisible(out_channels[0], 8) * expands[1] / make_divisible(first_conv_filter, 8)
    return EfficientNetV2(**locals(), **kwargs)


@register_model
def EfficientNetEdgeTPUSmall(input_shape=(224, 224, 3), num_classes=1000, dropout=0.2, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    out_channels, depthes, first_conv_filter, output_conv_filter = get_expanded_width_depth(1.0, 1.0)
    return EfficientNetEdgeTPU(**locals(), model_name="efficientnet_edgetpu-small", **kwargs)


@register_model
def EfficientNetEdgeTPUMedium(input_shape=(240, 240, 3), num_classes=1000, dropout=0.2, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    out_channels, depthes, first_conv_filter, output_conv_filter = get_expanded_width_depth(1.0, 1.1)
    return EfficientNetEdgeTPU(**locals(), model_name="efficientnet_edgetpu-medium", **kwargs)


@register_model
def EfficientNetEdgeTPULarge(input_shape=(300, 300, 3), num_classes=1000, dropout=0.3, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    out_channels, depthes, first_conv_filter, output_conv_filter = get_expanded_width_depth(1.2, 1.4)
    return EfficientNetEdgeTPU(**locals(), model_name="efficientnet_edgetpu-large", **kwargs)
