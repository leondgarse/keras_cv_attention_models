from keras_cv_attention_models.convnext.convnext import ConvNeXt
from keras_cv_attention_models.models import register_model


def ConvNeXtV2(
    num_blocks=[3, 3, 9, 3],
    out_channels=[96, 192, 384, 768],
    stem_width=-1,
    layer_scale_init_value=0,  # 1e-6 for v1, 0 for v2
    use_grn=True,  # False for v1, True for v2
    head_init_scale=1.0,
    layer_norm_epsilon=1e-6,  # 1e-5 for ConvNeXtXXlarge, 1e-6 for others
    output_num_filters=-1,  # If apply additional dense + activation before output dense, <0 for not using
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="gelu",
    drop_connect_rate=0.1,
    classifier_activation="softmax",
    dropout=0,
    pretrained=None,
    model_name="convnext_v2",
    kwargs=None,
):
    return ConvNeXt(**locals())


@register_model
def ConvNeXtV2Atto(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 2, 6, 2]
    out_channels = [40, 80, 160, 320]
    return ConvNeXtV2(**locals(), model_name="convnext_v2_atto", **kwargs)


@register_model
def ConvNeXtV2Femto(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 2, 6, 2]
    out_channels = [48, 96, 192, 384]
    return ConvNeXtV2(**locals(), model_name="convnext_v2_femto", **kwargs)


@register_model
def ConvNeXtV2Pico(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 2, 6, 2]
    out_channels = [64, 128, 256, 512]
    return ConvNeXtV2(**locals(), model_name="convnext_v2_pico", **kwargs)


@register_model
def ConvNeXtV2Nano(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 2, 8, 2]
    out_channels = [80, 160, 320, 640]
    return ConvNeXtV2(**locals(), model_name="convnext_v2_nano", **kwargs)


@register_model
def ConvNeXtV2Tiny(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 3, 9, 3]
    out_channels = [96, 192, 384, 768]
    return ConvNeXtV2(**locals(), model_name="convnext_v2_tiny", **kwargs)


@register_model
def ConvNeXtV2Base(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 3, 27, 3]
    out_channels = [128, 256, 512, 1024]
    return ConvNeXtV2(**locals(), model_name="convnext_v2_base", **kwargs)


@register_model
def ConvNeXtV2Large(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 3, 27, 3]
    out_channels = [192, 384, 768, 1536]
    return ConvNeXtV2(**locals(), model_name="convnext_v2_large", **kwargs)


@register_model
def ConvNeXtV2Huge(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 3, 27, 3]
    out_channels = [352, 704, 1408, 2816]
    return ConvNeXtV2(**locals(), model_name="convnext_v2_huge", **kwargs)
