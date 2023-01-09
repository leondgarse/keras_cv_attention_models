from keras_cv_attention_models.convnext.convnext import ConvNeXt


def ConvNeXtV2(
    num_blocks=[3, 3, 9, 3],
    out_channels=[96, 192, 384, 768],
    stem_width=-1,
    layer_scale_init_value=0,  # 1e-6 for v1, 0 for v2
    use_grn=True,  # False for v1, True for v2
    head_init_scale=1.0,
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


def ConvNeXtV2Atto(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 2, 6, 2]
    out_channels = [40, 80, 160, 320]
    return ConvNeXtV2(**locals(), model_name="convnext_v2_atto", **kwargs)


def ConvNeXtV2Femto(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 2, 6, 2]
    out_channels = [48, 96, 192, 384]
    return ConvNeXtV2(**locals(), model_name="convnext_v2_femto", **kwargs)


def ConvNeXtV2Pico(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 2, 6, 2]
    out_channels = [64, 128, 256, 512]
    return ConvNeXtV2(**locals(), model_name="convnext_v2_pico", **kwargs)


def ConvNeXtV2Nano(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 2, 8, 2]
    out_channels = [80, 160, 320, 640]
    return ConvNeXtV2(**locals(), model_name="convnext_v2_nano", **kwargs)


def ConvNeXtV2Tiny(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 3, 9, 3]
    out_channels = [96, 192, 384, 768]
    return ConvNeXtV2(**locals(), model_name="convnext_v2_tiny", **kwargs)


def ConvNeXtV2Base(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 3, 27, 3]
    out_channels = [128, 256, 512, 1024]
    return ConvNeXtV2(**locals(), model_name="convnext_v2_base", **kwargs)


def ConvNeXtV2Large(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 3, 27, 3]
    out_channels = [192, 384, 768, 1536]
    return ConvNeXtV2(**locals(), model_name="convnext_v2_large", **kwargs)


def ConvNeXtV2Huge(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 3, 27, 3]
    out_channels = [352, 704, 1408, 2816]
    return ConvNeXtV2(**locals(), model_name="convnext_v2_huge", **kwargs)
