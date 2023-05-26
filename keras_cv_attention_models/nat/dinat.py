from keras_cv_attention_models.nat.nat import NAT
from keras_cv_attention_models.models import register_model


@register_model
def DiNAT_Mini(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 6, 5]
    # dilation_rates = [[1, 8, 1], [1, 4, 1, 4], [1, 2] * 3, [1, 1, 1, 1, 1]]
    use_every_other_dilations = True
    return NAT(**locals(), model_name="dinat_mini", **kwargs)


@register_model
def DiNAT_Tiny(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 18, 5]
    # dilation_rates = [[1, 8, 1], [1, 4, 1, 4], [1, 2] * 9, [1, 1, 1, 1, 1]]
    use_every_other_dilations = True
    return NAT(**locals(), model_name="dinat_tiny", **kwargs)


@register_model
def DiNAT_Small(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 18, 5]
    num_heads = [3, 6, 12, 24]
    out_channels = [96, 192, 384, 768]
    mlp_ratio = kwargs.pop("mlp_ratio", 2)
    # layer_scale = kwargs.pop("layer_scale", 1e-5)
    # dilation_rates = [[1, 8, 1], [1, 4, 1, 4], [1, 2] * 9, [1, 1, 1, 1, 1]]
    use_every_other_dilations = True
    return NAT(**locals(), model_name="dinat_small", **kwargs)


@register_model
def DiNAT_Base(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 18, 5]
    num_heads = [4, 8, 16, 32]
    out_channels = [128, 256, 512, 1024]
    mlp_ratio = kwargs.pop("mlp_ratio", 2)
    layer_scale = kwargs.pop("layer_scale", 1e-5)
    # dilation_rates = [[1, 8, 1], [1, 4, 1, 4], [1, 2] * 9, [1, 1, 1, 1, 1]]
    use_every_other_dilations = True
    return NAT(**locals(), model_name="dinat_base", **kwargs)


@register_model
def DiNAT_Large(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet21k-ft1k", **kwargs):
    num_blocks = [3, 4, 18, 5]
    num_heads = [6, 12, 24, 48]
    out_channels = [192, 384, 768, 1536]
    mlp_ratio = kwargs.pop("mlp_ratio", 2)
    # layer_scale = kwargs.pop("layer_scale", 1e-5)
    # dilation_rates = [[1, 8, 1], [1, 4, 1, 4], [1, 2] * 9, [1, 1, 1, 1, 1]]
    use_every_other_dilations = True
    return NAT(**locals(), model_name="dinat_large", **kwargs)


@register_model
def DiNAT_Large_K11(input_shape=(384, 384, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet21k-ft1k", **kwargs):
    num_blocks = [3, 4, 18, 5]
    num_heads = [6, 12, 24, 48]
    out_channels = [192, 384, 768, 1536]
    mlp_ratio = kwargs.pop("mlp_ratio", 2)
    # layer_scale = kwargs.pop("layer_scale", 1e-5)
    # dilation_rates = [[1, 8, 1], [1, 4, 1, 4], [1, 2] * 9, [1, 1, 1, 1, 1]]
    use_every_other_dilations = True
    attn_kernel_size = 11
    return NAT(**locals(), model_name="dinat_large_k11", **kwargs)
