from keras_cv_attention_models.beit.beit import Beit, keras_model_load_weights_from_pytorch_model
from keras_cv_attention_models.models import register_model


def FlexiViT(
    attn_qv_bias=False,
    attn_qkv_bias=True,
    use_abs_pos_emb=True,
    use_abs_pos_emb_on_cls_token=False,  # no_embed_class in timm
    layer_scale=0,
    use_mean_pooling_head=False,
    model_name="flexivit",
    **kwargs,
):
    kwargs.pop("kwargs", None)
    patch_size = kwargs.pop("patch_size", 16)
    force_reload_mismatch = patch_size != 16  # If patch_size not 16, force reload pos_emb and stem_conv weights
    return Beit(**locals(), **kwargs)


@register_model
def FlexiViTSmall(input_shape=(240, 240, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    embed_dim = 384
    depth = 12
    num_heads = 6
    return FlexiViT(**locals(), model_name="flexivit_small", **kwargs)


@register_model
def FlexiViTBase(input_shape=(240, 240, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    embed_dim = 768
    depth = 12
    num_heads = 12
    return FlexiViT(**locals(), model_name="flexivit_base", **kwargs)


@register_model
def FlexiViTLarge(input_shape=(240, 240, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    embed_dim = 1024
    depth = 24
    num_heads = 16
    return FlexiViT(**locals(), model_name="flexivit_large", **kwargs)
