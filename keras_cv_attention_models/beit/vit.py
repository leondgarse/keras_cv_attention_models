from keras_cv_attention_models.beit.beit import Beit, keras_model_load_weights_from_pytorch_model
from keras_cv_attention_models.models import register_model


def ViT(attn_qv_bias=False, attn_qkv_bias=True, use_abs_pos_emb=True, layer_scale=0, use_mean_pooling_head=False, model_name="vit", **kwargs):
    kwargs.pop("kwargs", None)
    return Beit(**locals(), **kwargs)


def ViTText(
    vocab_size=49408,
    max_block_size=77,
    text_positional_dropout=0,
    text_use_positional_embedding=True,
    include_top=True,
    layer_norm_epsilon=1e-5,
    activation="gelu/quick",
    model_name="vit_text",
    **kwargs,
):
    attn_qv_bias = kwargs.pop("attn_qv_bias", False)
    attn_qkv_bias = kwargs.pop("attn_qkv_bias", True)
    use_abs_pos_emb = kwargs.pop("use_abs_pos_emb", True)
    layer_scale = kwargs.pop("layer_scale", 0)
    use_mean_pooling_head = kwargs.pop("use_mean_pooling_head", False)
    kwargs.pop("kwargs", None)
    return Beit(**locals(), **kwargs)


@register_model
def ViTTinyPatch16(input_shape=(196, 196, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet21k-ft1k", **kwargs):
    embed_dim = 192
    depth = 12
    num_heads = 3
    patch_size = kwargs.pop("patch_size", 16)
    return ViT(**locals(), model_name="vit_tiny_patch16", **kwargs)


@register_model
def ViTBasePatch16(input_shape=(196, 196, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet21k-ft1k", **kwargs):
    embed_dim = 768
    depth = 12
    num_heads = 12
    patch_size = kwargs.pop("patch_size", 16)
    return ViT(**locals(), model_name="vit_base_patch16", **kwargs)


@register_model
def ViTLargePatch14(input_shape=(196, 196, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet21k-ft1k", **kwargs):
    embed_dim = 1024
    depth = 24
    num_heads = 16
    patch_size = kwargs.pop("patch_size", 14)
    return ViT(**locals(), model_name="vit_large_patch14", **kwargs)


@register_model
def ViTTextLargePatch14(vocab_size=49408, max_block_size=77, activation="gelu/quick", include_top=True, pretrained="clip", **kwargs):
    embed_dim = 768
    depth = 12
    num_heads = 12
    patch_size = kwargs.pop("patch_size", 14)
    return ViTText(**locals(), model_name="vit_text_large_patch14", **kwargs)
