from keras_cv_attention_models.beit.beit import Beit, keras_model_load_weights_from_pytorch_model
from keras_cv_attention_models.models import register_model


def MetaTransformer(
    use_patch_bias=False,
    use_pre_norm=True,
    use_abs_pos_emb=True,
    attn_qv_bias=False,
    attn_qkv_bias=True,
    use_mean_pooling_head=False,
    layer_scale=0,
    model_name="meta_transformer",
    **kwargs,
):
    kwargs.pop("kwargs", None)
    return Beit(**locals(), **kwargs)


@register_model
def MetaTransformerBasePatch16(
    input_shape=(384, 384, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="laion_2b", **kwargs
):
    depth = 12
    embed_dim = 768
    num_heads = 12
    patch_size = kwargs.pop("patch_size", 16)
    force_reload_mismatch = patch_size != 16  # If patch_size not match, force reload pos_emb and stem_conv weights
    return MetaTransformer(**locals(), model_name="meta_transformer_base_patch16", **kwargs)


@register_model
def MetaTransformerLargePatch14(
    input_shape=(336, 336, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="laion_2b", **kwargs
):
    depth = 24
    embed_dim = 1024
    num_heads = 16
    patch_size = kwargs.pop("patch_size", 14)
    force_reload_mismatch = patch_size != 14  # If patch_size not match, force reload pos_emb and stem_conv weights
    return MetaTransformer(**locals(), model_name="meta_transformer_large_patch14", **kwargs)
