from keras_cv_attention_models.beit.beit import Beit
from keras_cv_attention_models.models import register_model


def ViT5(
    with_cls_token=True,
    num_reg_tokens=4,
    use_qk_norm=True,
    use_rms_norm=True,
    use_abs_pos_emb=True,
    use_abs_pos_emb_on_cls_token=False,
    use_rot_pos_emb=True,
    use_mean_pooling_head=False,
    layer_scale=0.1,
    ref_feature_shape=-1,
    reg_ref_feature_shape=2,
    attn_qv_bias=False,
    attn_qkv_bias=False,
    attn_out_bias=True,
    model_name="vit5",
    **kwargs,
):
    kwargs.pop("kwargs", None)
    if ref_feature_shape <= 0:
        ref_feature_shape = kwargs.get("input_shape", (224, 224, 3))[0] // kwargs.get("patch_size", 16)
    return Beit(**locals(), **kwargs)


@register_model
def ViT5_Small_Patch16(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    embed_dim = 384
    num_heads = 6
    return ViT5(**locals(), model_name="vit5_small_patch16", **kwargs)


@register_model
def ViT5_Base_Patch16(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    embed_dim = 768
    num_heads = 12
    return ViT5(**locals(), model_name="vit5_base_patch16", **kwargs)


@register_model
def ViT5_Large_Patch16(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    depth = 24
    embed_dim = 1024
    num_heads = 16
    return ViT5(**locals(), model_name="vit5_large_patch16", **kwargs)
