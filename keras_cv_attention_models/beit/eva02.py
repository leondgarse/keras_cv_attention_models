from keras_cv_attention_models.beit.beit import Beit, keras_model_load_weights_from_pytorch_model
from keras_cv_attention_models.models import register_model


def EVA02(mlp_ratio=4 * 2 / 3, layer_scale=0, use_abs_pos_emb=True, use_rot_pos_emb=True, use_gated_mlp=True, activation="swish", model_name="eva02", **kwargs):
    kwargs.pop("kwargs", None)
    patch_size = kwargs.pop("patch_size", 14)
    force_reload_mismatch = patch_size != 14  # If patch_size not 14, force reload pos_emb and stem_conv weights
    return Beit(**locals(), **kwargs)


@register_model
def EVA02TinyPatch14(input_shape=(336, 336, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="mim_in22k_ft1k", **kwargs):
    embed_dim = 192
    depth = 12
    num_heads = 3
    return EVA02(**locals(), model_name="eva02_tiny_patch14", **kwargs)


@register_model
def EVA02SmallPatch14(input_shape=(336, 336, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="mim_in22k_ft1k", **kwargs):
    embed_dim = 384
    depth = 12
    num_heads = 6
    return EVA02(**locals(), model_name="eva02_small_patch14", **kwargs)


@register_model
def EVA02BasePatch14(
    input_shape=(448, 448, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="mim_in22k_ft22k_ft1k", **kwargs
):
    embed_dim = 768
    depth = 12
    num_heads = 12
    use_norm_mlp = True  # scale_mlp = True
    return EVA02(**locals(), model_name="eva02_base_patch14", **kwargs)


@register_model
def EVA02LargePatch14(
    input_shape=(448, 448, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="mim_m38m_ft22k_ft1k", **kwargs
):
    embed_dim = 1024
    depth = 24
    num_heads = 16
    use_norm_mlp = True  # scale_mlp = True
    return EVA02(**locals(), model_name="eva02_large_patch14", **kwargs)
