from keras_cv_attention_models.beit.beit import Beit, keras_model_load_weights_from_pytorch_model
from keras_cv_attention_models.models import register_model


def EVA(layer_scale=0, use_abs_pos_emb=True, model_name="eva", **kwargs):
    kwargs.pop("kwargs", None)
    patch_size = kwargs.pop("patch_size", 14)
    force_reload_mismatch = patch_size != 14  # If patch_size not 14, force reload pos_emb and stem_conv weights
    return Beit(**locals(), **kwargs)


@register_model
def EvaLargePatch14(input_shape=(196, 196, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet21k-ft1k", **kwargs):
    embed_dim = 1024
    depth = 24
    num_heads = 16
    attn_qkv_bias = True
    return EVA(**locals(), model_name="eva_large_patch14", **kwargs)


@register_model
def EvaGiantPatch14(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet21k-ft1k", **kwargs):
    mlp_ratio = 6144 / 1408
    embed_dim = 1408
    depth = 40
    num_heads = 16
    return EVA(**locals(), model_name="eva_giant_patch14", **kwargs)
