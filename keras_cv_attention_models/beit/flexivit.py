from keras_cv_attention_models.beit.beit import Beit, keras_model_load_weights_from_pytorch_model


def ViT(
    depth=12,
    embed_dim=768,
    num_heads=12,
    mlp_ratio=4,
    patch_size=16,
    attn_key_dim=0,
    attn_qv_bias=False,  # Deault False for Vit, True for Beit, if True and attn_qkv_bias seing False, will add BiasLayer for query and key.
    attn_qkv_bias=True,  # True for Vit, False for Beit, if True, will just use bias in qkv_dense, and set qv_bias False.
    attn_out_weight=True,
    attn_out_bias=True,
    attn_dropout=0,
    gamma_init_value=0,  # 0 for Vit, 0.1 for Beit, if > 0 will use `layer_scale` on block output
    use_abs_pos_emb=True,  # True for Vit, False for Beit, whether use abcolute positional embedding or relative one in attention blocks
    use_abs_pos_emb_on_cls_token=True,  # False for FlexiViT, no_embed_class in timm. If use_abs_pos_emb is True, whether apply pos_emb on cls_token.
    use_mean_pooling=False,  # False for Vit, True for Beit, whether use use mean output or `class_token` output
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="gelu",
    drop_connect_rate=0,
    classifier_activation="softmax",
    pretrained=None,
    force_reload_mismatch=False,  # set True if patch_size changed, will force reloading pos_emb and stem_conv weights
    model_name="vit",
    kwargs=None,
):
    return Beit(**locals())


""" FlexiViT """


def FlexiViTSmall(input_shape=(240, 240, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    embed_dim = 384
    depth = 12
    num_heads = 6
    use_abs_pos_emb_on_cls_token = False  # no_embed_class in timm
    force_reload_mismatch = kwargs.get("patch_size", 16) != 16  # If patch_size not 16, force reload pos_emb and stem_conv weights
    return ViT(**locals(), model_name="flexivit_small", **kwargs)


def FlexiViTBase(input_shape=(240, 240, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    embed_dim = 768
    depth = 12
    num_heads = 12
    use_abs_pos_emb_on_cls_token = False  # no_embed_class in timm
    force_reload_mismatch = kwargs.get("patch_size", 16) != 16  # If patch_size not 16, force reload pos_emb and stem_conv weights
    return ViT(**locals(), model_name="flexivit_base", **kwargs)


def FlexiViTLarge(input_shape=(240, 240, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    embed_dim = 1024
    depth = 24
    num_heads = 16
    use_abs_pos_emb_on_cls_token = False  # no_embed_class in timm
    force_reload_mismatch = kwargs.get("patch_size", 16) != 16  # If patch_size not 16, force reload pos_emb and stem_conv weights
    return Beit(**locals(), model_name="flexivit_large", **kwargs)
