# ___Keras AotNet___
***

## Summary
  - `AotNet` is just a `ResNet` / `ResNetV2` like framework, that set parameters like `attn_types` and `attn_params` and others, which is used to apply different types attention layers. Works like `byoanet` / `byobnet` from `timm`.
  - Default parameters set is a typical `ResNet` architecture with `Conv2D use_bias=False` and `padding` like `PyTorch`.
  - `AotNet` means `Attention Over Template network`! Honestly, just a name after `BotNet` and `CotNet`...
***

## Usage
  - **attn_types** is a `string` or `list`, indicates attention layer type for each stack. Each element can also be a `string` or `list`, indicates attention layer type for each block.
    - `"bot"`: `mhsa_with_relative_position_embedding` from `botnet`.
    - `"cot"`: `cot_attention` from `cotnet`.
    - `"halo"`: `halo_attention` from `halonet`.
    - `"outlook"`: `outlook_attention` from `volo`.
    - `"sa"`: `split_attention_conv2d` from `resnest`.
    - `None`: `Conv2D`. Can add `groups` like `ResNeXt` or add `se` and `eca` attention.
  - **attn_params**: like `attn_types`, is a dict or list, each element in list can also be a dict or list. Indicates specific attention layer parameters for relative `attn_types`.
  - **se_ratio**: value in `(0, 1)`, where `0` means not using `se_module`. Should be a `number` or `list`, indicates `se_ratio` for each stack. Each element can also be a `number` or `list`, indicates `se_ratio` for each block.
  - **use_eca**: boolean value if use `eca` attention. Can also be a list like `se_ratio`.
  - **groups**: `groups` for `Conv2D` layer if relative `attn_types` is `None`. `ResNeXt` like archeticture. Note it's NOT the `group_size`. Default value `1` means not using group.
  - **Definition of `BotNet26T`**
    ```py
    from keras_cv_attention_models import aotnet
    model = aotnet.AotNet(
        num_blocks=[2, 2, 2, 2],
        attn_types=[None, None, [None, "bot"], "bot"],
        attn_params={"num_heads": 4, "out_weight": False},
        stem_type="tiered",
        input_shape=(256, 256, 3),
        model_name="botnet26t",
    )
    model.summary()
    ```
  - **Definition of `CotNet101`**
    ```py
    from keras_cv_attention_models import aotnet
    model = aotnet.AotNet101(
        attn_types="cot",
        bn_after_attn=False,
        shortcut_type="avg",
        model_name="cotnet101",
    )
    model.summary()
    ```
  - **Definition of `HaloNet50T`**
    ```py
    from keras_cv_attention_models import aotnet
    attn_params = [
        None,
        [None, None, None, {"block_size": 8, "halo_size": 3, "num_heads": 4, "out_weight": False}],
        [None, {"block_size": 8, "halo_size": 3, "num_heads": 8, "out_weight": False}] * 3,
        [None, {"block_size": 8, "halo_size": 3, "num_heads": 8, "out_weight": False}, None],
    ]
    model = aotnet.AotNet50(
        attn_types=[None, [None, None, None, "halo"], [None, "halo"] * 3, [None, "halo", None]],
        attn_params=attn_params,
        stem_type="tiered",
        input_shape=(256, 256, 3),
        model_name="halonet50t",
    )
    model.summary()
    ```
  - **Definition of `ResNest50`**
    ```py
    from keras_cv_attention_models import aotnet
    model = aotnet.AotNet50(
        stem_type="deep",
        shortcut_type="avg",
        attn_types="sa",
        bn_after_attn=False,
        model_name="resnest50",
    )
    model.summary()
    ```
  - **Mixing se and outlook and halo and bot and cot**, 21M parameters
    ```py
    # 50 is just a picked number that larger than the relative `num_block`
    model = aotnet.AotNet50V2(
        attn_types=[None, "outlook", ["bot", "halo"] * 50, "cot"],
        se_ratio=[0.25, 0, 0, 0],
        stem_type="deep",
        strides=1,
    )
    model.summary()
    ```
  - `AotNet50V2` / `AotNet101V2` / `AotNet152V2` / `AotNet200V2` is the `ResNetV2` like template.
***
