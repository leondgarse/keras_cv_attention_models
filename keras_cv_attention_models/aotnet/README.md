# ___Keras AotNet___
***

## Summary
  - `AotNet` is just a `ResNet` / `ResNetV2` like framework, that set parameters like `attn_types` and `se_ratio` and others, which is used to apply different types attention layer.
  - `AotNet` means...`Attention Over Template network`! Honestly, just a name after `BotNet` and `CotNet`...
***

## Usage
  - **attn_types** is a `string` or `list`, indicates attention layer type for each stack. Each element can also be a `string` or `list`, indicates attention layer type for each block.
    - `None`: `Conv2D`
    - `"cot"`: `attention_layers.cot_attention`. Default values: `kernel_size=3`.
    - `"halo"`: `attention_layers.HaloAttention`. Default values: `num_heads=8, key_dim=16, block_size=4, halo_size=1, out_bias=True`.
    - `"mhsa"`: `attention_layers.MHSAWithPositionEmbedding`. Default values: `num_heads=4, relative=True, out_bias=True`.
    - `"outlook"`: `attention_layers.outlook_attention`. Default values: `num_head=6, kernel_size=3`.
    - `"sa"`: `attention_layers.split_attention_conv2d`. Default values: `kernel_size=3, groups=2`.
  - **se_ratio** value in `(0, 1)`, where `0` means not using `se_module`. Should be a `number` or `list`, indicates `se_ratio` for each stack. Each element can also be a `number` or `list`, indicates `se_ratio` for each block.
  - **Apply different attention layers**
    ```py
    # basic ResNet50 like, 25.6M parameters
    mm = aotnet.AotNet50(attn_types=None, deep_stem=False, strides=2)

    # se_ir_r50 like, 22.4M parameters
    mm = aotnet.AotNet50(expansion=1, deep_stem=False, se_ratio=0.25, stem_downsample=False, strides=2)

    # ResNest50 like, 27.6M parameters
    mm = aotnet.AotNet50(attn_types="sa", deep_stem=True, strides=2)

    # BotNet50 like, 19.7M parameters
    mm = aotnet.AotNet50(attn_types=[None, None, "mhsa", "mhsa"], deep_stem=False, strides=1)

    # HaloNet like, 16.2M parameters
    mm = aotnet.AotNet50(attn_types="halo", deep_stem=False, strides=[1, 1, 1, 1])

    # CotNet50 like, 22.2M parameters
    mm = aotnet.AotNet50(attn_types="cot", deep_stem=False, strides=2)

    # SECotnetD50 like, 23.5M parameters
    mm = aotnet.AotNet50(attn_types=["sa", "sa", ["cot", "sa"] * 50, "cot"], deep_stem=True, strides=2)
    ```
  - **Mixing different attention layers together**
    ```py
    # Mixing se and outlook and halo and mhsa and cot_attention, 21M parameters
    # 50 is just a picked number that larger than the relative `num_block`
    attn_types = [None, "outlook", ["mhsa", "halo"] * 50, "cot"]
    mm = aotnet.AotNet50V2(attn_types=attn_types, se_ratio=[0.25, 0, 0, 0], deep_stem=True, strides=1)
    ```
  - `AotNet50V2` / `AotNet101V2` / `AotNet152V2` / `AotNet200V2` is the `ResNetV2` like template.
***
