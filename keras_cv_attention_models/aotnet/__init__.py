from keras_cv_attention_models.aotnet.aotnet import (
    AotNet,
    AotNet50,
    AotNet101,
    AotNet152,
    AotNet200,
    AotNetV2,
    AotNet50V2,
    AotNet101V2,
    AotNet152V2,
    AotNet200V2,
    MHSA_PARAMS,
    HALO_PARAMS,
    SA_PARAMS,
    COT_PARAMS,
    OUTLOOK_PARAMS,
    GROUPS_CONV_PARAMS,
    attn_block,
    anti_alias_downsample,
    batchnorm_with_activation,
    conv2d_no_bias,
    se_module,
    block,
    stack,
    stem,
)


__head_doc__ = """
AotNet is just a `ResNet` / `ResNetV2` like framework.
Set parameters like `attn_types` and `se_ratio` and others, which is used to apply different types attention layer.
"""

__tail_doc__ = """  out_channels: default as `[64, 128, 256, 512]`. Output channel for each stack.
  expansion: filter expansion for each block output channel. The larger the wider. For default `ResNet` / `ResNetV2` like, it's `4`.
  stem_width: output dimension for stem block.
  deep_stem: Boolean value if use deep stem.
  stem_downsample: Boolean value if add `MaxPooling2D` layer after stem block.
  attn_types: is a `string` or `list`, indicates attention layer type for each stack.
      Each element can also be a `string` or `list`, indicates attention layer type for each block.
      - `None`: `Conv2D`
      - `"cot"`: `attention_layers.cot_attention`. Default values: `kernel_size=3`.
      - `"groups_conv"`: `Conv2D` with `groups=32`. `ResNeXt` method.
      - `"halo"`: `attention_layers.HaloAttention`. Default values: `num_heads=8, key_dim=16, block_size=4, halo_size=1, out_bias=True`.
      - `"mhsa"`: `attention_layers.MHSAWithPositionEmbedding`. Default values: `num_heads=4, relative=True, out_bias=True`.
      - `"outlook"`: `attention_layers.outlook_attention`. Default values: `num_head=6, kernel_size=3`.
      - `"sa"`: `attention_layers.split_attention_conv2d`. Default values: `kernel_size=3, groups=2`.
  se_ratio: value in `(0, 1)`, where `0` means not using `se_module`. Should be a `number` or `list`, indicates `se_ratio` for each stack.
      Each element can also be a `number` or `list`, indicates `se_ratio` for each block.
  num_features: none `0` value to add another `conv2d + bn + activation` layers before `GlobalAveragePooling2D`.
  use_3x3_kernel: Boolean value if use two `3x3` conv instead of `1x1 + 3x3 + 1x1` conv in each block.
  avg_pool_down: Boolean value if use `AvgPool2D` in shortcut branch. `True` for `ResNetD` model like.
  anti_alias_down: Boolean value if use `anti_alias_downsample` in shortcut branch.
  input_shape: it should have exactly 3 inputs channels, default `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation used in whole model, default `relu`.
  drop_connect_rate: is used for [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382).
      Can be value like `0.2`, indicates the drop probability linearly changes from `0 --> 0.2` for `top --> bottom` layers.
      A higher value means a higher probability will drop the deep branch.
      or `0` to disable (default).
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
      Default is `softmax`.
  **kwargs: other parameters if available.

Returns:
    A `keras.Model` instance.
"""

AotNet.__doc__ = __head_doc__ + """
Args:
  num_blocks: number of blocks in each stack.
  preact: whether to use pre-activation or not. False for ResNet like, True for ResNetV2 like.
  strides: a `number` or `list`, indicates strides used in the last stack or list value for all stacks.
      If a number, for `ResNet` like models, it will be `[1, 2, 2, strides]`.
      If a number, for `ResNetV2` like models, it will be `[2, 2, 2, strides]`.
  strides_first: Boolean value if use downsample in the first block in each stack.
      `True` for ResNet like, `False` for ResNetV2 like.
  model_name: string, model name.
""" + __tail_doc__ + """
Example:
>>> # basic ResNet50 like, 25.6M parameters
>>> mm = aotnet.AotNet50(attn_types=None, deep_stem=False, strides=2)

>>> # se_ResNetRS50 like, 22.4M parameters
>>> mm = aotnet.AotNet50(expansion=1, deep_stem=False, se_ratio=0.25, stem_down_sample=False, strides=2)

>>> # ResNest50 like, 27.6M parameters
>>> mm = aotnet.AotNet50(attn_types="sa", deep_stem=True, strides=2)

>>> # BotNet50 like, 19.7M parameters
>>> mm = aotnet.AotNet50(attn_types=[None, None, "mhsa", "mhsa"], deep_stem=False, strides=1)

>>> # HaloNet like, 16.2M parameters
>>> mm = aotnet.AotNet50(attn_types="halo", deep_stem=False, strides=[1, 2, 2, 2])

>>> # CotNet50 like, 22.2M parameters
>>> mm = aotnet.AotNet50(attn_types="cot", deep_stem=False, strides=2)

>>> # SECotnetD50 like, 23.5M parameters
>>> mm = aotnet.AotNet50(attn_types=["sa", "sa", ["cot", "sa"] * 50, "cot"], deep_stem=True, strides=2)

>>> # Mixing se and outlook and halo and mhsa and cot_attention, 21M parameters
>>> # 50 is just a picked number that larger than the relative `num_block`
>>> mm = aotnet.AotNet50V2(attn_types=[None, "outlook", ["mhsa", "halo"] * 50, "cot"], se_ratio=[0.25, 0.25, 0, 0], deep_stem=True, strides=1)
"""

AotNet50.__doc__ = __head_doc__ + """
Args:
  strides: a `number` or `list`
      number value indicates strides used in the last stack, final `strides` will be `[1, 2, 2, strides]`,
      or list value for all stacks.
""" + __tail_doc__

AotNet101.__doc__ = AotNet50.__doc__
AotNet152.__doc__ = AotNet50.__doc__

AotNetV2.__doc__ = AotNet.__doc__
AotNet50V2.__doc__ = __head_doc__ + """
Args:
  strides: a `number` or `list`
      number value indicates strides used in the last stack, final `strides` will be `[2, 2, 2, strides]`,
      or list value for all stacks.
""" + __tail_doc__
AotNet101V2.__doc__ = AotNet50V2.__doc__
AotNet152V2.__doc__ = AotNet50V2.__doc__
