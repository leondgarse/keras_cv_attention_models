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
    DEFAULT_PARAMS,
)

__head_doc__ = """
AotNet is just a `ResNet` / `ResNetV2` like framework.
Set parameters like `attn_types` and `attn_params` and others, which is used to apply different types attention layers.
Default parameters set is a typical `ResNet` architecture with `Conv2D use_bias=False` and `padding` like `PyTorch`.
"""

__tail_doc__ = """
  [Stack parameters]
  preact: whether to use pre-activation or not. Default `False` for ResNet like, `True` for ResNetV2 like.
  strides: a `number` or `list`, indicates strides used in the last stack or list value for all stacks.
      If a number, for `ResNet` like models, it will be `[1, 2, 2, strides]`.
      If a number, for `ResNetV2` like models, it will be `[2, 2, 2, strides]`.
      Default [1, 2, 2, 2]
  strides_first: boolean value if use downsample in the first block in each stack.
      Default `True` for ResNet like, `False` for ResNetV2 like.
  out_channels: default as `[64, 128, 256, 512]`. Output channel for each stack.
  hidden_channel_ratio: filter expansion for each block hidden layers. The larger the wider. For default `ResNet` / `ResNetV2` like, it's `0.25`.
  use_3x3_kernel: boolean value if use two `3x3` conv instead of `1x1 + 3x3 + 1x1` conv in each block. Default `False`.
  use_block_output_activation: boolean value if add activation after block `Add` layer. Default `True`.

  [Stem parameters]
  stem_width: output dimension for stem block. Default 64.
  stem_type: string in value `[None, "deep", "quad", "tiered", "kernel_3x3"]`. Indicates diffrerent stem type. Default None.
  quad_stem_act: boolean value if add `BN + act` after first `Conv` layer for `stem_type="quad"`. Default `False`.
  stem_last_strides: the last strides in stem block. Default `1`
  stem_downsample: boolean value if add `MaxPooling2D` layer after stem block. Default `True`.

  [Attention block parameters]
  attn_types: is a `string` or `list`, indicates attention layer type for each stack.
      Each element can also be a `string` or `list`, indicates attention layer type for each block.
      - `"bot"`: `mhsa_with_relative_position_embedding` from `botnet`. Default values: `num_heads=4, relative=True, out_bias=False`.
      - `"cot"`: `cot_attention` from `cotnet`. Default values: `kernel_size=3, downsample_first=True`.
      - `"halo"`: `halo_attention` from `halonet`. Default values: `num_heads=8, block_size=4, halo_size=1`.
      - `"outlook"`: `outlook_attention` from `volo`. Default values: `num_head=8, kernel_size=3`.
      - `"sa"`: `split_attention_conv2d` from `resnest`. Default values: `kernel_size=3, groups=2, downsample_first=False`.
      - `None`: `Conv2D`. Can add `groups` like `ResNeXt` or add `se` and `eca` attention.
  attn_params: like `attn_types`, is a dict or list, each element in list can also be a dict or list.
      Indicates specific attention layer parameters for relative `attn_types`.
  se_ratio: value in `(0, 1)`, where `0` means not using `se_module`. Should be a `number` or `list`, indicates `se_ratio` for each stack.
      Each element can also be a `number` or `list`, indicates `se_ratio` for each block.
  use_eca: boolean value if use `eca` attention. Can also be a list like `se_ratio`.
  groups: `groups` for `Conv2D` layer if relative `attn_types` is `None`. `ResNeXt` like archeticture. Note it's NOT the `group_size`.
      Default value `1` means not using group.
  group_size: group size for `Conv2D` layer if relative `attn_types` is `None`. `groups = filters / group_size`.
      Priority is higher than `groups`. Default `0` means not using group.
  bn_after_attn: boolean value if add `batchnorm + activation` layers after attention layer. Default `True`.

  [Shortcut branch parameters]
  shortcut_type: value in `["conv", "avg", "anti_alias", None]`. Indicates shortcut branch type if should apply `conv_shortcut`.
      - "conv": basic `Resnet` like conv branch.
      - "avg": `AvgPool2D + Conv2D`.
      - "anti_alias": `anti_alias_downsample + Conv2D`.
      - None: no shortcut.
      Default is `"conv"`.

  [Model common parameters]
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
  output_num_features: none `0` value to add another `conv2d + bn + activation` layers before `GlobalAveragePooling2D`.
  dropout: dropout rate if top layers is included.
  pretrained: mostly it's `None` or `"imagenet"` if any.
  kwargs: Not used, only recieving parameter.

Returns:
    A `keras.Model` instance.
"""

AotNet.__doc__ = __head_doc__ + """
Args:
  num_blocks: number of blocks in each stack.
  model_name: string, model name.
""" + __tail_doc__ + """
Example:

Definition of `BotNet26T`
>>> from keras_cv_attention_models import aotnet
>>> model = aotnet.AotNet(
>>>     num_blocks=[2, 2, 2, 2],
>>>     attn_types=[None, None, [None, "bot"], "bot"],
>>>     attn_params={"num_heads": 4, "out_weight": False},
>>>     stem_type="tiered",
>>>     input_shape=(256, 256, 3),
>>>     model_name="botnet26t",
>>> )
>>> model.summary()

Definition of `HaloNet50T`
>>> from keras_cv_attention_models import aotnet
>>> attn_params = [
>>>     None,
>>>     [None, None, None, {"block_size": 8, "halo_size": 3, "num_heads": 4, "out_weight": False}],
>>>     [None, {"block_size": 8, "halo_size": 3, "num_heads": 8, "out_weight": False}] * 3,
>>>     [None, {"block_size": 8, "halo_size": 3, "num_heads": 8, "out_weight": False}, None],
>>> ]
>>> model = aotnet.AotNet(
>>>     num_blocks=[3, 4, 6, 3],
>>>     attn_types=[None, [None, None, None, "halo"], [None, "halo"] * 3, [None, "halo", None]],
>>>     attn_params=attn_params,
>>>     stem_type="tiered",
>>>     input_shape=(256, 256, 3),
>>>     model_name="halonet50t",
>>> )
>>> model.summary()

Mixing se and outlook and halo and bot and cot, 21M parameters
>>> # 50 is just a picked number that larger than the relative `num_block`
>>> model = aotnet.AotNet50V2(
>>>     attn_types=[None, "outlook", ["bot", "halo"] * 50, "cot"],
>>>     se_ratio=[0.25, 0, 0, 0],
>>>     stem_type="deep",
>>>     strides=1,
>>> )
>>> model.summary()
"""

AotNet50.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

AotNet101.__doc__ = AotNet50.__doc__
AotNet152.__doc__ = AotNet50.__doc__

AotNetV2.__doc__ = AotNet.__doc__
AotNet50V2.__doc__ = AotNet50.__doc__
AotNet101V2.__doc__ = AotNet50.__doc__
AotNet152V2.__doc__ = AotNet50.__doc__
