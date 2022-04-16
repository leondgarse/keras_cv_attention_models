from keras_cv_attention_models.nfnets.nfnets import (
    NormFreeNet,
    NFNetF0,
    NFNetF1,
    NFNetF2,
    NFNetF3,
    NFNetF4,
    NFNetF5,
    NFNetF6,
    NFNetF7,
    NormFreeNet_Light,
    NFNetL0,
    ECA_NFNetL0,
    ECA_NFNetL1,
    ECA_NFNetL2,
    ECA_NFNetL3,
    ScaledStandardizedConv2D,
    ZeroInitGain,
)


__head_doc__ = """
Keras implementation of [NFNets](https://github.com/deepmind/deepmind-research/tree/master/nfnets).
Paper [PDF 2102.06171 High-Performance Large-Scale Image Recognition Without Normalization](https://arxiv.org/pdf/2102.06171.pdf).
"""

__tail_doc__ = """  stem_width: output dimension for stem block. Default 128.
  out_channels: default as `[256, 512, 1536, 1536]`. Output channel for each stack.
  channel_ratio: filter scale ratio for block hidden layers. `0.5` for `F` original models, `0.25` for `L` light models.
  num_features_factor: none `0` value to added a `ScaledStandardizedConv2D` layer before output,
      added layer `filters = num_features_factor * out_channels[-1] * width_factor`.
      Default `1.5` for `L0` models, `2` for others.
  strides: list value indicates strides used in the each stack. Default `[1, 2, 2, 2]`.
  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
      Set `(None, None, 3)` for dynamic input resolution.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  se_ratio: value in `(0, 1)`, where `0` means not using `se_module`, indicates `se_ratio` for all blocks.
      Default `0.5` for `F` original models, `0.25` for `L` light models.
  group_size: groups size for all blocks 3x3 Conv2D layers, `groups = filters // group_size`.
      Default `128` for `F` original models, `64` for `L` light models.
  use_zero_init_gain: boolean value if add a `ZeroInitGain` layer for deep branch in all blocks.
      Default `True` for `F` original models, `False` for `L` light models.
  torch_padding: boolean value if using PyTorch like `padding`. Default `False` for `F` original models, `True` for `L` light models.
  gamma_in_act: boolean value if using non linear gamma in activation or conv2d layer.
      Default `True` for `F` original models, `False` for `L` light models.
  alpha: scale factor for deep branch in all blocks. Currently all using `0.2`.
  width_factor: width expansion for entire model. Currently all using `1.0`.
  activation: activation used in whole model.
  drop_connect_rate: is used for [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382).
      Can be value like `0.2`, indicates the drop probability linearly changes from `0 --> 0.2` for `top --> bottom` layers.
      A higher value means a higher probability will drop the deep branch.
      or `0` to disable (default).
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
      Default is `softmax`.
  dropout: dropout rate if top layers is included.
  pretrained: one of `None` (random initialization) or 'imagenet' (pre-training on ImageNet).
      Will try to download and load pre-trained model weights if not None.

Returns:
    A `keras.Model` instance.
"""

NormFreeNet.__doc__ = __head_doc__ + """
Args:
  num_blocks: number of blocks in each stack.
  attn_type: attention types for all blocks, could be `se` or `eca`. Default is `se`.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model       | Params | FLOPs   | Input | Top1 Acc |
  | ----------- | ------ | ------- | ----- | -------- |
  | NFNetL0     | 35.07M | 7.13G   | 288   | 82.75    |
  | NFNetF0     | 71.5M  | 12.58G  | 256   | 83.6     |
  | NFNetF1     | 132.6M | 35.95G  | 320   | 84.7     |
  | NFNetF2     | 193.8M | 63.24G  | 352   | 85.1     |
  | NFNetF3     | 254.9M | 115.75G | 416   | 85.7     |
  | NFNetF4     | 316.1M | 216.78G | 512   | 85.9     |
  | NFNetF5     | 377.2M | 291.73G | 544   | 86.0     |
  | NFNetF6 SAM | 438.4M | 379.75G | 576   | 86.5     |
  | NFNetF7     | 499.5M | 481.80G | 608   |          |
  | ECA_NFNetL0 | 24.14M | 7.12G   | 288   | 82.58    |
  | ECA_NFNetL1 | 41.41M | 14.93G  | 320   | 84.01    |
  | ECA_NFNetL2 | 56.72M | 30.12G  | 384   | 84.70    |
  | ECA_NFNetL3 | 72.04M | 52.73G  | 448   |          |
"""

NFNetF0.__doc__ = __head_doc__ + """
Args:
  attn_type: attention types for all blocks, could be `se` or `eca`. Default is `se`.
""" + __tail_doc__

NFNetF1.__doc__ = NFNetF0.__doc__
NFNetF2.__doc__ = NFNetF0.__doc__
NFNetF3.__doc__ = NFNetF0.__doc__
NFNetF4.__doc__ = NFNetF0.__doc__
NFNetF5.__doc__ = NFNetF0.__doc__
NFNetF6.__doc__ = NFNetF0.__doc__
NFNetF7.__doc__ = NFNetF0.__doc__

__light_head_doc__ = __head_doc__ + """Light versions of `NFNetF` from `timm`.
Weights reloaded from [Github rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models).
"""

NormFreeNet_Light.__doc__ = __light_head_doc__ + """
Args:
  num_blocks: number of blocks in each stack.
  attn_type: attention types for all blocks, could be `se` or `eca`. Default is `se`.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model       | Params | FLOPs   | Input | Top1 Acc |
  | ----------- | ------ | ------- | ----- | -------- |
  | NFNetL0     | 35.07M | 7.13G   | 288   | 82.75    |
  | ECA_NFNetL0 | 24.14M | 7.12G   | 288   | 82.58    |
  | ECA_NFNetL1 | 41.41M | 14.93G  | 320   | 84.01    |
  | ECA_NFNetL2 | 56.72M | 30.12G  | 384   | 84.70    |
  | ECA_NFNetL3 | 72.04M | 52.73G  | 448   |          |
"""

NFNetL0.__doc__ = __light_head_doc__ + """
Args:
  attn_type: attention types for all blocks, could be `se` or `eca`. Default is `se`.
""" + __tail_doc__

ECA_NFNetL0.__doc__ = __light_head_doc__ + """Model Using `attn_type="eca"` instead of `attn_type="se"`.

Args:
""" + __tail_doc__

ECA_NFNetL1.__doc__ = ECA_NFNetL0.__doc__
ECA_NFNetL2.__doc__ = ECA_NFNetL0.__doc__
ECA_NFNetL3.__doc__ = ECA_NFNetL0.__doc__

ScaledStandardizedConv2D.__doc__ = __head_doc__ + """
Scaled Standardized Conv2D layer, sub-class of typical Conv2D.

Additional args from Conv2D:
  gamma: is a scaling factor multipled to scale.
  eps: is small constant used to avoid dividing 0.

Examples:

>>> from keras_cv_attention_models import attention_layers
>>> aa = attention_layers.ScaledStandardizedConv2D(filters=64, kernel_size=3)
>>> print(f"{aa(tf.ones([1, 28, 28, 32])).shape = }")
# aa(tf.ones([1, 28, 28, 32])).shape = TensorShape([1, 26, 26, 64])
>>> print({ii.name:ii.shape for ii in aa.weights})
# {'scaled_standardized_conv2d/kernel:0': TensorShape([3, 3, 32, 64]),
#  'scaled_standardized_conv2d/bias:0': TensorShape([64]),
#  'scaled_standardized_conv2d/gain:0': TensorShape([64])}
"""

ZeroInitGain.__doc__ = __head_doc__ + """
Zero Init Gain.

Examples:

>>> from keras_cv_attention_models import attention_layers
>>> aa = attention_layers.ZeroInitGain()
>>> print(f"{aa(tf.ones([1, 28, 28, 32])).shape = }")
# aa(tf.ones([1, 28, 28, 32])).shape = TensorShape([1, 26, 26, 64])
>>> print({ii.name:ii.shape for ii in aa.weights})
# {'zero_init_gain/gain:0': TensorShape([])}
>>> print(aa.gain.numpy())
# 0.0
"""
