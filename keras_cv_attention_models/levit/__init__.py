from keras_cv_attention_models.levit.levit import (
    LeViT,
    LeViT128S,
    LeViT128,
    LeViT192,
    LeViT256,
    LeViT384,
    MultiHeadPositionalEmbedding,
    mhsa_with_multi_head_position_and_strides,
    mhsa_with_multi_head_position,
)


__head_doc__ = """
Keras implementation of [Github facebookresearch/LeViT](https://github.com/facebookresearch/LeViT).
Paper [PDF 2104.01136 LeViT: a Vision Transformer in ConvNet’s Clothing for Faster Inference](https://arxiv.org/pdf/2104.01136.pdf).
"""

__tail_doc__ = """  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation used in whole model, default `hard_swish`.
  drop_connect_rate: is used for [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382).
      Can be a constant value like `0.2`,
      or a tuple value like `(0, 0.2)` indicates the drop probability linearly changes from `0 --> 0.2` for `top --> bottom` layers.
      A higher value means a higher probability will drop the deep branch.
      or `0` to disable (default).
  dropout: top dropout rate if top layers is included. Default 0.
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
      Default is `None`.
  use_distillation: Boolean value if output `distill_head`. Default `True`.
  pretrained: one of `None` (random initialization) or 'imagenet' (pre-training on ImageNet).
      Will try to download and load pre-trained model weights if not None.
  **kwargs: other parameters if available.

Returns:
    A `keras.Model` instance.
"""

LeViT.__doc__ = __head_doc__ + """
Args:
  patch_channel: channel dimension output for `path_stem`.
  out_channels: output channels for each stack.
  num_heads: heads number for transformer blocks in each stack.
  depthes: number of block for each stack.
  key_dims: key dimension for transformer blocks in each stack.
  attn_ratios: `value` channel dimension expansion for transformer blocks in each stack.
  mlp_ratios: dimension expansion ration for `mlp_block` in each stack.
  strides: strides for each stack.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model                   | Params | FLOPs | Input | Top1 Acc |
  | ----------------------- | ------ | ----- | ----- | -------- |
  | LeViT128S, distillation | 7.8M   | 0.31G | 224   | 76.6     |
  | LeViT128, distillation  | 9.2M   | 0.41G | 224   | 78.6     |
  | LeViT192, distillation  | 11M    | 0.66G | 224   | 80.0     |
  | LeViT256, distillation  | 19M    | 1.13G | 224   | 81.6     |
  | LeViT384, distillation  | 39M    | 2.36G | 224   | 82.6     |
"""

LeViT128S.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

LeViT128.__doc__ = LeViT128S.__doc__
LeViT192.__doc__ = LeViT128S.__doc__
LeViT256.__doc__ = LeViT128S.__doc__
LeViT384.__doc__ = LeViT128S.__doc__

MultiHeadPositionalEmbedding.__doc__ = __head_doc__ + """
Multi Head Positional Embedding layer.

Positional embedding shape is `[height * width, num_heads]`.
input: `[batch, num_heads, query_blocks, key_blocks]`.
output: `[batch, num_heads, query_blocks, key_blocks] + positional_bias`.
conditions: key_height >= query_height, key_width >= query_width

Args:
  query_height: specify `height` for `query_blocks` if not square `query_height != query_width`.
  key_height: specify `height` for `key_blocks`.
      `-1` for auto calculating from `query_height` and `kk_blocks / qq_blocks` ratio. Works in most cases.

Examples:

>>> from keras_cv_attention_models import attention_layers
>>> aa = attention_layers.MultiHeadPositionalEmbedding()
>>> print(f"{aa(tf.ones([1, 8, 16, 49])).shape = }")
# aa(tf.ones([1, 8, 16, 49])).shape = TensorShape([1, 8, 16, 49])

>>> print({ii.name:ii.numpy().shape for ii in aa.weights})
# {'multi_head_positional_embedding/positional_embedding:0': (49, 8)}

>>> plt.imshow(aa.bb_pos)
"""

mhsa_with_multi_head_position.__doc__ = __head_doc__ + """
Multi Head Self Attention with MultiHeadPositionalEmbedding.
Using additional `BatchNormalization` for `query / key / value`,
and adding `MultiHeadPositionalEmbedding` to `attention_scores`.

input: `[batch, height, width, channel]`.
output: `[batch, height, width, channel]`.

Args:
  inputs: Input tensor.
  num_heads: Number of attention heads.
  key_dim: Size of each attention head for query and key. Default `-1` for `key_dim = input_channels // num_heads`.
  output_dim: The expected channel dimension of output. Default `-1` for same with input.
  attn_ratio: value channel dimension expansion.
  use_bn: boolean value if use BN layers for qkv and output.
  qkv_bias: boolean value if use bias for qkv layer.
  out_bias: boolean value if use bias for output layer.
  activation: activation for output, `None` to disable.

Examples:

>>> from keras_cv_attention_models import attention_layers
>>> inputs = keras.layers.Input([14, 12, 192])
>>> nn = attention_layers.mhsa_with_multi_head_position(inputs, output_dim=384, num_heads=4, key_dim=16)
>>> print(f"{nn.shape = }")
# nn.shape = TensorShape([None, 14, 12, 384])

>>> mm = keras.models.Model(inputs, nn)
>>> print({ii.name:ii.numpy().shape for ii in mm.weights})
# {'dense/kernel:0': (192, 192),
#  'batch_normalization/gamma:0': (192,),
#  'batch_normalization/beta:0': (192,),
#  'batch_normalization/moving_mean:0': (192,),
#  'batch_normalization/moving_variance:0': (192,),
#  'multi_head_positional_embedding/positional_embedding:0': (168, 4),
#  'dense_1/kernel:0': (64, 384),
#  'batch_normalization_1/gamma:0': (384,),
#  'batch_normalization_1/beta:0': (384,),
#  'batch_normalization_1/moving_mean:0': (384,),
#  'batch_normalization_1/moving_variance:0': (384,)}
"""

mhsa_with_multi_head_position_and_strides.__doc__ = __head_doc__ + """
Multi Head Self Attention with MultiHeadPositionalEmbedding and enabled strides on `query`.
Using additional `BatchNormalization` for `query / key / value`,
and adding `MultiHeadPositionalEmbedding` to `attention_scores`.
Also with a `strides` parameter which can further reduce calculation.

input: `[batch, height, width, channel]`.
output: `[batch, height, width, channel]`.

Args:
  inputs: Input tensor.
  output_dim: The expected channel dimension of output.
  num_heads: Number of attention heads.
  key_dim: Size of each attention head for query and key. Default `-1` for `key_dim = input_channels // num_heads`.
  output_dim: The expected channel dimension of output. Default `-1` for same with input.
  attn_ratio: value channel dimension expansion.
  strides: query strides on height and width dimension.
  use_bn: boolean value if use BN layers for qkv and output.
  qkv_bias: boolean value if use bias for qkv layer.
  out_bias: boolean value if use bias for output layer.
  activation: activation for output, `None` to disable.

Examples:

>>> from keras_cv_attention_models import attention_layers
>>> inputs = keras.layers.Input([14, 12, 192])
>>> nn = attention_layers.mhsa_with_multi_head_position_and_strides(inputs, output_dim=384, num_heads=4, key_dim=16, strides=2)
>>> print(f"{nn.shape = }")
# nn.shape = TensorShape([None, 7, 6, 384])

>>> mm = keras.models.Model(inputs, nn)
>>> print({ii.name:ii.numpy().shape for ii in mm.weights})
# {'dense_4/kernel:0': (192, 192),
#  'dense_3/kernel:0': (192, 64),
#  'batch_normalization_4/gamma:0': (192,),
#  'batch_normalization_4/beta:0': (192,),
#  'batch_normalization_4/moving_mean:0': (192,),
#  'batch_normalization_4/moving_variance:0': (192,),
#  'batch_normalization_3/gamma:0': (64,),
#  'batch_normalization_3/beta:0': (64,),
#  'batch_normalization_3/moving_mean:0': (64,),
#  'batch_normalization_3/moving_variance:0': (64,),
#  'multi_head_positional_embedding_1/positional_embedding:0': (168, 4),
#  'dense_5/kernel:0': (128, 384),
#  'batch_normalization_5/gamma:0': (384,),
#  'batch_normalization_5/beta:0': (384,),
#  'batch_normalization_5/moving_mean:0': (384,),
#  'batch_normalization_5/moving_variance:0': (384,)}
"""
