from keras_cv_attention_models.levit.levit import (
    LeViT,
    LeViT128S,
    LeViT128,
    LeViT192,
    LeViT256,
    LeViT384,
    MultiHeadPositionalEmbedding,
    mhsa_with_multi_head_position_and_strides,
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
  | Model     | Params | Image resolution | Top1 Acc |
  | --------- | ------ | ---------------- | -------- |
  | LeViT128S | 7.8M   | 224              | 76.6     |
  | LeViT128  | 9.2M   | 224              | 78.6     |
  | LeViT192  | 11M    | 224              | 80.0     |
  | LeViT256  | 19M    | 224              | 81.6     |
  | LeViT384  | 39M    | 224              | 82.6     |
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

input: `[batch, num_heads, query_blocks, key_blocks]`.
output: `[batch, num_heads, query_blocks, key_blocks] + positional_bias`.
conditions: query_height == query_width, key_height == key_width, key_height >= query_height

Examples:

>>> from keras_cv_attention_models import attention_layers
>>> aa = attention_layers.MultiHeadPositionalEmbedding()
>>> print(f"{aa(tf.ones([1, 8, 16, 49])).shape = }")
# aa(tf.ones([1, 8, 16, 49])).shape = TensorShape([1, 8, 16, 49])

>>> print({ii.name:ii.numpy().shape for ii in aa.weights})
# {'multi_head_positional_embedding/positional_embedding:0': (49, 8)}

>>> plt.imshow(aa.bb_pos)
"""

mhsa_with_multi_head_position_and_strides.__doc__ = __head_doc__ + """
Multi Head Self Attention with MultiHeadPositionalEmbedding and enabled strides on `query`.

input: `[batch, height * width, channel]`.
output: `[batch, height * width, channel]`.

Args:
  inputs: Input tensor.
  output_dim: The expected channel dimension of output.
  num_heads: Number of attention heads.
  key_dim: Size of each attention head for query and key.
  attn_ratio: value channel dimension expansion.
  strides: query strides on height and width dimension.
  activation: activation for output, `None` to disable.

Examples:

>>> from keras_cv_attention_models import attention_layers
>>> inputs = keras.layers.Input([14 * 14, 192])
>>> nn = attention_layers.mhsa_with_multi_head_position_and_strides(inputs, output_dim=384, num_heads=4, key_dim=16, strides=2)
>>> print(f"{nn.shape = }")
# nn.shape = TensorShape([None, 49, 384])

>>> mm = keras.models.Model(inputs, nn)
>>> print({ii.name:ii.numpy().shape for ii in mm.weights})
{'kv/kernel:0': (192, 192),
 'q/kernel:0': (192, 64),
 'kv_bn/gamma:0': (192,),
 'kv_bn/beta:0': (192,),
 'kv_bn/moving_mean:0': (192,),
 'kv_bn/moving_variance:0': (192,),
 'q_bn/gamma:0': (64,), 'q_bn/beta:0': (64,),
 'q_bn/moving_mean:0': (64,),
 'q_bn/moving_variance:0': (64,),
 'attn_pos/positional_embedding:0': (196, 4),
 'out/kernel:0': (128, 384),
 'out_bn/gamma:0': (384,),
 'out_bn/beta:0': (384,),
 'out_bn/moving_mean:0': (384,),
 'out_bn/moving_variance:0': (384,)}
"""
