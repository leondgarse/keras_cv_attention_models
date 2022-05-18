from keras_cv_attention_models.swin_transformer_v2.swin_transformer_v2 import (
    ExpLogitScale,
    PairWiseRelativePositionalEmbedding,
    shifted_window_attention,
    WindowAttentionMask,
    window_mhsa_with_pair_wise_positional_embedding,
    SwinTransformerV2,
    SwinTransformerV2Tiny_window8,
    SwinTransformerV2Tiny_window16,
    SwinTransformerV2Small_window8,
    SwinTransformerV2Small_window16,
    SwinTransformerV2Base_window8,
    SwinTransformerV2Base_window12,
    SwinTransformerV2Base_window16,
    SwinTransformerV2Base_window24,
    SwinTransformerV2Large_window12,
    SwinTransformerV2Large_window16,
    SwinTransformerV2Large_window24,
)
from keras_cv_attention_models.swin_transformer_v2.swin_transformer_v2_timm import SwinTransformerV2Tiny_ns, SwinTransformerV2Small_ns

__head_doc__ = """
Keras implementation of [Github microsoft/Swin-Transformer](https://github.com/microsoft/Swin-Transformer).
Paper [PDF 2111.09883 Swin Transformer V2: Scaling Up Capacity and Resolution](https://arxiv.org/pdf/2111.09883.pdf).
"""

__tail_doc__ = """  window_size: window_size for all blocks.
  pos_scale: If pretrained weights are from different input_shape or window_size, pos_scale is previous actually using window_size.
  stem_patch_size: stem patch size for stem kernel_size and strides.
  layer_scale: layer scale init value. Default `-1` means not applying, any value `>=0` will add a scale value for each block output.
      [Going deeper with Image Transformers](https://arxiv.org/pdf/2103.17239.pdf).
  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  drop_connect_rate: is used for [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382).
      Can be value like `0.2`, indicates the drop probability linearly changes from `0 --> 0.2` for `top --> bottom` layers.
      A higher value means a higher probability will drop the deep branch.
      or `0` to disable (default).
  dropout: dropout rate if top layers is included.
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
  pretrained: value in {pretrained}.
      Will try to download and load pre-trained model weights if not None.

Returns:
    A `keras.Model` instance.
"""

SwinTransformerV2.__doc__ = __head_doc__ + """
Args:
  num_blocks: number of blocks in each stack.
  num_heads: num heads for each stack.
  embed_dim: basic hidden dims, expand * 2 for each stack.
  use_stack_norm: boolean value if apply an additional layer_norm after each stack.
  model_name: string, model name.
""" + __tail_doc__.format(pretrained=[None, "imagenet", "imagenet22k", "imagenet21k"]) + """
Model architectures:
  | Model                                | Params | FLOPs  | Input | Top1 Acc |
  | ------------------------------------ | ------ | ------ | ----- | -------- |
  | SwinTransformerV2Tiny_ns             | 28.3M  | 4.69G  | 224   | 81.8     |
  | SwinTransformerV2Small_ns            | 49.7M  | 9.12G  | 224   | 83.5     |
  |                                      |        |        |       |          |
  | SwinTransformerV2Tiny_window8        | 28.3M  | 5.99G  | 256   | 81.8     |
  | SwinTransformerV2Tiny_window16       | 28.3M  | 6.75G  | 256   | 82.8     |
  | SwinTransformerV2Small_window8       | 49.7M  | 11.63G | 256   | 83.7     |
  | SwinTransformerV2Small_window16      | 49.7M  | 12.93G | 256   | 84.1     |
  | SwinTransformerV2Base_window8        | 87.9M  | 20.44G | 256   | 84.2     |
  | SwinTransformerV2Base_window16       | 87.9M  | 22.17G | 256   | 84.6     |
  | SwinTransformerV2Base_window16, 22k  | 87.9M  | 22.17G | 256   | 86.2     |
  | SwinTransformerV2Base_window24, 22k  | 87.9M  | 55.89G | 384   | 87.1     |
  | SwinTransformerV2Large_window16, 22k | 196.7M | 48.03G | 256   | 86.9     |
  | SwinTransformerV2Large_window24, 22k | 196.7M | 117.1G | 384   | 87.6     |
"""

__default_doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

SwinTransformerV2Tiny_window8.__doc__ = __default_doc__.format(pretrained=[None, "imagenet"])
SwinTransformerV2Tiny_window16.__doc__ = __default_doc__.format(pretrained=[None, "imagenet"])
SwinTransformerV2Small_window8.__doc__ = __default_doc__.format(pretrained=[None, "imagenet"])
SwinTransformerV2Small_window16.__doc__ = __default_doc__.format(pretrained=[None, "imagenet"])
SwinTransformerV2Base_window8.__doc__ = __default_doc__.format(pretrained=[None, "imagenet"])
SwinTransformerV2Base_window12.__doc__ = __default_doc__.format(pretrained=[None, "imagenet21k"])
SwinTransformerV2Base_window16.__doc__ = __default_doc__.format(pretrained=[None, "imagenet", "imagenet22k"])
SwinTransformerV2Base_window24.__doc__ = __default_doc__.format(pretrained=[None, "imagenet", "imagenet22k"])
SwinTransformerV2Large_window12.__doc__ = __default_doc__.format(pretrained=[None, "imagenet21k"])
SwinTransformerV2Large_window16.__doc__ = __default_doc__.format(pretrained=[None, "imagenet22k"])
SwinTransformerV2Large_window24.__doc__ = __default_doc__.format(pretrained=[None, "imagenet22k"])
SwinTransformerV2Tiny_ns.__doc__ = __default_doc__.format(pretrained=[None, "imagenet"])
SwinTransformerV2Small_ns.__doc__ = __default_doc__.format(pretrained=[None, "imagenet"])

ExpLogitScale.__doc__ = __head_doc__ + """
Apply `inputs / tf.maximum(scale, min_value)` on given axis.

Args:
  axis: list or int number, specific axis apply scaling.
  init_value: weight init value. Actual using is `tf.math.log(init_value)`.
  max_value: limit scaled max value.

Examples:
>>> from keras_cv_attention_models import attention_layers
>>> aa = attention_layers.ExpLogitScale()
>>> print(f"{aa(tf.ones([1, 32, 32, 192])).shape = }")
# aa(tf.ones([1, 32, 32, 192])).shape = TensorShape([1, 32, 32, 192])
>>> print({ii.name:ii.shape for ii in aa.weights})
# {'divide_scale/weight:0': TensorShape([192])}

>>> bb = attention_layers.ExpLogitScale(axis=[1, 2])
>>> print(f"{bb(tf.ones([1, 32, 32, 192])).shape = }")
# bb(tf.ones([1, 32, 32, 192])).shape = TensorShape([1, 32, 32, 192])
>>> print({ii.name:ii.shape for ii in bb.weights})
# {'divide_scale_1/weight:0': TensorShape([1, 32, 32, 1])}
"""

PairWiseRelativePositionalEmbedding.__doc__ = __head_doc__ + """
Pair Wise Relative Positional Embedding layer.
No weight, just need to wrapper a layer, or will not in model structure.

input: `[batch * window_patch, window_height, window_width, channel]`.
output:
    relative_log_coords `[(2 * window_height - 1) * (2 * window_width - 1), 2]`.
    relative_position_index `[window_height * window_width, window_height * window_width]`

Args:
  pos_scale: If pretrained weights are from different input_shape or window_size, pos_scale is previous actually using window_size.
      Default -1 for using `[height, width]` from input_shape.

Examples:
>>> from keras_cv_attention_models import attention_layers
>>> aa = attention_layers.PairWiseRelativePositionalEmbedding()
>>> relative_log_coords, relative_position_index = aa(tf.ones([1 * 9, 4, 4, 192]))
>>> print(f"{relative_log_coords.shape = }, {relative_position_index.shape = }")
# relative_log_coords.shape = TensorShape([49, 2]), relative_position_index.shape = TensorShape([16, 16])
>>> print(f"{tf.gather(relative_log_coords, relative_position_index).shape = }")
# tf.gather(relative_log_coords, relative_position_index).shape = TensorShape([16, 16, 2])
"""

WindowAttentionMask.__doc__ = __head_doc__ + """
Window Attention Mask layer.
No weight, just need to wrapper a layer, or will meet some error in model saving or loading.

query_blocks = `window_height * window_width`, blocks = `(height // window_height) * (width // window_width)`
input: `[batch_size * blocks, num_heads, query_blocks, query_blocks]`.
attn_mask: `[1, blocks, 1, query_blocks, query_blocks]`.
output: `[batch_size * blocks, num_heads, query_blocks, query_blocks]` + attn_mask.

attn_mask is assignd value with split blocks:
  hh_split = `[0, height - window_height, height - shift_height, height]`
  ww_split = `[0, width - window_width, width - shift_width, width]`

Args:
  height: mask height.
  width: mask width.
  window_height: mask window_height.
  window_width: mask window_width.
  shift_height: mask shift_height, should be smaller than window_height.
  shift_width: mask shift_width, should be smaller than window_width.

Examples:
>>> from keras_cv_attention_models import attention_layers
>>> height, width, window_height, window_width, shift_height, shift_width = 36, 48, 6, 6, 3, 3
>>> num_heads, query_blocks = 8, window_height * window_width
>>> aa = attention_layers.WindowAttentionMask(height, width, window_height, window_width, shift_height, shift_width)
>>> inputs = tf.ones([1 * (height // window_height) * (width // window_width), num_heads, query_blocks, query_blocks])
>>> print(f"{inputs.shape = }, {aa(inputs).shape = }")
# inputs.shape = TensorShape([48, 8, 36, 36]), aa(inputs).shape = TensorShape([48, 8, 36, 36])
>>> print(f"{aa.attn_mask.shape = }")
# aa.attn_mask.shape = TensorShape([1, 48, 1, 36, 36])
"""

window_mhsa_with_pair_wise_positional_embedding.__doc__ = __head_doc__ + """
Multi head self attention block with PairWiseRelativePositionalEmbedding, also supports mask.
Generating `attention_scores` by calculating cosine similarity between `query` and `key`,
and applying `PairWiseRelativePositionalEmbedding`.

input: `[batch * patch_height * patch_width, window_height, window_width, input_channel]`.
output: `[batch * patch_height * patch_width, window_height, window_width, num_heads * key_dim]`.

Args:
  inputs: input tensor.
  num_heads: Number of attention heads.
  key_dim: Size of each attention head for query and key. Default `0` for `key_dim = inputs.shape[-1] // num_heads`.
  out_shape: The expected shape of an output tensor. If not specified, projects back to the input dim.
  meta_hidden_dim: hidden channel for `mlp_block` applied to `PairWiseRelativePositionalEmbedding` output.
  mask: `WindowAttentionMask` layer instance. Default `None` for not applying mask.
  pos_scale: If pretrained weights are from different input_shape or window_size, pos_scale is previous actually using window_size.
      Default -1 for using `[height, width]` from input_shape.
  out_bias: Boolean, whether the ouput dense layer use bias vectors/matrices.
  qv_bias: Boolean, if use bias for `query` and `value`.
  attn_dropout: Dropout probability for attention scores.
  out_dropout: Dropout probability for attention output.

Examples:

>>> from keras_cv_attention_models import attention_layers
>>> inputs = keras.layers.Input([14, 16, 256])
>>> nn = tf.reshape(inputs, [-1, 7, 2, 8, 2, 256])
>>> nn = tf.transpose(nn, [0, 2, 4, 1, 3, 5])
>>> nn = tf.reshape(nn, [-1, 7, 8, 256])
>>> nn = attention_layers.window_mhsa_with_pair_wise_positional_embedding(nn, num_heads=4, name="attn_")
>>> print(f"{nn.shape = }")
# nn.shape = TensorShape([None, 7, 8, 256])

>>> mm = keras.models.Model(inputs, nn)
>>> mm.summary()
>>> print({ii.name: ii.shape for ii in mm.weights})
# {'attn_qkv/kernel:0': TensorShape([256, 768]),
#  'attn_qkv/bias:0': TensorShape([768]),
#  'attn_meta_Dense_0/kernel:0': TensorShape([2, 384]),
#  'attn_meta_Dense_0/bias:0': TensorShape([384]),
#  'attn_meta_Dense_1/kernel:0': TensorShape([384, 4]),
#  'attn_meta_Dense_1/bias:0': TensorShape([4]),
#  'attn_scale/weight:0': TensorShape([1, 4, 1, 1]),
#  'attn_output/kernel:0': TensorShape([256, 256]),
#  'attn_output/bias:0': TensorShape([256])}
"""

shifted_window_attention.__doc__ = __head_doc__ + """
Window multi head self attention. Defined as function, not layer.
`window_mhsa_with_pair_wise_positional_embedding` with `window_partition` process ahead and `window_reverse` process after.
Also supports window shift.

Args:
  inputs: input tensor.
  window_size: window partition size.
  num_heads: Number of attention heads.
  shift_size: window shift retio in `(0, 1)`.
  pos_scale: If pretrained weights are from different input_shape or window_size, pos_scale is previous actually using window_size.
      Default -1 for using `[height, width]` from input_shape.

Examples:

>>> from keras_cv_attention_models import attention_layers
>>> inputs = keras.layers.Input([14, 16, 256])
>>> nn = attention_layers.shifted_window_attention(inputs, 7, num_heads=4, shift_size=0.5, name="attn_")
>>> print(f"{nn.shape = }")
# nn.shape = TensorShape([None, 14, 16, 256])

>>> mm = keras.models.Model(inputs, nn)
>>> mm.summary()
>>> print({ii.name: ii.shape for ii in mm.weights})
# {'attn_qkv/kernel:0': TensorShape([256, 768]),
#  'attn_qkv/bias:0': TensorShape([768]),
#  'attn_meta_Dense_0/kernel:0': TensorShape([2, 384]),
#  'attn_meta_Dense_0/bias:0': TensorShape([384]),
#  'attn_meta_Dense_1/kernel:0': TensorShape([384, 4]),
#  'attn_meta_Dense_1/bias:0': TensorShape([4]),
#  'attn_scale/weight:0': TensorShape([1, 4, 1, 1]),
#  'attn_output/kernel:0': TensorShape([256, 256]),
#  'attn_output/bias:0': TensorShape([256])}
"""
