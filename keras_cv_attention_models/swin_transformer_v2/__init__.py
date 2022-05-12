from keras_cv_attention_models.swin_transformer_v2.swin_transformer_v2 import (
    DivideScale,
    PairWiseRelativePositionalEmbedding,
    shifted_window_attention,
    WindowAttentionMask,
    window_mhsa_with_pair_wise_positional_embedding,
    SwinTransformerV2,
    SwinTransformerV2Tiny,
    SwinTransformerV2Tiny_ns,
    SwinTransformerV2Small,
    SwinTransformerV2Small_ns,
    SwinTransformerV2Base,
    SwinTransformerV2Large,
    SwinTransformerV2Giant,
)

__head_doc__ = """
Keras implementation of [Github timm/swin_transformer_v2_cr](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/swin_transformer_v2_cr.py).
Paper [PDF 2111.09883 Swin Transformer V2: Scaling Up Capacity and Resolution](https://arxiv.org/pdf/2111.09883.pdf).
"""

__tail_doc__ = """  window_ratio: window_size ratio, window_size = [input_shape[0] // window_ratio, input_shape[1] // window_ratio].
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
  pretrained: None or one of ["imagenet", "token_label"].

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
""" + __tail_doc__ + """
Model architectures:
  | Model                           | Params | FLOPs   | Input | Top1 Acc |
  | ------------------------------- | ------ | ------- | ----- | -------- |
  | SwinTransformerV2Tiny_ns        | 28.3M  | 4.69G   | 224   | 81.8     |
  | SwinTransformerV2Small          | 49.7M  | 9.12G   | 224   | 83.13    |
  | SwinTransformerV2Small_ns       | 49.7M  | 9.12G   | 224   | 83.5     |
  | SwinTransformerV2Base, 22k      | 87.9M  | 50.89G  | 384   | 87.1     |
  | SwinTransformerV2Large, 22k     | 196.7M | 109.40G | 384   | 87.7     |
  | SwinTransformerV2Giant, 22k+ext | 2.60B  | 4.26T   | 640   | 90.17    |
"""

SwinTransformerV2Tiny.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

SwinTransformerV2Tiny_ns.__doc__ = SwinTransformerV2Tiny.__doc__
SwinTransformerV2Small.__doc__ = SwinTransformerV2Tiny.__doc__
SwinTransformerV2Small_ns.__doc__ = SwinTransformerV2Tiny.__doc__
SwinTransformerV2Base.__doc__ = SwinTransformerV2Tiny.__doc__
SwinTransformerV2Large.__doc__ = SwinTransformerV2Tiny.__doc__
SwinTransformerV2Giant.__doc__ = SwinTransformerV2Tiny.__doc__

DivideScale.__doc__ = __head_doc__ + """
Apply `inputs / tf.maximum(scale, min_value)` on given axis.

Args:
  axis: list or int number, specific axis apply scaling.
  initializer: weight initializer.
  min_value: min value avoiding dividing zero.

Examples:
>>> from keras_cv_attention_models import attention_layers
>>> aa = attention_layers.DivideScale()
>>> print(f"{aa(tf.ones([1, 32, 32, 192])).shape = }")
# aa(tf.ones([1, 32, 32, 192])).shape = TensorShape([1, 32, 32, 192])
>>> print({ii.name:ii.shape for ii in aa.weights})
# {'divide_scale/weight:0': TensorShape([192])}

>>> bb = attention_layers.DivideScale(axis=[1, 2])
>>> print(f"{bb(tf.ones([1, 32, 32, 192])).shape = }")
# bb(tf.ones([1, 32, 32, 192])).shape = TensorShape([1, 32, 32, 192])
>>> print({ii.name:ii.shape for ii in bb.weights})
# {'divide_scale_1/weight:0': TensorShape([1, 32, 32, 1])}
"""

PairWiseRelativePositionalEmbedding.__doc__ = __head_doc__ + """
Pair Wise Relative Positional Embedding layer.
No weight, just need to wrapper a layer, or will not in model structure.
Returns a `log` encoded bias depending on input `[height, width]`.

input: `[batch * window_patch, window_height, window_width, channel]`.
output: relative_coords_log `[window_height * window_width, window_height * window_width, 2]`.

No args.

Examples:
>>> from keras_cv_attention_models import attention_layers
>>> aa = attention_layers.PairWiseRelativePositionalEmbedding()
>>> print(f"{aa(tf.ones([1 * 9, 4, 4, 192])).shape = }")
# aa(tf.ones([1 * 9, 4, 4, 192])).shape = TensorShape([16, 16, 2])
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
  out_bias: Boolean, whether the ouput dense layer use bias vectors/matrices.
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
