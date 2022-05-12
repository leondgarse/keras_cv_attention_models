from keras_cv_attention_models.nat.nat import (
    NAT,
    NAT_Mini,
    NAT_Tiny,
    NAT_Small,
    NAT_Base,
    MultiHeadRelativePositionalKernelBias,
    neighborhood_attention,
)

__head_doc__ = """
Keras implementation of [NAT](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer).
Paper [PDF 2204.07143 Neighborhood Attention Transformer](https://arxiv.org/pdf/2204.07143.pdf).
"""

__tail_doc__ = """  stem_width: output dimension for stem block. Default -1 means using `out_channels[0]`
  attn_kernel_size: kernel_size for `neighborhood_attention` block, defualt 7.
  mlp_ratio: channel expansion ratio for mlp hidden layers, default 3 for NAT_Mini and NAT_Tiny, 2 for NAT_Small and NAT_Base.
  layer_scale: layer scale init value. `-1` means not applying, any value `>=0` will add a scale value for each block output.
      [Going deeper with Image Transformers](https://arxiv.org/pdf/2103.17239.pdf).
      Default -1 for NAT_Mini and NAT_Tiny, 1e-5 for NAT_Small and NAT_Base
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

NAT.__doc__ = __head_doc__ + """
Args:
  num_blocks: number of blocks in each stack.
  out_channels: output channels for each stack.
  num_heads: num heads for each stack.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model     | Params | FLOPs  | Input | Top1 Acc |
  | --------- | ------ | ------ | ----- | -------- |
  | NAT_Mini  | 20.0M  | 2.73G  | 224   | 81.8     |
  | NAT_Tiny  | 27.9M  | 4.34G  | 224   | 83.2     |
  | NAT_Small | 50.7M  | 7.84G  | 224   | 83.7     |
  | NAT_Base  | 89.8M  | 13.76G | 224   | 84.3     |
"""

NAT_Mini.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

NAT_Tiny.__doc__ = NAT_Mini.__doc__
NAT_Small.__doc__ = NAT_Mini.__doc__
NAT_Base.__doc__ = NAT_Mini.__doc__

MultiHeadRelativePositionalKernelBias.__doc__ = __head_doc__ + """
Multi Head Relative Positional Kernel Bias layer. Weights depends on `num_heads` and `kernel_size` not on `input_shape`.

input (is_heads_first=False): `[batch, height * width, num_heads, ..., size * size]`
input (is_heads_first=True): `[batch, num_heads, height * width, ..., size * size]`
positional_bias: `[num_heads, (2 * size - 1) * (2 * size - 1)]`
output: `input + gathered positional_bias`.
condition: height >= size, width >= size

Args:
  input_height: specify `height` for `input` if not square, default -1 assums `input_height == input_width`.
  is_heads_first: boolean value if input is `[batch, num_heads, height * width]` or `[batch, height * width, num_heads]`.

Examples:

# Basic
>>> from keras_cv_attention_models import attention_layers
>>> aa = attention_layers.MultiHeadRelativePositionalKernelBias()
>>> print(f"{aa(tf.ones([1, 29 * 29, 8, 5 * 5])).shape = }")
# aa(tf.ones([1, 29 * 29, 8, 5 * 5])).shape = TensorShape([1, 841, 8, 25])
>>> print(f'{aa.pos_bias.shape = }, {aa.bias_coords.shape = }')
# aa.pos_bias.shape = TensorShape([8, 81]), aa.bias_coords.shape = TensorShape([841, 25])

# Specific input_height and a longer input shape
>>> aa = attention_layers.MultiHeadRelativePositionalKernelBias(input_height=7)
>>> print(f"{aa(tf.ones([1, 7 * 13, 8, 23, 46, 4, 3 * 3])).shape = }")
# aa(tf.ones([1, 7 * 13, 8, 23, 46, 4, 3 * 3])).shape = TensorShape([1, 91, 8, 23, 46, 4, 9])
>>> print(f'{aa.pos_bias.shape = }, {aa.bias_coords.shape = }')
# aa.pos_bias.shape = TensorShape([8, 25]), aa.bias_coords.shape = TensorShape([91, 1, 1, 1, 9])

# Specific is_heads_first=True
>>> aa = attention_layers.MultiHeadRelativePositionalKernelBias(input_height=19, is_heads_first=True)
>>> print(f"{aa(tf.ones([1, 8, 19 * 29, 5 * 5])).shape = }")
# aa(tf.ones([1, 8, 19 * 29, 5 * 5])).shape = TensorShape([1, 8, 551, 25])
>>> print(f'{aa.pos_bias.shape = }, {aa.bias_coords.shape = }')
# aa.pos_bias.shape = TensorShape([8, 81]), aa.bias_coords.shape = TensorShape([551, 25])
"""

neighborhood_attention.__doc__ = __head_doc__ + """
Neighborhood Attention block, not a layer.
Extract patches with a `kernel_size` from `key_value` as an enlarged attention area. Balancing global and local attention.
Also adds `MultiHeadRelativePositionalKernelBias` with `attention_scores`.

Args:
  inputs: input tensor.
  kernel_size: extracting patch kernel_size for key and value.
  num_heads: Number of attention heads.
  key_dim: Size of each attention head for query and key. Default `0` for `key_dim = inputs.shape[-1] // num_heads`.
  out_weight: Boolean, whether use an ouput dense.
  qkv_bias: Boolean, whether the qkv dense layer use bias vectors/matrices.
  out_bias: Boolean, whether the ouput dense layer use bias vectors/matrices.
  attn_dropout: Dropout probability for attention scores.
  output_dropout: Dropout probability for attention output.

Examples:

>>> from keras_cv_attention_models import attention_layers
>>> inputs = keras.layers.Input([14, 16, 256])
>>> nn = attention_layers.neighborhood_attention(inputs, num_heads=4)
>>> print(f"{nn.shape = }")
# nn.shape = TensorShape([None, 14, 16, 512])

>>> mm = keras.models.Model(inputs, nn)
>>> mm.summary()
>>> print({ii.name: ii.shape for ii in mm.weights})
# {'dense/kernel:0': TensorShape([256, 768]),
#  'dense/bias:0': TensorShape([768]),
#  'multi_head_relative_positional_kernel_bias_11/positional_embedding:0': TensorShape([4, 169]),
#  'dense_1/kernel:0': TensorShape([256, 256]),
#  'dense_1/bias:0': TensorShape([256])}
"""
