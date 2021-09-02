from keras_cv_attention_models.botnet.botnet import BotNet, BotNet50, BotNet101, BotNet152, RelativePositionalEmbedding, mhsa_with_relative_position_embedding


__head_doc__ = """
Keras implementation of [botnet](https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2).
Paper [PDF 2101.11605 Bottleneck Transformers for Visual Recognition](https://arxiv.org/pdf/2101.11605.pdf).
"""

__tail_doc__ = """  strides: strides used in the last stack. It's reported `1` works better than `2`, but slower.
  preact: whether to use pre-activation or not.
  use_bias: whether to use biases for convolutional layers or not.
  input_shape: it should have exactly 3 inputs channels, default `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation used in whole model, default `relu`.
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
      Default is `softmax`.
  pretrained: one of `None` (random initialization) or 'imagenet' (pre-training on ImageNet).
      Will try to download and load pre-trained model weights if not None.
  **kwargs: other parameters if available.

Returns:
    A `keras.Model` instance.
"""

BotNet.__doc__ = __head_doc__ + """
Args:
  num_blocks: number of blocks in each stack.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model        | Params |
  | ------------ | ------ |
  | botnet50     | 21M    |
  | botnet101    | 41M    |
  | botnet152    | 56M    |
"""

BotNet50.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

BotNet101.__doc__ = BotNet50.__doc__
BotNet152.__doc__ = BotNet50.__doc__

mhsa_with_relative_position_embedding.__doc__ = __head_doc__ + """
Multi head self attention with positional embedding. Defined as function, not layer.

Args:
  inputs: input tensor.
  num_heads: Number of attention heads.
  key_dim: Size of each attention head for query and key. Default `0` for `key_dim = inputs.shape[-1] // num_heads`.
  relative: Boolean, whether using relative or absolute positional embedding.
  out_weight: Boolean, whether use an ouput dense.
  out_bias: Boolean, whether the ouput dense layer use bias vectors/matrices.
  out_shape: The expected shape of an output tensor. If not specified, projects back to the input dim.
  attn_dropout: Dropout probability for attention.

Examples:

>>> from keras_cv_attention_models import attention_layers
>>> inputs = keras.layers.Input([14, 16, 256])
>>> nn = attention_layers.mhsa_with_relative_position_embedding(inputs, num_heads=4)
>>> print(f"{nn.shape = }")
nn.shape = TensorShape([None, 14, 16, 256])

>>> mm = keras.models.Model(inputs, nn)
>>> mm.summary()
>>> print({ii.name: ii.shape for ii in mm.weights})
{'dense/kernel:0': TensorShape([256, 768]),
 'relative_positional_embedding/r_height:0': TensorShape([64, 27]),
 'relative_positional_embedding/r_width:0': TensorShape([64, 31]),
 'dense_1/kernel:0': TensorShape([256, 256])}
"""

RelativePositionalEmbedding.__doc__ = __head_doc__ + """
Relative Positional Embedding layer. Supports also absolute positional embedding.

input: `[batch, num_heads, height, width, key_dim]`.
output: `[batch, num_heads, height, width, position_height, position_width]`

Args:
  position_height: positional embedding height. Default `0` for using `input_shape[2]`.
      Should be larger than `input_shape[2]`.
  position_width: positional embedding width. Default `0` for using `input_shape[3]` or `position_height` if set.
      Should be larger than `input_shape[3]`.
  use_absolute_pos: Set `True` to use absolute positional embeddings.

Examples:

>>> from keras_cv_attention_models import attention_layers
>>> aa = attention_layers.RelativePositionalEmbedding()
>>> print(f"{aa(tf.ones([1, 4, 14, 16, 32])).shape = }")
aa(tf.ones([1, 4, 14, 16, 32])).shape = TensorShape([1, 4, 14, 16, 14, 16])

>>> print({ii.name:ii.shape for ii in aa.weights})
{'relative_positional_embedding_6/r_height:0': TensorShape([32, 27]),
 'relative_positional_embedding_6/r_width:0': TensorShape([32, 31])}
"""
