from keras_cv_attention_models.coat.coat import CoaT, CoaTLiteTiny, CoaTLiteMini, CoaTLiteSmall, CoaTTiny, CoaTMini, ConvPositionalEncoding, ConvRelativePositionalEncoding, ClassToken

__head_doc__ = """
Keras implementation of [Github mlpc-ucsd/CoaT](https://github.com/mlpc-ucsd/CoaT).
Paper [PDF 2104.06399 CoaT: Co-Scale Conv-Attentional Image Transformers](http://arxiv.org/abs/2104.06399).
"""

__tail_doc__ = """  head_splits: split head list for `ConvRelativePositionalEncoding`.
      Should be sum eqauls `num_heads`. Default `[2, 3, 3]` indicates split `8` head into 3 groups.
  head_kernel_size: kernel_size for each split head in `ConvRelativePositionalEncoding`. Defualt `[3, 5, 7]`.
  use_shared_cpe: set `False` to disable using shared `ConvPositionalEncoding` blocks.
  use_shared_crpe: set `False` to disable using shared `ConvRelativePositionalEncoding` blocks.
  out_features: None or a list of number in `[0, 1, 2, 3]` for output of relative `serial block` output.
  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation used in whole model, default `gelu`.
  drop_connect_rate: Not using.
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
  pretrained: one of `None` (random initialization) or 'imagenet' (pre-training on ImageNet).
      Will try to download and load pre-trained model weights if not None.
  **kwargs: other parameters if available.

Returns:
    A `keras.Model` instance.
"""

CoaT.__doc__ = __head_doc__ + """
Args:
  serial_depths: number of serial blocks.
  embed_dims: output channel for each serial stack.
  mlp_ratios: dimension expansion ration for `mlp_block` in each `serial` and `parallel` stack.
  parallel_depth: number of parallel blocks. `0` for tiny models, `6` for none tiny models.
  patch_size: patch size extracted from input for all `patch_embed` blocks.
  num_heads: heads number for transformer block.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model         | Params | Top1 Acc |
  | ------------- | ------ | -------- |
  | CoaTLiteTiny  | 5.7M   | 77.5     |
  | CoaTLiteMini  | 11M    | 79.1     |
  | CoaTLiteSmall | 20M    | 81.9     |
  | CoaTTiny      | 5.5M   | 78.3     |
  | CoaTMini      | 10M    | 81.0     |
"""

__default_doc__ = __head_doc__ + """
[{model_name} architecture] serial_depths: {serial_depths}, embed_dims: {embed_dims}, mlp_ratios: {mlp_ratios},
                            parallel_depth: {parallel_depth}, patch_size: {patch_size}, num_heads: {num_heads}.
Args:
""" + __tail_doc__

CoaTLiteTiny.__doc__ = __default_doc__.format(model_name="CoaTLiteTiny", **coat.BLOCK_CONFIGS["lite_tiny"])
CoaTLiteMini.__doc__ = __default_doc__.format(model_name="CoaTLiteMini", **coat.BLOCK_CONFIGS["lite_mini"])
CoaTLiteSmall.__doc__ = __default_doc__.format(model_name="CoaTLiteSmall", **coat.BLOCK_CONFIGS["lite_small"])
CoaTTiny.__doc__ = __default_doc__.format(model_name="CoaTTiny", **coat.BLOCK_CONFIGS["tiny"])
CoaTMini.__doc__ = __default_doc__.format(model_name="CoaTMini", **coat.BLOCK_CONFIGS["mini"])

ConvPositionalEncoding.__doc__ = __head_doc__ + """
Convolutional Position Encoding. Note: This module is similar to the conditional position encoding in CPVT.

input: `[batch, class_token + height * width, channel]`.
output: `[batch, class_token + height * width, channel]`.

Args:
  kernel_size: `DepthwiseConv2D` kernel size.

Examples:
>>> from keras_cv_attention_models import attention_layers
>>> aa = attention_layers.ConvPositionalEncoding()
>>> print(f"{aa(tf.ones([1, 1 + 14 * 14, 256])).shape = }")
aa(tf.ones([1, 1 + 14 * 14, 256])).shape = TensorShape([1, 197, 256])

>>> print({ii.name:ii.numpy().shape for ii in aa.weights})
{'conv_positional_encoding/depthwise_kernel:0': (3, 3, 256, 1), 'conv_positional_encoding/bias:0': (256,)}
"""

ConvRelativePositionalEncoding.__doc__ = __head_doc__ + """
Convolutional with Relative Position Encoding.

input:
    query: `[batch, num_heads, class_token + height * width, channel // num_heads]`.
    value: `[batch, num_heads, class_token + height * width, channel // num_heads]`.
output: `[batch, num_heads, 1 * zero + height * width, channel // num_heads]`.

Args:
  head_splits: split head list. Should be sum eqauls `num_heads`. Default `[2, 3, 3]` indicates split `8` head into 3 groups.
  head_kernel_size: kernel_size for each split head. Defualt `[3, 5, 7]`.

Examples:
>>> from keras_cv_attention_models import attention_layers
>>> aa = attention_layers.ConvRelativePositionalEncoding()
>>> print(f"{aa(tf.ones([1, 8, 1 + 14 * 14, 6]), tf.ones([1, 8, 1 + 14 * 14, 6])).shape = }")
aa(tf.ones([1, 8, 1 + 14 * 14, 6]), tf.ones([1, 8, 1 + 14 * 14, 6])).shape = TensorShape([1, 8, 197, 6])

>>> print({ii.name:ii.numpy().shape for ii in aa.weights})
{'conv_relative_positional_encoding/depth_conv_1/depthwise_kernel:0': (3, 3, 12, 1),
 'conv_relative_positional_encoding/depth_conv_1/bias:0': (12,),
 'conv_relative_positional_encoding/depth_conv_2/depthwise_kernel:0': (5, 5, 18, 1),
 'conv_relative_positional_encoding/depth_conv_2/bias:0': (18,),
 'conv_relative_positional_encoding/depth_conv_3/depthwise_kernel:0': (7, 7, 18, 1),
 'conv_relative_positional_encoding/depth_conv_3/bias:0': (18,)}
"""
