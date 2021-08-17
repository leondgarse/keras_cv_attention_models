from keras_cv_attention_models.cotnet.cotnet import CotNet, CotNet50, CotNet101, SECotNetD50, SECotNetD101, SECotNetD152, cot_attention


__head_doc__ = """
Keras implementation of [Github JDAI-CV/CoTNet](https://github.com/JDAI-CV/CoTNet).
Paper [PDF 2107.12292 Contextual Transformer Networks for Visual Recognition](https://arxiv.org/pdf/2107.12292.pdf).
"""

__tail_doc__ = """  expansion: model structure parameter, channel ouput expansion for each block.
  cardinality: not used.
  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation used in whole model, default `relu`.
  drop_connect_rate: is used for [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382).
      Can be value like `0.2`, indicates the drop probability linearly changes from `0 --> 0.2` for `top --> bottom` layers.
      A higher value means a higher probability will drop the deep branch.
      or `0` to disable (default).
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
      Default is `softmax`.
  pretrained: one of `None` (random initialization) or 'imagenet' (pre-training on ImageNet).
  **kwargs: other parameters if available.

Returns:
    A `keras.Model` instance.
"""

CotNet.__doc__ = __head_doc__ + """
Args:
  num_blocks: model structure parameter, number of blocks in each stack.
  stem_width: model structure parameter, output dimension after stem block.
  deep_stem: model structure parameter, boolean value if use deep conv in stem block.
  attn_types: model structure parameter, a list of `"sa"` or `"cot"` or `None`, indicates attnetion type for each stack or block.
      For `"sa"` means `split_attention_conv2d` from `resnest`.
      For `"cot"` means `cot_attention` from this `CotNet` architecture.
      For `None` means `Conv2D` layer.
  strides: model structure parameter, strides value for the first block in each stack.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model          | Params | Image resolution | FLOPs | Top1 Acc |
  | -------------- |:------:| ---------------- | ----- |:--------:|
  | CoTNet-50      | 22.2M  | 224              | 3.3   |   81.3   |
  | CoTNeXt-50     | 30.1M  | 224              | 4.3   |   82.1   |
  | SE-CoTNetD-50  | 23.1M  | 224              | 4.1   |   81.6   |
  | CoTNet-101     | 38.3M  | 224              | 6.1   |   82.8   |
  | CoTNeXt-101    | 53.4M  | 224              | 8.2   |   83.2   |
  | SE-CoTNetD-101 | 40.9M  | 224              | 8.5   |   83.2   |
  | SE-CoTNetD-152 | 55.8M  | 224              | 17.0  |   84.0   |
  | SE-CoTNetD-152 | 55.8M  | 320              | 26.5  |   84.6   |
"""

CotNet50.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

CotNet101.__doc__ = CotNet50.__doc__
SECotNetD50.__doc__ = CotNet50.__doc__
SECotNetD101.__doc__ = CotNet50.__doc__
SECotNetD152.__doc__ = CotNet50.__doc__

cot_attention.__doc__ = __head_doc__ + """
Contextual transformer. Callable function, NOT defined as a layer.

Args:
  inputs: input tensor.
  kernel_size: kernel size in this full process.
  activation: activation used after `BatchNormalization`.

Examples:

>>> from keras_cv_attention_models import attention_layers
>>> inputs = keras.layers.Input([28, 28, 192])
>>> nn = attention_layers.cot_attention(inputs, kernel_size=3)
>>> dd = keras.models.Model(inputs, nn)
>>> dd.summary()
>>> dd.output_shape
(None, 28, 28, 192)

>>> {ii.name: ii.shape for ii in dd.weights}
{'key_conv/kernel:0': TensorShape([3, 3, 48, 192]),
 'key_bn/gamma:0': TensorShape([192]),
 'key_bn/beta:0': TensorShape([192]),
 'key_bn/moving_mean:0': TensorShape([192]),
 'key_bn/moving_variance:0': TensorShape([192]),
 'embed_ww_1_conv/kernel:0': TensorShape([1, 1, 384, 96]),
 'embed_ww_1_bn/gamma:0': TensorShape([96]),
 'embed_ww_1_bn/beta:0': TensorShape([96]),
 'embed_ww_1_bn/moving_mean:0': TensorShape([96]),
 'embed_ww_1_bn/moving_variance:0': TensorShape([96]),
 'embed_1_conv/kernel:0': TensorShape([1, 1, 192, 192]),
 'embed_ww_2_conv/kernel:0': TensorShape([1, 1, 96, 216]),
 'embed_ww_2_conv/bias:0': TensorShape([216]),
 'embed_1_bn/gamma:0': TensorShape([192]),
 'embed_1_bn/beta:0': TensorShape([192]),
 'embed_1_bn/moving_mean:0': TensorShape([192]),
 'embed_1_bn/moving_variance:0': TensorShape([192]),
 'embed_ww_group_norm/gamma:0': TensorShape([216]),
 'embed_ww_group_norm/beta:0': TensorShape([216]),
 'embed_2_bn/gamma:0': TensorShape([192]),
 'embed_2_bn/beta:0': TensorShape([192]),
 'embed_2_bn/moving_mean:0': TensorShape([192]),
 'embed_2_bn/moving_variance:0': TensorShape([192]),
 'attn_se_1_conv/kernel:0': TensorShape([1, 1, 192, 96]),
 'attn_se_1_conv/bias:0': TensorShape([96]),
 'attn_se_bn/gamma:0': TensorShape([96]),
 'attn_se_bn/beta:0': TensorShape([96]),
 'attn_se_bn/moving_mean:0': TensorShape([96]),
 'attn_se_bn/moving_variance:0': TensorShape([96]),
 'attn_se_2_conv/kernel:0': TensorShape([1, 1, 96, 384]),
 'attn_se_2_conv/bias:0': TensorShape([384])}
"""
