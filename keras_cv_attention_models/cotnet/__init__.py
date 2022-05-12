from keras_cv_attention_models.cotnet.cotnet import CotNet, CotNet50, CotNet101, CotNetSE50D, CotNetSE101D, CotNetSE152D, cot_attention


__head_doc__ = """
Keras implementation of [Github JDAI-CV/CoTNet](https://github.com/JDAI-CV/CoTNet).
Paper [PDF 2107.12292 Contextual Transformer Networks for Visual Recognition](https://arxiv.org/pdf/2107.12292.pdf).
"""

__tail_doc__ = """  bn_after_attn: boolean value if add `batchnorm + activation` layers after attention layer. Default `True`.
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
  **kwargs: other parameters from `AotNet` if not conflict.

Returns:
    A `keras.Model` instance.
"""

CotNet.__doc__ = __head_doc__ + """
Args:
  num_blocks: model structure parameter, number of blocks in each stack.
  strides: model structure parameter, strides value for the first block in each stack.
      `[2, 2, 2, 2]` for `CotNetSExxD` models, `[1, 2, 2, 2]` for others.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model        | Params | FLOPs  | Input | Top1 Acc |
  | ------------ |:------:| ------ | ----- |:--------:|
  | CotNet50     | 22.2M  | 3.25G  | 224   |   81.3   |
  | CotNeXt50    | 30.1M  | 4.3G   | 224   |   82.1   |
  | CotNetSE50D  | 23.1M  | 4.05G  | 224   |   81.6   |
  | CotNet101    | 38.3M  | 6.07G  | 224   |   82.8   |
  | CotNeXt101   | 53.4M  | 8.2G   | 224   |   83.2   |
  | CotNetSE101D | 40.9M  | 8.44G  | 224   |   83.2   |
  | CotNetSE152D | 55.8M  | 12.22G | 224   |   84.0   |
  | CotNetSE152D | 55.8M  | 24.92G | 320   |   84.6   |
"""

CotNet50.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

CotNet101.__doc__ = CotNet50.__doc__
CotNetSE50D.__doc__ = CotNet50.__doc__
CotNetSE101D.__doc__ = CotNet50.__doc__
CotNetSE152D.__doc__ = CotNet50.__doc__

cot_attention.__doc__ = __head_doc__ + """
Contextual transformer. Callable function, NOT defined as a layer.
It's using `GroupNormalization` / grouped `Conv2D` / `extract_patches` and other strategies.

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
