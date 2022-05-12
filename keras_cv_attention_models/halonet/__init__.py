from keras_cv_attention_models.halonet.halonet import (
    halo_attention,
    HaloNet,
    HaloNetH0,
    HaloNetH1,
    HaloNetH2,
    HaloNetH3,
    HaloNetH4,
    HaloNetH5,
    HaloNetH6,
    HaloNetH7,
    HaloNet26T,
    HaloNet50T,
    HaloNetSE33T,
    HaloNextECA26T,
    HaloRegNetZB,
    HaloBotNet50T,
)

__head_doc__ = """
Keras implementation of [Github lucidrains/halonet-pytorch](https://github.com/lucidrains/halonet-pytorch).
Paper [PDF 2103.12731 Scaling Local Self-Attention for Parameter Efficient Visual Backbones](https://arxiv.org/pdf/2103.12731.pdf).
"""

__tail_doc__ = """  out_channels: output channels for each stack, default `[64, 128, 256, 512]`.
  strides: strides for each stack, default `[1, 2, 2, 2]`.
  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation used in whole model.
  drop_connect_rate: is used for [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382).
      Can be value like `0.2`, indicates the drop probability linearly changes from `0 --> 0.2` for `top --> bottom` layers.
      A higher value means a higher probability will drop the deep branch.
      or `0` to disable (default).
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
      Default is `softmax`.
  pretrained: one of `None` (random initialization) or 'imagenet' (pre-training on ImageNet).
      Will try to download and load pre-trained model weights if not None.
      No pretrained available for `H` models.
  **kwargs: other parameters from `AotNet` if not conflict.

Returns:
    A `keras.Model` instance.
"""

HaloNet.__doc__ = __head_doc__ + """
Args:
  num_blocks: number of blocks in each stack.
  halo_block_size: block_size for halo attention layers.
  halo_halo_size: halo_size for halo attention layers.
  halo_expansion: filter expansion for halo attention output channel.
  num_heads: number of heads in each stack. Can be a number or list of number like `[4, 8, 8, 8]`.
  expansion: filter expansion for each block output channel.
  output_num_features: none `0` value to add another `conv2d + bn + activation` layers before `GlobalAveragePooling2D`.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model          | Params | FLOPs   | Input | Top1 Acc |
  | -------------- | ------ | ------- | ----- | -------- |
  | HaloNetH0      | 5.5M   | 2.40G   | 256   | 77.9     |
  | HaloNetH1      | 8.1M   | 3.04G   | 256   | 79.9     |
  | HaloNetH2      | 9.4M   | 3.37G   | 256   | 80.4     |
  | HaloNetH3      | 11.8M  | 6.30G   | 320   | 81.9     |
  | HaloNetH4      | 19.1M  | 12.17G  | 384   | 83.3     |
  | - 21k          | 19.1M  | 12.17G  | 384   | 85.5     |
  | HaloNetH5      | 30.7M  | 32.61G  | 448   | 84.0     |
  | HaloNetH6      | 43.4M  | 53.20G  | 512   | 84.4     |
  | HaloNetH7      | 67.4M  | 119.64G | 600   | 84.9     |
  | HaloNextECA26T | 10.7M  | 2.43G   | 256   | 79.50    |
  | HaloNet26T     | 12.5M  | 3.18G   | 256   | 79.13    |
  | HaloNetSE33T   | 13.7M  | 3.55G   | 256   | 80.99    |
  | HaloRegNetZB   | 11.68M | 1.97G   | 224   | 81.042   |
  | HaloNet50T     | 22.7M  | 5.29G   | 256   | 81.70    |
  | HaloBotNet50T  | 22.6M  | 5.02G   | 256   | 82.0     |

"""

HaloNetH0.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

HaloNetH1.__doc__ = HaloNetH0.__doc__
HaloNetH2.__doc__ = HaloNetH0.__doc__
HaloNetH3.__doc__ = HaloNetH0.__doc__
HaloNetH4.__doc__ = HaloNetH0.__doc__
HaloNetH5.__doc__ = HaloNetH0.__doc__
HaloNetH6.__doc__ = HaloNetH0.__doc__
HaloNetH7.__doc__ = HaloNetH0.__doc__

HaloNet26T.__doc__ = __head_doc__ + """Model weights are reloaded from timm [Github rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models).

Args:
""" + __tail_doc__
HaloNetSE33T.__doc__ = HaloNet26T.__doc__
HaloNextECA26T.__doc__ = HaloNet26T.__doc__
HaloNet50T.__doc__ = HaloNet26T.__doc__
HaloRegNetZB.__doc__ = HaloNet26T.__doc__
HaloBotNet50T.__doc__ = HaloNet26T.__doc__

halo_attention.__doc__ = __head_doc__ + """
Halo Attention. Defined as function, not layer.
Extract patches with a `kernel_size` from `key_value` as an enlarged attention area.
Also adds `RelativePositionalEmbedding` to `attention_scores`.

Args:
  inputs: input tensor.
  num_heads: Number of attention heads.
  key_dim: Size of each attention head for query and key. Default `0` for `key_dim = inputs.shape[-1] // num_heads`
  block_size: works like `kernel_size` from `Conv2D`, extracting input patches as `query`.
  halo_size: expansion to `block_size`, extracting input patches as `key` and `value`.
  strides: downsample strides for `query`.
  out_shape: The expected shape of an output tensor. If not specified, projects back to the input dim.
  out_weight: Boolean, whether use an ouput dense.
  out_bias: Boolean, whether the ouput dense layer use bias vectors/matrices.
  attn_dropout: Dropout probability for attention.

Examples:

>>> from keras_cv_attention_models import attention_layers
>>> inputs = keras.layers.Input([14, 16, 256])
>>> nn = attention_layers.halo_attention(inputs, num_heads=4)
>>> print(f"{nn.shape = }")
# nn.shape = TensorShape([None, 14, 16, 256])

>>> mm = keras.models.Model(inputs, nn)
>>> mm.summary()
>>> print({ii.name: ii.shape for ii in mm.weights})
# {'conv2d_1/kernel:0': TensorShape([1, 1, 256, 512]),
#  'conv2d/kernel:0': TensorShape([1, 1, 256, 256]),
#  'relative_positional_embedding/r_height:0': TensorShape([64, 7]),
#  'relative_positional_embedding/r_width:0': TensorShape([64, 7]),
#  'dense/kernel:0': TensorShape([256, 256])}
"""
