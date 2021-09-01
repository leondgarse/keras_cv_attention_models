from keras_cv_attention_models.halonet.halonet import HaloNet, HaloNetH0, HaloNetH1, HaloNetH2, HaloNetH3, HaloNetH4, HaloNetH5, HaloNetH6, HaloNetH7, HaloAttention


__head_doc__ = """
Keras implementation of [Github lucidrains/halonet-pytorch](https://github.com/lucidrains/halonet-pytorch).
Paper [PDF 2103.12731 Scaling Local Self-Attention for Parameter Efficient Visual Backbones](https://arxiv.org/pdf/2103.12731.pdf).
"""

__tail_doc__ = """  out_channels: output channels for each stack, default `[64, 128, 256, 512]`.
  strides: strides for each stack, default `[1, 2, 2, 2]`.
  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation used in whole model, default `relu`.
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
      Default is `softmax`.
  pretrained: None available.
  **kwargs: other parameters if available.

Returns:
    A `keras.Model` instance.
"""

HaloNet.__doc__ = __head_doc__ + """
Args:
  halo_block_size: block_size for halo attention layers.
  halo_halo_size: halo_size for halo attention layers.
  halo_expansion: filter expansion for halo attention output channel.
  expansion: filter expansion for each block output channel.
  output_conv_channel: none `0` value to add another `conv2d + bn + activation` layers before `GlobalAveragePooling2D`.
  num_blocks: number of blocks in each stack.
  num_heads: number of heads in each stack.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model     | Params | Image resolution | Top1 Acc |
  | --------- | ------ | ---------------- | -------- |
  | HaloNetH0 | 6.6M   | 256              |          |
  | HaloNetH1 | 9.1M   | 256              |          |
  | HaloNetH2 | 10.3M  | 256              |          |
  | HaloNetH3 | 12.5M  | 320              |          |
  | HaloNetH4 | 19.5M  | 384              | 85.5     |
  | HaloNetH5 | 31.6M  | 448              |          |
  | HaloNetH6 | 44.3M  | 512              |          |
  | HaloNetH7 | 67.9M  | 640              |          |
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

HaloAttention.__doc__ = __head_doc__ + """
Halo-Attention layer.

Args:
  num_heads: Number of attention heads.
  key_dim: Size of each attention head for query and key.
  block_size: works like `kernel_size` from `Conv2D`, extracting input patches as `query`.
  halo_size: expansion to `block_size`, extracting input patches as `key` and `value`.
  strides: downsample strides for `query`.
  out_shape: The expected shape of an output tensor. If not specified, projects back to the key feature dim.
  out_weight: Boolean, whether use an ouput dense.
  out_bias: Boolean, whether the ouput dense layer use bias vectors/matrices.
  attn_dropout: Dropout probability for attention.

Examples:

>>> from keras_cv_attention_models import attention_layers
>>> bb = attention_layers.HaloAttention()
>>> print(f"{bb(tf.ones([1, 14, 16, 256])).shape = }")
bb(tf.ones([1, 14, 16, 256])).shape = TensorShape([1, 14, 16, 512])

>>> print({ii.name:ii.numpy().shape for ii in bb.weights})
{'halo_attention_2/query:0': (256, 512),
 'halo_attention_2/key_value:0': (256, 1024),
 'halo_attention_2/output_weight:0': (512, 512),
 'halo_attention_2/r_width:0': (128, 7),
 'halo_attention_2/r_height:0': (128, 7)}
"""
