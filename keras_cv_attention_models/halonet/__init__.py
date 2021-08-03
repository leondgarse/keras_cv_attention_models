from keras_cv_attention_models.halonet.halonet import HaloNet, HaloNetB0, HaloNetB1, HaloNetB2, HaloNetB3, HaloNetB4, HaloNetB5, HaloNetB6, HaloNetB7, HaloAttention

__head_doc__ = """
Keras implementation of [Github lucidrains/halonet-pytorch](https://github.com/lucidrains/halonet-pytorch).
Paper [PDF 2103.12731 Scaling Local Self-Attention for Parameter Efficient Visual Backbones](https://arxiv.org/pdf/2103.12731.pdf).
"""

__tail_doc__ = """  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
  activation: activation used in whole model, default `relu`.
  pretrained: one of `None` (random initialization) or 'imagenet' (pre-training on ImageNet).
      Will try to download and load pre-trained model weights if not None.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
      Default is `softmax`.
  model_name: string, model name.
  **kwargs: other parameters if available.

Returns:
    A `keras.Model` instance.
"""

HaloNet.__doc__ = __head_doc__ + """
Args:
  model_type: value in ['b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7'].
""" + __tail_doc__ + """
Model architectures:
  | Model   | params | Image resolution | Top1 Acc |
  | ------- | ------ | ---------------- | -------- |
  | halo_b0 | 4.6M   | 256              |          |
  | halo_b1 | 8.8M   | 256              |          |
  | halo_b2 | 11.04M | 256              |          |
  | halo_b3 | 15.1M  | 320              |          |
  | halo_b4 | 31.4M  | 384              | 85.5%    |
  | halo_b5 | 34.4M  | 448              |          |
  | halo_b6 | 47.98M | 512              |          |
  | halo_b7 | 68.4M  | 600              |          |
"""

HaloNetB0.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

HaloNetB1.__doc__ = HaloNetB0.__doc__
HaloNetB2.__doc__ = HaloNetB0.__doc__
HaloNetB3.__doc__ = HaloNetB0.__doc__
HaloNetB4.__doc__ = HaloNetB0.__doc__
HaloNetB5.__doc__ = HaloNetB0.__doc__
HaloNetB6.__doc__ = HaloNetB0.__doc__
HaloNetB7.__doc__ = HaloNetB0.__doc__

HaloAttention.__doc__ = __head_doc__ + """
Halo-Attention layer.

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

Args:
  num_heads: Number of attention heads.
  key_dim: Size of each attention head for query and key.
  block_size: works like `kernel_size` from `Conv2D`, extracting input patches as `query`.
  halo_size: expansion to `block_size`, extracting input patches as `key` and `value`.
  out_shape: The expected shape of an output tensor. If not specified, projects back to the key feature dim.
  out_weight: Boolean, whether use an ouput dense.
  out_bias: Boolean, whether the ouput dense layer use bias vectors/matrices.
  attn_dropout: Dropout probability for attention.
"""
