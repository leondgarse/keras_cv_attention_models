from keras_cv_attention_models.pvt.pvt import PyramidVisionTransformerV2, PVT_V2B0, PVT_V2B1, PVT_V2B2, PVT_V2B2_linear, PVT_V2B3, PVT_V2B4, PVT_V2B5

__head_doc__ = """
Keras implementation of [Github whai362/PVT](https://github.com/whai362/PVT/tree/v2/classification).
Paper [PDF 2106.13797 PVTv2: Improved Baselines with Pyramid Vision Transformer](https://arxiv.org/pdf/2106.13797.pdf).
"""

__tail_doc__ = """  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation used in whole model, default `gelu`.
  drop_connect_rate: is used for [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382).
      Can be a constant value like `0.2`,
      or a tuple value like `(0, 0.2)` indicates the drop probability linearly changes from `0 --> 0.2` for `top --> bottom` layers.
      A higher value means a higher probability will drop the deep branch.
      or `0` to disable (default).
  layer_scale: layer scale init value, [Going deeper with Image Transformers](https://arxiv.org/abs/2103.17239).
      Default 0 for not using.
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
      Default is `None`.
  pretrained: one of `None` (random initialization) or 'imagenet21k-ft1k' (pre-training on ImageNet21k and fine-tuned ImageNet).
      Will try to download and load pre-trained model weights if not None.

Returns:
    A `keras.Model` instance.
"""

PyramidVisionTransformerV2.__doc__ = __head_doc__ + """
Args:
  num_blocks: number of blocks in each stack.
  embed_dims: output channels for each stack.
  num_heads: int or list value indicates heads number for transformer blocks in each stack.
  mlp_ratios: int or list value indicates expand ratio for mlp blocks hidden channel in each stack.
  sr_ratios: int or list value indicates attention blocks key_value downsample rate in each stack.
  stem_patch_size: stem patch size. Default `7`.
  use_linear: boolean value if using linear complexity attention layer with `AvgPool2D`. True for `PVT_V2B2_linear`.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model           | Params | FLOPs  | Input | Top1 Acc |
  | --------------- | ------ | ------ | ----- | -------- |
  | PVT_V2B0        | 3.7M   | 580.3M | 224   | 70.5     |
  | PVT_V2B1        | 14.0M  | 2.14G  | 224   | 78.7     |
  | PVT_V2B2        | 25.4M  | 4.07G  | 224   | 82.0     |
  | PVT_V2B2_linear | 22.6M  | 3.94G  | 224   | 82.1     |
  | PVT_V2B3        | 45.2M  | 6.96G  | 224   | 83.1     |
  | PVT_V2B4        | 62.6M  | 10.19G | 224   | 83.6     |
  | PVT_V2B5        | 82.0M  | 11.81G | 224   | 83.8     |
"""

PVT_V2B0.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

PVT_V2B1.__doc__ = PVT_V2B0.__doc__
PVT_V2B2.__doc__ = PVT_V2B0.__doc__
PVT_V2B2_linear.__doc__ = PVT_V2B0.__doc__
PVT_V2B3.__doc__ = PVT_V2B0.__doc__
PVT_V2B4.__doc__ = PVT_V2B0.__doc__
PVT_V2B5.__doc__ = PVT_V2B0.__doc__
