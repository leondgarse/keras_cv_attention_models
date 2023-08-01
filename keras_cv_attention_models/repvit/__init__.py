from keras_cv_attention_models.repvit.repvit import RepViT, RepViT_M1, RepViT_M2, RepViT_M3


__head_doc__ = """
Keras implementation of [Github facebookresearch/LeViT](https://github.com/facebookresearch/LeViT).
Paper [PDF 2104.01136 LeViT: a Vision Transformer in ConvNet’s Clothing for Faster Inference](https://arxiv.org/pdf/2104.01136.pdf).
"""

__tail_doc__ = """  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation used in whole model, default `hard_swish`.
  drop_connect_rate: is used for [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382).
      Can be a constant value like `0.2`,
      or a tuple value like `(0, 0.2)` indicates the drop probability linearly changes from `0 --> 0.2` for `top --> bottom` layers.
      A higher value means a higher probability will drop the deep branch.
      or `0` to disable (default).
  dropout: top dropout rate if top layers is included. Default 0.
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
      Default is `None`.
  use_distillation: Boolean value if output `distill_head`. Default `True`.
  pretrained: one of `None` (random initialization) or 'imagenet' (pre-training on ImageNet).
      Will try to download and load pre-trained model weights if not None.
  **kwargs: other parameters if available.

Returns:
    A `keras.Model` instance.
"""

RepViT.__doc__ = __head_doc__ + """
Args:
  patch_channel: channel dimension output for `path_stem`.
  out_channels: output channels for each stack.
  num_heads: heads number for transformer blocks in each stack.
  depthes: number of block for each stack.
  key_dims: key dimension for transformer blocks in each stack.
  attn_ratios: `value` channel dimension expansion for transformer blocks in each stack.
  mlp_ratios: dimension expansion ration for `mlp_block` in each stack.
  strides: strides for each stack.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model                   | Params | FLOPs | Input | Top1 Acc |
  | ----------------------- | ------ | ----- | ----- | -------- |
  | LeViT128S, distillation | 7.8M   | 0.31G | 224   | 76.6     |
  | LeViT128, distillation  | 9.2M   | 0.41G | 224   | 78.6     |
  | LeViT192, distillation  | 11M    | 0.66G | 224   | 80.0     |
  | LeViT256, distillation  | 19M    | 1.13G | 224   | 81.6     |
  | LeViT384, distillation  | 39M    | 2.36G | 224   | 82.6     |
"""

RepViT_M1.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

RepViT_M2.__doc__ = RepViT_M1.__doc__
RepViT_M3.__doc__ = RepViT_M1.__doc__
