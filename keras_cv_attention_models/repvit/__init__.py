from keras_cv_attention_models.repvit.repvit import RepViT, RepViT_M1, RepViT_M2, RepViT_M3, switch_to_deploy


__head_doc__ = """
Keras implementation of [Github THU-MIG/RepViT](https://github.com/THU-MIG/RepViT).
Paper [PDF 2307.09283 RepViT: Revisiting Mobile CNN From ViT Perspective](https://arxiv.org/pdf/2307.09283.pdf).
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
  num_blocks: number of block for each stack.
  out_channels: output channels for each stack.
  stem_width: channel dimension output for stem block, default -1 for using out_channels[0].
  se_ratio: float value for se_ratio for each stack, will use `se_module` every other block in each stack if > 0.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model                   | Params | FLOPs | Input | Top1 Acc |
  | ----------------------- | ------ | ----- | ----- | -------- |
  | RepViT_M1, distillation | 5.10M  | 0.82G | 224   | 78.5     |
  | - switch_to_deploy      | 5.07M  | 0.82G | 224   | 78.5     |
  | RepViT_M2, distillation | 8.28M  | 1.35G | 224   | 80.6     |
  | - switch_to_deploy      | 8.25M  | 1.35G | 224   | 80.6     |
  | RepViT_M3, distillation | 10.2M  | 1.87G | 224   | 81.4     |
  | - switch_to_deploy      | 10.12M | 1.87G | 224   | 81.4     |
"""

RepViT_M1.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

RepViT_M2.__doc__ = RepViT_M1.__doc__
RepViT_M3.__doc__ = RepViT_M1.__doc__
