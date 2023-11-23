from keras_cv_attention_models.repvit.repvit import RepViT, RepViT_M09, RepViT_M10, RepViT_M11, RepViT_M15, RepViT_M23, switch_to_deploy


__head_doc__ = """
Keras implementation of [Github THU-MIG/RepViT](https://github.com/THU-MIG/RepViT).
Paper [PDF 2307.09283 RepViT: Revisiting Mobile CNN From ViT Perspective](https://arxiv.org/pdf/2307.09283.pdf).
"""

__tail_doc__ = """  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
  deploy: boolean value if build a fused model. **Evaluation only, not good for training**.
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
  use_distillation: Boolean value if output `distill_head`. Default `False`.
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
  | Model                    | Params | FLOPs | Input | Top1 Acc |
  | ------------------------ | ------ | ----- | ----- | -------- |
  | RepViT_M09, distillation | 5.10M  | 0.82G | 224   | 79.1     |
  | - deploy=True            | 5.07M  | 0.82G | 224   | 79.1     |
  | RepViT_M10, distillation | 6.85M  | 1.12G | 224   | 80.3     |
  | - deploy=True            | 6.81M  | 1.12G | 224   | 80.3     |
  | RepViT_M11, distillation | 8.29M  | 1.35G | 224   | 81.2     |
  | - deploy=True            | 8.24M  | 1.35G | 224   | 81.2     |
  | RepViT_M15, distillation | 14.13M | 2.30G | 224   | 82.5     |
  | - deploy=True            | 14.05M | 2.30G | 224   | 82.5     |
  | RepViT_M23, distillation | 23.01M | 4.55G | 224   | 83.7     |
  | - deploy=True            | 22.93M | 4.55G | 224   | 83.7     |
"""

RepViT_M09.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

RepViT_M10.__doc__ = RepViT_M09.__doc__
RepViT_M11.__doc__ = RepViT_M09.__doc__
RepViT_M15.__doc__ = RepViT_M09.__doc__
RepViT_M23.__doc__ = RepViT_M09.__doc__
