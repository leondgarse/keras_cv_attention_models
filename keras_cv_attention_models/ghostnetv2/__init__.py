from keras_cv_attention_models.ghostnetv2.ghostnetv2 import GhostNetV2, GhostNetV2_1X

__head_doc__ = """
Keras implementation of [Gitee mindspore/models/ghostnetv2](https://gitee.com/mindspore/models/tree/master/research/cv/ghostnetv2).
Paper [PDF GhostNetV2: Enhance Cheap Operation with Long-Range Attention](https://openreview.net/pdf/6db544c65bbd0fa7d7349508454a433c112470e2.pdf).
"""

__tail_doc__ = """  stem_strides: strides for stem `Conv2D`, default `2`.
  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation used in whole model, default "relu".
  dropout: dropout rate if top layers is included.
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
  pretrained: One of `[None, "imagenet"]`.
  **kwargs: other parameters if available.

Returns:
  A `keras.Model` instance.
"""

GhostNetV2.__doc__ = __head_doc__ + """
Args:
  width_mul: expansion ratio of `fist_ghost_channels` and `out_channels` in each block.
  stem_width: output dimension for stem block.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model             | Params | FLOPs  | Input | Top1 Acc |
  | ----------------- | ------ | ------ | ----- | -------- |
  | GhostNetV2_1X     | 6.12M  | 168.5M | 224   | 74.41    |
  | GhostNetV2 (1.0x) | 6.12M  | 168.5M | 224   | 75.3     |
  | GhostNetV2 (1.3x) | 8.96M  | 271.1M | 224   | 76.9     |
  | GhostNetV2 (1.6x) | 12.39M | 400.9M | 224   | 77.8     |
"""

GhostNetV2_1X.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__
