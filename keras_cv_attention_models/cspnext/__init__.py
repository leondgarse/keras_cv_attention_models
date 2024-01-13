from keras_cv_attention_models.cspnext.cspnext import (
    CSPNeXt,
    CSPNeXtTiny,
    CSPNeXtSmall,
    CSPNeXtMedium,
    CSPNeXtLarge,
    CSPNeXtXLarge,
)

__head_doc__ = """
Keras implementation of [Github open-mmlab/mmdetection/rtmdet](https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet#classification).
CSPNeXt is the backbone from Paper [PDF 2212.07784 RTMDet: An Empirical Study of Designing Real-Time Object Detectors](https://arxiv.org/abs/2212.07784).
"""

__tail_doc__ = """  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation used in whole model, default `gelu`.
  dropout: dropout rate if top layers is included.
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
  pretrained: one of None or "imagenet".
      Will try to download and load pre-trained model weights if not None.

Returns:
    A `keras.Model` instance.
"""

CSPNeXt.__doc__ = __head_doc__ + """
Args:
  num_blocks: number of blocks in each stack.
  out_channels: output channels for each stack.
  stem_width: hidden dimension stem blocks.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model         | Params | FLOPs | Input | Top1 Acc |
  | ------------- | ------ | ----- | ----- | -------- |
  | CSPNeXtTiny   | 2.73M  | 0.34G | 224   | 69.44    |
  | CSPNeXtSmall  | 4.89M  | 0.66G | 224   | 74.41    |
  | CSPNeXtMedium | 13.05M | 1.92G | 224   | 79.27    |
  | CSPNeXtLarge  | 27.16M | 4.19G | 224   | 81.30    |
  | CSPNeXtXLarge | 48.85M | 7.75G | 224   | 82.10    |
"""

CSPNeXtTiny.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

CSPNeXtSmall.__doc__ = CSPNeXtTiny.__doc__
CSPNeXtMedium.__doc__ = CSPNeXtTiny.__doc__
CSPNeXtLarge.__doc__ = CSPNeXtTiny.__doc__
CSPNeXtXLarge.__doc__ = CSPNeXtTiny.__doc__
