from keras_cv_attention_models.ghostnet.ghostnet_v2 import GhostNetV2, GhostNetV2_100
from keras_cv_attention_models.ghostnet.ghostnet import GhostNet, GhostNet_050, GhostNet_100, GhostNet_130

__v2_head_doc__ = """
Keras implementation of [Gitee mindspore/models/ghostnetv2](https://gitee.com/mindspore/models/tree/master/research/cv/ghostnetv2).
Paper [PDF GhostNetV2: Enhance Cheap Operation with Long-Range Attention](https://openreview.net/pdf/6db544c65bbd0fa7d7349508454a433c112470e2.pdf).
"""

__tail_doc__ = """  kernel_sizes: kernel_size for each stack.
  first_ghost_channels: num channels for first ghost module in each stack.
  out_channels: output channels for each stack.
  se_ratios: se_ratio for each stack.
  strides: stride for each stack.
  stem_width: output dimension for stem block.
  stem_strides: strides for stem `Conv2D`, default `2`.
  num_ghost_module_v1_stacks: num of `ghost_module` stcks on the head, others are `ghost_module_multiply`.
      - for `GhostNet` v1 way, default `-1` for all using `ghost_module`.
      - for `GhostNetV2` way, default `2` for only using `ghost_module` in the first 2 stacks.
  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation used in whole model, default "relu".
  dropout: dropout rate if top layers is included.
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
  pretrained: One of `[None, "imagenet", "ssld"]`. "ssld" if for `GhostNet_130`.
  **kwargs: other parameters if available.

Returns:
  A `keras.Model` instance.
"""

GhostNetV2.__doc__ = __v2_head_doc__ + """
Args:
  width_mul: expansion ratio of `fist_ghost_channels` and `out_channels` in each block.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model             | Params | FLOPs  | Input | Top1 Acc |
  | ----------------- | ------ | ------ | ----- | -------- |
  | GhostNetV2_100    | 6.12M  | 168.5M | 224   | 74.41    |
  | GhostNetV2 (1.0x) | 6.12M  | 168.5M | 224   | 75.3     |
  | GhostNetV2 (1.3x) | 8.96M  | 271.1M | 224   | 76.9     |
  | GhostNetV2 (1.6x) | 12.39M | 400.9M | 224   | 77.8     |
"""

GhostNetV2_100.__doc__ = __v2_head_doc__ + """
Args:
""" + __tail_doc__

__v1_head_doc__ = """
Keras implementation of [Github huawei-noah/ghostnet_pytorch](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/ghostnet_pytorch).
Paper [PDF 1911.11907 GhostNet: More Features from Cheap Operations](https://arxiv.org/pdf/1911.11907.pdf).
"""

GhostNet.__doc__ = __v1_head_doc__ + """
Args:
  width_mul: expansion ratio of `fist_ghost_channels` and `out_channels` in each block.
  stem_width: output dimension for stem block.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model        | Params | FLOPs  | Input | Top1 Acc |
  | ------------ | ------ | ------ | ----- | -------- |
  | GhostNet_050 | 2.59M  | 42.6M  | 224   | 66.88    |
  | GhostNet_100 | 5.18M  | 141.7M | 224   | 74.16    |
  | GhostNet_130 | 7.36M  | 227.7M | 224   | 75.79    |
  | - ssld       | 7.36M  | 227.7M | 224   | 79.38    |
"""

GhostNet_050.__doc__ = __v1_head_doc__ + """
Args:
""" + __tail_doc__

GhostNet_100.__doc__ = GhostNet_050.__doc__
GhostNet_130.__doc__ = GhostNet_050.__doc__
