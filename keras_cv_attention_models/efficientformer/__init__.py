from keras_cv_attention_models.efficientformer.efficientformer import EfficientFormer, EfficientFormerL1, EfficientFormerL3, EfficientFormerL7
from keras_cv_attention_models.efficientformer.efficientformer_v2 import EfficientFormerV2, EfficientFormerV2S0, EfficientFormerV2S1, EfficientFormerV2S2, EfficientFormerV2L

__v1_head_doc__ = """
Keras implementation of [Github snap-research/efficientformer](https://github.com/snap-research/efficientformer).
Paper [PDF 2206.01191 EfficientFormer: Vision Transformers at MobileNet Speed](https://arxiv.org/pdf/2206.01191.pdf).
"""

__common_head_doc__ = """
Args:
  num_blocks: number of block for each stack.
  out_channels: output channels for each stack.
  mlp_ratios: int or list value indicates expand ratio for mlp blocks hidden channel in each stack.
      Each element can also be a int or list, indicates `mlp_ratio` for each block.
  num_attn_blocks_each_stack: int or list value indicates number of attention blocks in each stack tail.
      If int, means using same value for all stacks, or should be same length with `num_blocks`.
      `0` means not using attention block in relative stack, or should be less than relative `num_blocks`.
      Default `[0, 0, 0, 1]` for `EfficientFormer`, `[0, 0, 2, 2]` for `EfficientFormerV2`.
  stem_width: output dimension for stem block, default `-1` for using out_channels[0].
  stem_activation: activation for stem branch. None for same with `activation`
  model_name: string, model name.
"""

__tail_doc__ = """  layer_scale: layer scale init value. Default `-1` means not applying, any value `>=0` will add a scale value for each block output.
      [Going deeper with Image Transformers](https://arxiv.org/pdf/2103.17239.pdf).
  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation used in whole stacks, default `gelu`.
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

EfficientFormer.__doc__ = __v1_head_doc__ + __common_head_doc__ + __tail_doc__ + """
Model architectures:
  | Model                      | Params | FLOPs | Input | Top1 Acc |
  | -------------------------- | ------ | ----- | ----- | -------- |
  | EfficientFormerL1, distill | 12.3M  | 1.31G | 224   | 79.2     |
  | EfficientFormerL3, distill | 31.4M  | 3.95G | 224   | 82.4     |
  | EfficientFormerL7, distill | 74.4M  | 9.79G | 224   | 83.3     |
"""

EfficientFormerL1.__doc__ = __v1_head_doc__ + """
Args:
""" + __tail_doc__

EfficientFormerL3.__doc__ = EfficientFormerL1.__doc__
EfficientFormerL7.__doc__ = EfficientFormerL1.__doc__

__v2_head_doc__ = """
Keras implementation of [Github snap-research/efficientformer](https://github.com/snap-research/efficientformer).
Paper [PDF 2212.08059 Rethinking Vision Transformers for MobileNet Size and Speed](https://arxiv.org/pdf/2212.08059.pdf).
"""

EfficientFormerV2.__doc__ = __v2_head_doc__ + __common_head_doc__ + __tail_doc__ + """
Model architectures:
  | Model                        | Params | FLOPs  | Input | Top1 Acc |
  | ---------------------------- | ------ | ------ | ----- | -------- |
  | EfficientFormerV2S0, distill | 3.60M  | 405.2M | 224   | 76.2     |
  | EfficientFormerV2S1, distill | 6.19M  | 665.6M | 224   | 79.7     |
  | EfficientFormerV2S2, distill | 12.7M  | 1.27G  | 224   | 82.0     |
  | EfficientFormerV2L, distill  | 26.3M  | 2.59G  | 224   | 83.5     |
"""

EfficientFormerV2S0.__doc__ = __v2_head_doc__ + """
Args:
""" + __tail_doc__

EfficientFormerV2S1.__doc__ = EfficientFormerV2S0.__doc__
EfficientFormerV2S2.__doc__ = EfficientFormerV2S0.__doc__
EfficientFormerV2L.__doc__ = EfficientFormerV2S0.__doc__
