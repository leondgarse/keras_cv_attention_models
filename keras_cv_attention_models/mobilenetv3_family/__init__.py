from keras_cv_attention_models.mobilenetv3_family.fbnetv3 import FBNetV3, FBNetV3B, FBNetV3D, FBNetV3G
from keras_cv_attention_models.mobilenetv3_family.lcnet import LCNet, LCNet050, LCNet075, LCNet100, LCNet150, LCNet200, LCNet250
from keras_cv_attention_models.mobilenetv3_family.mobilenetv3 import (
    MobileNetV3,
    MobileNetV3Large,
    MobileNetV3Small,
    MobileNetV3Large075,
    MobileNetV3Large100,
    MobileNetV3Small050,
    MobileNetV3Small075,
    MobileNetV3Small100,
)
from keras_cv_attention_models.mobilenetv3_family.tinynet import TinyNet, TinyNetA, TinyNetB, TinyNetC, TinyNetD, TinyNetE

__head_doc__ = """
Args:
  [Stack parameters]
  num_blocks: number of blocks in each stack.
  out_channels: output channels for each stack.
  expands: should be a `number` or `list`, indicates hidden expand ratio for each stack.
      Each element can also be a `number` or `list`, indicates hidden expand ratio for each block.
  kernel_sizes: a list matching num_blocks, indicates kernel_size for each stack.
  strides: a list value matching num_blocks, indicates stride in first block of each stack.
  activations: a string for activation of all stacks, or a list value matching num_blocks indicates activation of each stack.
  disable_shortcut: boolean value force disable using shortcut in `inverted_residual_block` for all stacks. True for LCNet, False for others.
  use_blocks_output_activation: boolean value if adding additional activation layer after each block.
      True for LCNet, False for others.
  width_ratio: out_channels expansion ratio.

  [Stem parameters]
  stem_width: output dimension for stem block.
  fix_stem: boolean value if fix stem_width, False for expand with `width_ratio`.
  stem_feature_activation: activation for stem and output feature. "swish" for TinyNet, "hard_swish" for others.

  [SE module parameters]
  se_ratios: a float number for se_ratio of all stacks, or a list value matching num_blocks indicates se_ratio of each stack.
  se_activation: activation for semodule, can be one string value like "relu" for activation after se reduction,
      or two string value like ("relu", "hard_sigmoid_torch") for both activation after se reduction and expansion.
      Set None for se reduction activation same with block activation, expansion activation using "sigmoid".
      ("hard_swish", "hard_sigmoid_torch") for FBNetV3, None for TinyNet, ("relu", "hard_sigmoid_torch") for LCNet and MobileNetV3.
  se_limit_round_down: make sure that round down does not go down by more than [num] reatio in se module.
      0.95 for FBNetV3, 0.9 for others.
  se_divisor: divisor in se module. 1 for TinyNet, 8 for others.
  use_expanded_se_ratio: boolean value contols se reduction ratio.
      False for FBNetV3 and TinyNet, se reduction channel is calculated from `input_channel`, means not expanded with block `expand`.
      True for LCNet and MobileNetV3, se reduction channel is calculated from `hidden_channel`, means also expanded with block `expand`.

  [Output parameters]
  output_num_features: none `0` value to add another `conv2d + bn + activation` layers before classifier output.
  use_additional_output_conv: boolean value if using additional `conv + bn + activation` block after all stacks.
      False for LCNet and TinyNet, True for FBNetV3 and MobileNetV3.
  use_output_feature_bias: boolean value if using bias for `features` block. False for FBNetV3 and TinyNet, True for LCNet and MobileNetV3.
  use_avg_pool_conv_output: boolean value if using `avg_pool -> conv -> activation` or `conv -> bn -> activation -> avg_pool`
      as `features` block. False for TinyNet, True for others.
"""

__tail_doc__ = """  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`. Set `(None, None, 3)` for dynamic input shape.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  drop_connect_rate: is used for [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382).
      Can be value like `0.2`, indicates the drop probability linearly changes from `0 --> 0.2` for `top --> bottom` layers.
      A higher value means a higher probability will drop the deep branch.
      or `0` to disable (default).
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
  dropout: dropout rate if top layers is included.
  pretrained: None or one of ["imagenet", "miil", "miil_21k", "ssld"].
      ["miil", "miil_21k"] for `MobileNetV3Large100`.
      "ssld" for `LCNet050` `LCNet100` `LCNet250`.
  **kwargs: other parameters if available.

Returns:
    A `keras.Model` instance.
"""

__mobilenetv3_head_doc__ = """
Keras implementation of [timm/mobilenetv3](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/mobilenetv3.py).
Paper [PDF 1905.02244 Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf).
"""

MobileNetV3.__doc__ = __mobilenetv3_head_doc__ + __head_doc__ + """
  [Model common parameters]
""" + __tail_doc__ + """
Model architectures:
  | Model               | Params | FLOPs   | Input | Top1 Acc |
  | ------------------- | ------ | ------- | ----- | -------- |
  | MobileNetV3Small050 | 1.29M  | 24.92M  | 224   | 57.89    |
  | MobileNetV3Small075 | 2.04M  | 44.35M  | 224   | 65.24    |
  | MobileNetV3Small100 | 2.54M  | 57.62M  | 224   | 67.66    |
  | MobileNetV3Large075 | 3.99M  | 156.30M | 224   | 73.44    |
  | MobileNetV3Large100 | 5.48M  | 218.73M | 224   | 75.77    |
  | - miil              | 5.48M  | 218.73M | 224   | 77.92    |
"""

MobileNetV3Large.__doc__ = MobileNetV3.__doc__
MobileNetV3Small.__doc__ = MobileNetV3.__doc__

MobileNetV3Large075.__doc__ = __mobilenetv3_head_doc__ + """
Args:
""" + __tail_doc__

MobileNetV3Large100.__doc__ = MobileNetV3Large075.__doc__
MobileNetV3Small050.__doc__ = MobileNetV3Large075.__doc__
MobileNetV3Small075.__doc__ = MobileNetV3Large075.__doc__
MobileNetV3Small100.__doc__ = MobileNetV3Large075.__doc__


__fbnet_head_doc__ = """
Keras implementation of [timm/mobilenetv3](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/mobilenetv3.py).
Paper [PDF 2006.02049 FBNetV3: Joint Architecture-Recipe Search using Predictor Pretraining](https://arxiv.org/pdf/2006.02049.pdf).
"""

FBNetV3.__doc__ = __fbnet_head_doc__ + __head_doc__ + """
  [Model common parameters]
""" + __tail_doc__ + """
Model architectures:
  | Model    | Params | FLOPs    | Input | Top1 Acc |
  | -------- | ------ | -------- | ----- | -------- |
  | FBNetV3B | 5.57M  | 539.82M  | 256   | 79.15    |
  | FBNetV3D | 10.31M | 665.02M  | 256   | 79.68    |
  | FBNetV3G | 16.62M | 1379.30M | 256   | 82.05    |
"""

FBNetV3B.__doc__ = __fbnet_head_doc__ + """
Args:
""" + __tail_doc__

FBNetV3D.__doc__ = FBNetV3B.__doc__
FBNetV3G.__doc__ = FBNetV3B.__doc__


__lcnet_head_doc__ = """
Keras implementation of [Github PaddlePaddle/PaddleClas](https://github.com/PaddlePaddle/PaddleClas).
Paper [PDF 2109.15099 PP-LCNet: A Lightweight CPU Convolutional Neural Network](https://arxiv.org/pdf/2109.15099.pdf).
"""

LCNet.__doc__ = __lcnet_head_doc__ + __head_doc__ + """
  [Model common parameters]
""" + __tail_doc__ + """
Model architectures:
  | Model    | Params | FLOPs   | Input | Top1 Acc |
  | -------- | ------ | ------- | ----- | -------- |
  | LCNet050 | 1.88M  | 46.02M  | 224   | 63.10    |
  | - ssld   | 1.88M  | 46.02M  | 224   | 66.10    |
  | LCNet075 | 2.36M  | 96.82M  | 224   | 68.82    |
  | LCNet100 | 2.95M  | 158.28M | 224   | 72.10    |
  | - ssld   | 2.95M  | 158.28M | 224   | 74.39    |
  | LCNet150 | 4.52M  | 338.05M | 224   | 73.71    |
  | LCNet200 | 6.54M  | 585.35M | 224   | 75.18    |
  | LCNet250 | 9.04M  | 900.16M | 224   | 76.60    |
  | - ssld   | 9.04M  | 900.16M | 224   | 80.82    |
"""

LCNet050.__doc__ = __lcnet_head_doc__ + """
Args:
""" + __tail_doc__

LCNet075.__doc__ = LCNet050.__doc__
LCNet100.__doc__ = LCNet050.__doc__
LCNet150.__doc__ = LCNet050.__doc__
LCNet200.__doc__ = LCNet050.__doc__
LCNet250.__doc__ = LCNet050.__doc__


__tinynet_head_doc__ = """
Keras implementation of [Github huawei-noah/CV-Backbones/tinynet_pytorch](https://github.com/huawei-noah/CV-Backbones/tree/master/tinynet_pytorch).
Paper [PDF 2010.14819 Model Rubik’s Cube: Twisting Resolution, Depth and Width for TinyNets](https://arxiv.org/pdf/2010.14819.pdf).
"""

TinyNet.__doc__ = __tinynet_head_doc__ + __head_doc__ + """
  [Model common parameters]
""" + __tail_doc__ + """
Model architectures:
  | Model    | Params | FLOPs   | Input | Top1 Acc |
  | -------- | ------ | ------- | ----- | -------- |
  | TinyNetE | 2.04M  | 25.22M  | 106   | 59.86    |
  | TinyNetD | 2.34M  | 53.35M  | 152   | 66.96    |
  | TinyNetC | 2.46M  | 103.22M | 184   | 71.23    |
  | TinyNetB | 3.73M  | 206.28M | 188   | 74.98    |
  | TinyNetA | 6.19M  | 343.74M | 192   | 77.65    |
"""

TinyNetE.__doc__ = __tinynet_head_doc__ + """
Args:
""" + __tail_doc__

TinyNetD.__doc__ = TinyNetE.__doc__
TinyNetC.__doc__ = TinyNetE.__doc__
TinyNetB.__doc__ = TinyNetE.__doc__
TinyNetA.__doc__ = TinyNetE.__doc__
