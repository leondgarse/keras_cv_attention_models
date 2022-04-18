from keras_cv_attention_models.resnet_family.resnext import ResNeXt, ResNeXt50, ResNeXt101, ResNeXt50D, ResNeXt101W, ResNeXt101W_64
from keras_cv_attention_models.resnet_family.resnet_quad import ResNetQ, ResNet51Q, ResNet61Q
from keras_cv_attention_models.resnet_family.resnet_deep import ResNetD, ResNet50D, ResNet101D, ResNet152D, ResNet200D
from keras_cv_attention_models.resnet_family.regnet import (
    RegNetY,
    RegNetY040,
    RegNetY064,
    RegNetY080,
    RegNetY160,
    RegNetY320,
    RegNetZB16,
    RegNetZC16,
    RegNetZC16_EVO,
    RegNetZD32,
    RegNetZD8,
    RegNetZD8_EVO,
    RegNetZE8
)


__resnext_head_doc__ = """
Keras implementation of [Github facebookresearch/ResNeXt](https://github.com/facebookresearch/ResNeXt).
Paper [PDF 1611.05431 Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf).

Pre-trained `swsl` means `Semi-Weakly Supervised ResNe*t`
    from [Github facebookresearch/semi-supervised-ImageNet1K-models](https://github.com/facebookresearch/semi-supervised-ImageNet1K-models).
    **Please note the CC-BY-NC 4.0 license on theses weights, non-commercial use only**.
"""

__tail_doc__ = """  input_shape: it should have exactly 3 inputs channels, default `(224, 224, 3)`.
    Set `(None, None, 3)` for dynamic input shape.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation used in whole model.
  drop_connect_rate: is used for [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382).
      Can be value like `0.2`, indicates the drop probability linearly changes from `0 --> 0.2` for `top --> bottom` layers.
      A higher value means a higher probability will drop the deep branch.
      or `0` to disable (default).
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
      Default is `softmax`.
  pretrained: value in {pretrained_list}.
      Will try to download and load pre-trained model weights if not None.
      Save path is `~/.keras/models/`.
  **kwargs: other parameters in `keras_cv_attention_models.aotnet.AotNet` if not conflict.

Returns:
    A `keras.Model` instance.
"""

ResNeXt.__doc__ = __resnext_head_doc__ + """
Args:
  num_blocks: number of blocks in each stack.
  out_channels: Output channel for each stack.
      `[128, 256, 512, 1024]`. for regular `32x4d` models,
      `[256, 512, 1024, 2048]` for `32x8d` wider models.
  strides: a `number` or `list`, indicates strides used in the last stack or list value for all stacks.
      If a number, it will be `[1, 2, 2, strides]`. Default `2`.
  model_name: string, model name.
""" + __tail_doc__.format(pretrained_list=[None, "imagenet", "swsl"]) + """
Model architectures:
  | Model                     | Params | FLOPs  | Input | Top1 Acc |
  | ------------------------- | ------ | ------ | ----- | -------- |
  | ResNeXt50 (32x4d)         | 25M    | 4.23G  | 224   | 79.768   |
  | - SWSL                    | 25M    | 4.23G  | 224   | 82.182   |
  | ResNeXt50D (32x4d + deep) | 25M    | 4.47G  | 224   | 79.676   |
  | ResNeXt101 (32x4d)        | 42M    | 7.97G  | 224   | 80.334   |
  | - SWSL                    | 42M    | 7.97G  | 224   | 83.230   |
  | ResNeXt101W (32x8d)       | 89M    | 16.41G | 224   | 79.308   |
  | - SWSL                    | 89M    | 16.41G | 224   | 84.284   |
  | ResNeXt101W_64 (64x4d)    | 83.46M | 15.46G | 224   | 82.46    |
"""

__resnext_default_doc__ = __resnext_head_doc__ + """
Args:
""" + __tail_doc__.format(pretrained_list=[None, "imagenet", "swsl"])

ResNeXt50.__doc__ = __resnext_default_doc__
ResNeXt101.__doc__ = __resnext_default_doc__
ResNeXt101W.__doc__ = __resnext_default_doc__
ResNeXt50D.__doc__ = __resnext_head_doc__ + """
Args:
""" + __tail_doc__.format(pretrained_list=[None, "imagenet"])
ResNeXt101W_64.__doc__ = ResNeXt50D.__doc__

__resnetd_head_doc__ = """
Github source [leondgarse/keras_cv_attention_models](https://github.com/leondgarse/keras_cv_attention_models).
Keras implementation of ResNetD.
Paper [PDF 1812.01187 Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/pdf/1812.01187.pdf).
"""

ResNetD.__doc__ = __resnetd_head_doc__ + """
Args:
  num_blocks: number of blocks in each stack.
  deep_stem: Boolean value if use deep stem. Default `True`.
  stem_width: output dimension for stem block. Default `32`.
  strides: a `number` or `list`, indicates strides used in the last stack or list value for all stacks.
      If a number, it will be `[1, 2, 2, strides]`. Default `2`.
  model_name: string, model name.
""" + __tail_doc__.format(pretrained_list=[None, "imagenet"]) + """
Model architectures:
  | Model      | Params | FLOPs  | Input | Top1 Acc |
  | ---------- | ------ | ------ | ----- | -------- |
  | ResNet50D  | 25.58M | 4.33G  | 224   | 80.530   |
  | ResNet101D | 44.57M | 8.04G  | 224   | 83.022   |
  | ResNet152D | 60.21M | 11.75G | 224   | 83.680   |
  | ResNet200D | 64.69M | 15.25G | 224   | 83.962   |
"""

__resnetd_default_doc__ = __resnetd_head_doc__ + """
Args:
""" + __tail_doc__.format(pretrained_list=[None, "imagenet"])

ResNet50D.__doc__ = __resnetd_default_doc__
ResNet101D.__doc__ = __resnetd_default_doc__
ResNet152D.__doc__ = __resnetd_default_doc__
ResNet200D.__doc__ = __resnetd_default_doc__

__resnetq_head_doc__ = """
Github source [leondgarse/keras_cv_attention_models](https://github.com/leondgarse/keras_cv_attention_models).
Defined and model weights loaded from [Github timm/models/resnet.py](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/resnet.py).
"""

__resnetq_tail_doc__ = """  strides: a `number` or `list`, indicates strides used in the last stack or list value for all stacks.
      If a number, it will be `[1, 2, 2, strides]`. Default `2`.
  stem_downsample: Boolean value if add `MaxPooling2D` layer after stem block.
  input_shape: it should have exactly 3 inputs channels, default `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation used in whole model, default `relu`.
  drop_connect_rate: is used for [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382).
      Can be value like `0.2`, indicates the drop probability linearly changes from `0 --> 0.2` for `top --> bottom` layers.
      A higher value means a higher probability will drop the deep branch.
      or `0` to disable (default).
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
      Default is `softmax`.
  pretrained: value in [None, "imagenet"].
      Will try to download and load pre-trained model weights if not None.
      Save path is `~/.keras/models/`.

Returns:
    A `keras.Model` instance.
"""

ResNetQ.__doc__ = __resnetq_head_doc__ + """
Args:
  num_blocks: number of blocks in each stack.
  out_channels: Output channel for each stack.
  stem_width: output dimension for stem block.
  stem_act: Boolean value if add use `batchnorm + activation` in stem branch.
  expansion: filter expansion in each block. The larger the wider.
  groups_div: number value or list of number, controls `groups` in each stack.
      - `0` means using `edge block`, which removes the first `1x1` conv.
      - `(0, 1)` values means `groups=filter // groups_div` for `Conv2D`.
      - `1` means `groups == filter`, will apply `DepthwiseConv2D`.
  extra_conv: Boolean value or list of Boolean, if add another `3x3` conv in each block.
  num_features: none `0` value to add another `conv2d + bn + activation` layers before `GlobalAveragePooling2D`.
  model_name: string, model name.
""" + __resnetq_tail_doc__ + """
Model architectures:
  | Model     | Params | FLOPs | Input | Top1 Acc |
  | --------- | ------ | ----- | ----- | -------- |
  | ResNet51Q | 35.7M  | 4.87G | 224   | 82.36    |
  | ResNet61Q | 36.8M  | 5.96G | 224   |          |
"""

ResNet51Q.__doc__ = __resnetq_head_doc__ + """
Args:
""" + __resnetq_tail_doc__
ResNet61Q.__doc__ = __resnetq_head_doc__ + """
Args:
""" + __resnetq_tail_doc__

__regnety_head_doc__ = """
Keras implementation of [Github facebookresearch/regnet](https://github.com/facebookresearch/pycls/blob/main/pycls/models/regnet.py).
Paper [PDF 2003.13678 Designing Network Design Spaces](https://arxiv.org/pdf/2003.13678.pdf).
Model weights loaded from [timm/regnet](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/regnet.py).
Paper [PDF 2110.00476 ResNet strikes back: An improved training procedure in timm](https://arxiv.org/pdf/2110.00476.pdf).
"""

RegNetY.__doc__ = __regnety_head_doc__ + """
Args:
  num_blocks: number of blocks in each stack.
  out_channels: Output channel for each stack.
  model_name: string, model name.
""" + __tail_doc__.format(pretrained_list=[None, "imagenet"]) + """
Model architectures:
  | Model      | Params  | FLOPs  | Input | Top1 Acc |
  | ---------- | ------- | ------ | ----- | -------- |
  | RegNetY040 | 20.65M  | 3.98G  | 224   | 82.3     |
  | RegNetY064 | 30.58M  | 6.36G  | 224   | 83.0     |
  | RegNetY080 | 39.18M  | 7.97G  | 224   | 83.17    |
  | RegNetY160 | 83.59M  | 15.92G | 224   | 82.0     |
  | RegNetY320 | 145.05M | 32.29G | 224   | 82.5     |
"""

__regnety_default_doc__ = __regnety_head_doc__ + """
Args:
""" + __tail_doc__.format(pretrained_list=[None, "imagenet"])

RegNetY040.__doc__ = __regnety_default_doc__
RegNetY064.__doc__ = __regnety_default_doc__
RegNetY080.__doc__ = __regnety_default_doc__
RegNetY160.__doc__ = __regnety_default_doc__
RegNetY320.__doc__ = __regnety_default_doc__

__regnetz_head_doc__ = """
Github source [leondgarse/keras_cv_attention_models](https://github.com/leondgarse/keras_cv_attention_models).
Defined and model weights loaded from [timm](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/byobnet.py).
Related paper [PDF 2004.02967 Evolving Normalization-Activation Layers](https://arxiv.org/pdf/2004.02967.pdf).
"""

__regnetz_default_doc__ = __regnetz_head_doc__ + """
Args:
""" + __tail_doc__.format(pretrained_list=[None, "imagenet"]) + """
Model architectures:
  | Model          | Params | FLOPs | Input | Top1 Acc |
  | -------------- | ------ | ----- | ----- | -------- |
  | RegNetZB16     | 9.72M  | 1.44G | 224   | 79.868   |
  | RegNetZC16     | 13.46M | 2.50G | 256   | 82.164   |
  | RegNetZC16_EVO | 13.49M | 2.55G | 256   | 81.9     |
  | RegNetZD32     | 27.58M | 5.96G | 256   | 83.422   |
  | RegNetZD8      | 23.37M | 3.95G | 256   | 83.5     |
  | RegNetZD8_EVO  | 23.46M | 4.61G | 256   | 83.42    |
  | RegNetZE8      | 57.70M | 9.88G | 256   | 84.5     |
"""

RegNetZB16.__doc__ = __regnetz_default_doc__
RegNetZC16.__doc__ = __regnetz_default_doc__
RegNetZC16_EVO.__doc__ = __regnetz_default_doc__
RegNetZD32.__doc__ = __regnetz_default_doc__
RegNetZD8.__doc__ = __regnetz_default_doc__
RegNetZD8_EVO.__doc__ = __regnetz_default_doc__
RegNetZE8.__doc__ = __regnetz_default_doc__
