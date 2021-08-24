from keras_cv_attention_models.resnet_family.resnext import ResNeXt, ResNeXt50, ResNeXt101, ResNeXt50D, ResNeXt101W
from keras_cv_attention_models.resnet_family.resnet_quad import ResNetQ, ResNet51Q, ResNet61Q
from keras_cv_attention_models.resnet_family.resnet_deep import ResNetD, ResNet50D, ResNet101D, ResNet152D, ResNet200D


__resnext_head_doc__ = """
Keras implementation of [Github facebookresearch/ResNeXt](https://github.com/facebookresearch/ResNeXt).
Paper [PDF 1611.05431 Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf).

Pre-trained `swsl` means `Semi-Weakly Supervised ResNe*t`
    from [Github facebookresearch/semi-supervised-ImageNet1K-models](https://github.com/facebookresearch/semi-supervised-ImageNet1K-models).
    **Please note the CC-BY-NC 4.0 license on theses weights, non-commercial use only**.
"""

__tail_doc__ = """  input_shape: it should have exactly 3 inputs channels, default `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation used in whole model, default `relu`.
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
  expansion: filter expansion in each block. `2` for regular `32x4d` models, `1` for `32x8d` wider models.
  strides: a `number` or `list`, indicates strides used in the last stack or list value for all stacks.
      If a number, it will be `[1, 2, 2, strides]`. Default `2`.
  model_name: string, model name.
""" + __tail_doc__.format(pretrained_list=[None, "imagenet", "swsl"]) + """
Model architectures:
  | Model                     | Params | Image  resolution | Top1 Acc |
  | ------------------------- | ------ | ----------------- | -------- |
  | ResNeXt50 (32x4d)         | 25M    | 224               | 79.768   |
  | - SWSL                    | 25M    | 224               | 82.182   |
  | ResNeXt50D (32x4d + deep) | 25M    | 224               | 79.676   |
  | ResNeXt101 (32x4d)        | 42M    | 224               | 80.334   |
  | - SWSL                    | 42M    | 224               | 83.230   |
  | ResNeXt101W (32x8d)       | 89M    | 224               | 79.308   |
  | - SWSL                    | 89M    | 224               | 84.284   |
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
  avg_pool_down: Boolean value if use `AvgPool2D` in shortcut branch. `True` for `ResNetD` model.
  model_name: string, model name.
""" + __tail_doc__.format(pretrained_list=[None, "imagenet"]) + """
Model architectures:
  | Model      | Params | Image  resolution | Top1 Acc |
  | ---------- | ------ | ----------------- | -------- |
  | ResNet50D  | 25.58M | 224               | 80.530   |
  | ResNet101D | 44.57M | 224               | 83.022   |
  | ResNet152D | 60.21M | 224               | 83.680   |
  | ResNet200D | 64.69  | 224               | 83.962   |
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
  | Model     | Params | Image  resolution | Top1 Acc |
  | --------- | ------ | ----------------- | -------- |
  | ResNet51Q | 35.7M  | 224               | 82.36    |
  | ResNet61Q | 36.8M  | 224               |          |
"""

ResNet51Q.__doc__ = __resnetq_head_doc__ + """
Args:
""" + __resnetq_tail_doc__
ResNet61Q.__doc__ = __resnetq_head_doc__ + """
Args:
""" + __resnetq_tail_doc__
