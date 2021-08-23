from keras_cv_attention_models.resnet_family.resnext import ResNeXt, ResNeXt50, ResNeXt101, ResNeXt50D, ResNeXt101W
from keras_cv_attention_models.resnet_family.resnet_quad import ResNetQ, ResNet51Q, ResNet61Q
from keras_cv_attention_models.resnet_family.resnet_deep import ResNet50D, ResNet101D, ResNet152D, ResNet200D


__head_doc__ = """
Keras implementation of [Github facebookresearch/ResNeXt](https://github.com/facebookresearch/ResNeXt).
Paper [PDF 1611.05431 Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf).
"""

__tail_doc__ = """  strides: a `number` or `list`, indicates strides used in the last stack or list value for all stacks.
      If a number, it will be `[1, 2, 2, strides]`.
  out_channels: default as `[128, 256, 512, 1024]`. Output channel for each stack.
  stem_width: output dimension for stem block.
  deep_stem: Boolean value if use deep stem.
  stem_downsample: Boolean value if ass `MaxPooling2D` layer after stem block.
  cardinality: Control channel expansion in each block, the bigger the widder.
      Also the `groups` number for `groups_depthwise` in each block, bigger `cardinality` leads to less `groups`.
  input_shape: it should have exactly 3 inputs channels, default `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation used in whole model, default `relu`.
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
      Default is `softmax`.
  pretrained: one of `None` (random initialization) or 'imagenet' (pre-training on ImageNet).
      Will try to download and load pre-trained model weights if not None.
  **kwargs: other parameters if available.

Returns:
    A `keras.Model` instance.
"""

ResNeXt.__doc__ = __head_doc__ + """
Args:
  num_blocks: number of blocks in each stack.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model          | Params | Image  resolution | Top1 Acc |
  | -------------- | ------ | ----------------- | -------- |
  | resnext50      | 25M    | 224               | 77.8     |
  | resnext101     | 42M    | 224               | 80.9     |
"""

ResNeXt50.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

ResNeXt101.__doc__ = ResNeXt50.__doc__
