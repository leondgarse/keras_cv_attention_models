from keras_cv_attention_models.convnext.convnext import ConvNeXt, ConvNeXtTiny, ConvNeXtSmall, ConvNeXtBase, ConvNeXtLarge, ConvNeXtXlarge

__head_doc__ = """
Keras implementation of [ConvNeXt](https://github.com/facebookresearch/ConvNeXt).
Paper [PDF 2201.03545 A ConvNet for the 2020s](https://arxiv.org/pdf/2201.03545.pdf).
"""

__tail_doc__ = """  layer_scale_init_value: layer scale init value, [Going deeper with Image Transformers](https://arxiv.org/abs/2103.17239). Default `1e-6`.
  head_init_scale: init head layer scale value if create model from scratch or fine-tune. Default `1`.
  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation used in whole model, default `gelu`.
  drop_connect_rate: is used for [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382).
      Can be a constant value like `0.2`,
      or a tuple value like `(0, 0.2)` indicates the drop probability linearly changes from `0 --> 0.2` for `top --> bottom` layers.
      A higher value means a higher probability will drop the deep branch. or `0` to disable.
      Default 0.1.
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
      Default is `None`.
  dropout: dropout rate if top layers is included.
  pretrained: one of `None` (random initialization) or 'imagenet' (pre-training on ImageNet).
      or 'imagenet21k-ft1k' (pre-trined on ImageNet21k, fine-tuning on ImageNet).
      Will try to download and load pre-trained model weights if not None.

Returns:
    A `keras.Model` instance.

Model architectures:
  | Model               | Params | FLOPs   | Input | Top1 Acc |
  | ------------------- | ------ | ------- | ----- | -------- |
  | ConvNeXtTiny        | 28M    | 4.49G   | 224   | 82.1     |
  | ConvNeXtSmall       | 50M    | 8.73G   | 224   | 83.1     |
  | ConvNeXtBase        | 89M    | 15.42G  | 224   | 83.8     |
  | ConvNeXtBase        | 89M    | 45.32G  | 384   | 85.1     |
  | - ImageNet21k-ft1k  | 89M    | 15.42G  | 224   | 85.8     |
  | - ImageNet21k-ft1k  | 89M    | 45.32G  | 384   | 86.8     |
  | ConvNeXtLarge       | 198M   | 34.46G  | 224   | 84.3     |
  | ConvNeXtLarge       | 198M   | 101.28G | 384   | 85.5     |
  | - ImageNet21k-ft1k  | 198M   | 34.46G  | 224   | 86.6     |
  | - ImageNet21k-ft1k  | 198M   | 101.28G | 384   | 87.5     |
  | ConvNeXtXLarge, 21k | 350M   | 61.06G  | 224   | 87.0     |
  | ConvNeXtXLarge, 21k | 350M   | 179.43G | 384   | 87.8     |
"""

ConvNeXt.__doc__ = __head_doc__ + """
Args:
  num_blocks: number of blocks in each stack.
  out_channels: output channels for each stack.
  stem_width: output dimension for stem block. Default `-1` means using `out_channels[0]`.
  model_name: string, model name.
""" + __tail_doc__

ConvNeXtTiny.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

ConvNeXtSmall.__doc__ = ConvNeXtTiny.__doc__
ConvNeXtBase.__doc__ = ConvNeXtTiny.__doc__
ConvNeXtLarge.__doc__ = ConvNeXtTiny.__doc__
ConvNeXtLarge.__doc__ = ConvNeXtTiny.__doc__
