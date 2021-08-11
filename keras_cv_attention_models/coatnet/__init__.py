from keras_cv_attention_models.coatnet.coatnet import CoAtNet, CoAtNet0, CoAtNet1, CoAtNet2, CoAtNet3, CoAtNet4, CoAtNet5

__head_doc__ = """
Keras implementation of `CoAtNet`, or `Conv + Transformer` networks.
Paper [PDF 2106.04803 CoAtNet: Marrying Convolution and Attention for All Data Sizes](https://arxiv.org/pdf/2106.04803.pdf).
"""

__tail_doc__ = """  block_types: block types for each stack,
      - `conv` or any `c` / `C` starts word, means `res_MBConv` block.
      - `transfrom` or any not `c` / `C` starts word, means `res_mhsa` + `res_ffn` block.
      value could be in format like `"cctt"` or `"CCTT"` or `["conv", "conv", "transfrom", "transform"]`.
  expansion: filter expansion in each block. The larger the wider.
  se_ratio: value in `(0, 1)`, where `0` means not using `se_module`. Should be a `number` or `list`, indicates `se_ratio` for each stack.
      Each element can also be a `number` or `list`, indicates `se_ratio` for each block.
  num_heads: heads number for transformer block. `64` for `CoAtNet5`, `32` for others.
  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation used in whole model, default `relu`.
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
  pretrained: None available.
  **kwargs: other parameters if available.

Returns:
    A `keras.Model` instance.
"""

CoAtNet.__doc__ = __head_doc__ + """
Args:
  num_blocks: number of blocks in each stack.
  out_channels: output channels for each stack.
  stem_width: output dimension for stem block.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model                                | Params | Image resolution | Top1 Acc |
  | ------------------------------------ | ------ | ---------------- | -------- |
  | CoAtNet-0                            | 25M    | 224              | 81.6     |
  | CoAtNet-0                            | 25M    | 384              | 83.9     |
  | CoAtNet-1                            | 42M    | 224              | 83.3     |
  | CoAtNet-1                            | 42M    | 384              | 85.1     |
  | CoAtNet-2                            | 75M    | 224              | 84.1     |
  | CoAtNet-2                            | 75M    | 384              | 85.7     |
  | CoAtNet-2                            | 75M    | 512              | 85.9     |
  | CoAtNet-2, ImageNet-21k pretrain     | 75M    | 224              | 87.1     |
  | CoAtNet-2, ImageNet-21k pretrain     | 75M    | 384              | 87.1     |
  | CoAtNet-2, ImageNet-21k pretrain     | 75M    | 512              | 87.3     |
  | CoAtNet-3                            | 168M   | 224              | 84.5     |
  | CoAtNet-3                            | 168M   | 384              | 85.8     |
  | CoAtNet-3                            | 168M   | 512              | 86.0     |
  | CoAtNet-3, ImageNet-21k pretrain     | 168M   | 224              | 87.6     |
  | CoAtNet-3, ImageNet-21k pretrain     | 168M   | 384              | 87.6     |
  | CoAtNet-3, ImageNet-21k pretrain     | 168M   | 512              | 87.9     |
  | CoAtNet-4, ImageNet-21k pretrain     | 275M   | 384              | 87.9     |
  | CoAtNet-4, ImageNet-21k pretrain     | 275M   | 512              | 88.1     |
  | CoAtNet-4, ImageNet-21K + PT-RA-E150 | 275M   | 384              | 88.4     |
  | CoAtNet-4, ImageNet-21K + PT-RA-E150 | 275M   | 512              | 88.56    |
  | CoAtNet-5                            | 680M   |                  |          |
"""

CoAtNet0.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

CoAtNet1.__doc__ = CoAtNet0.__doc__
CoAtNet2.__doc__ = CoAtNet0.__doc__
CoAtNet3.__doc__ = CoAtNet0.__doc__
CoAtNet4.__doc__ = CoAtNet0.__doc__
CoAtNet5.__doc__ = CoAtNet0.__doc__
