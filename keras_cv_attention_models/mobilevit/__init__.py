from keras_cv_attention_models.mobilevit.mobilevit import MobileViT, MobileViT_XXS, MobileViT_XS, MobileViT_S

__head_doc__ = """
Keras implementation of [Github apple/ml-cvnets/mobilevit](https://github.com/apple/ml-cvnets/blob/main/cvnets/models/classification/mobilevit.py).
Paper [PDF 2110.02178 MOBILEVIT: LIGHT-WEIGHT, GENERAL-PURPOSE, AND MOBILE-FRIENDLY VISION TRANSFORMER](https://arxiv.org/pdf/2110.02178.pdf).
"""


__tail_doc__ = """  layer_scale: layer scale init value, [Going deeper with Image Transformers](https://arxiv.org/abs/2103.17239).
      Default `-1` for not using.
  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation used in whole model, default `relu`.
  drop_connect_rate: is used for [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382).
      Can be value like `0.2`, indicates the drop probability linearly changes from `0 --> 0.2` for `top --> bottom` layers.
      A higher value means a higher probability will drop the deep branch.
      or `0` to disable (default).
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
  dropout: dropout rate if top layers is included.
  pretrained: None or "imagenet". Only CMTTiny pretrained available.
  **kwargs: other parameters if available.

Returns:
    A `keras.Model` instance.
"""

MobileViT.__doc__ = __head_doc__ + """
Args:
  num_blocks: number of blocks in each stack.
  out_channels: output channels for each stack.
  attn_channels: hidden channel for transformer blocks. Can be a list matching out_channels,
      or a float number for expansion ratio of out_channels.
  block_types: block types for each stack,
      - `conv` or any `c` / `C` starts word, means `bottle_in_linear_out_block` block.
      - `transfrom` or any not `c` / `C` starts word, means `mhsa_mlp_block` block.
      value could be in format like `"cctt"` or `"CCTT"` or `["conv", "conv", "transfrom", "transform"]`.
  strides: list value for stride in first block of each stack.
  expand_ratio: conv blocks hidden channel expand ratio. Default 2 for MobileViT_XXS, 4 for MobileViT_XS and MobileViT_S.
  stem_width: output dimension for stem block.
  output_num_features: none `0` value to add another `conv2d + bn + activation` layers before `GlobalAveragePooling2D`.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model         | Params | Image resolution | Top1 Acc |
  | ------------- | ------ | ---------------- | -------- |
  | MobileViT_XXS | 1.3M   | 256              | 69.0     |
  | MobileViT_XS  | 2.3M   | 256              | 74.7     |
  | MobileViT_S   | 5.6M   | 256              | 78.3     |
"""

MobileViT_XXS.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

MobileViT_XS.__doc__ = MobileViT_XXS.__doc__
MobileViT_S.__doc__ = MobileViT_XXS.__doc__
