from keras_cv_attention_models.mobilevit.mobilevit import MobileViT, MobileViT_XXS, MobileViT_XS, MobileViT_S, linear_self_attention
from keras_cv_attention_models.mobilevit.mobilevit_v2 import (
    MobileViT_V2,
    MobileViT_V2_050,
    MobileViT_V2_075,
    MobileViT_V2_100,
    MobileViT_V2_125,
    MobileViT_V2_150,
    MobileViT_V2_175,
    MobileViT_V2_200,
)

__v1_head_doc__ = """
Keras implementation of [Github apple/ml-cvnets/mobilevit](https://github.com/apple/ml-cvnets/blob/main/cvnets/models/classification/mobilevit.py).
Paper [PDF 2110.02178 MOBILEVIT: LIGHT-WEIGHT, GENERAL-PURPOSE, AND MOBILE-FRIENDLY VISION TRANSFORMER](https://arxiv.org/pdf/2110.02178.pdf).
"""

__v2_head_doc__ = """
Keras implementation of [Github apple/ml-cvnets](https://github.com/apple/ml-cvnets).
Paper [PDF 2206.02680 Separable Self-attention for Mobile Vision Transformers](https://arxiv.org/pdf/2206.02680.pdf).
"""

__args__ = """
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
  patch_size: extracted patch size for transformer blocks.
  patches_to_batch: boolean value if stack extracted patches to batch dimension. True for V1, False for V2.
  resize_first: boolena value if perform `resize -> conv` or `conv -> resize` if input shape not divisible by 32.
      False for V1, True for V2.
  use_depthwise: boolean value if using depthwise_conv2d or conv2d for `transformer_pre_process` process.
      False for V1, True for V2.
  use_fusion: boolean value if fusing with tensor before attention blocks in `transformer_post_process` process.
      True for V1, False for V2.
  num_norm_groups: norm layer froups number. -1 or 0 for V1 using layer_norm, or 1 for V2 using group_norm.
  use_linear_attention: boolean value if using `linear_self_attention` or `multi_head_self_attention` in attention block.
      False for V1, True for V2.
  use_conv_mlp: boolean value if using `conv` or `dense` for mlp blocks. False for V1, True for V2.
  output_num_features: none `0` value to add another `conv2d + bn + activation` layers before `GlobalAveragePooling2D`.
  model_name: string, model name.
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
  pretrained: None or {pretrained}.
  **kwargs: other parameters if available.

Returns:
    A `keras.Model` instance.
"""

MobileViT.__doc__ = __v1_head_doc__ + __args__ + __tail_doc__.format(pretrained="imagenet") + """
Model architectures:
  | Model         | Params | FLOPs | Input | Top1 Acc |
  | ------------- | ------ | ----- | ----- | -------- |
  | MobileViT_XXS | 1.3M   | 0.42G | 256   | 69.0     |
  | MobileViT_XS  | 2.3M   | 1.05G | 256   | 74.7     |
  | MobileViT_S   | 5.6M   | 2.03G | 256   | 78.3     |
"""

MobileViT_V2.__doc__ = __v2_head_doc__ + __args__ + __tail_doc__.format(pretrained="one of [imagenet, imagenet22k]") + """
Model architectures:
  | Model              | Params | FLOPs | Input | Top1 Acc |
  | ------------------ | ------ | ----- | ----- | -------- |
  | MobileViT_V2_050   | 1.37M  | 0.47G | 256   | 70.18    |
  | MobileViT_V2_075   | 2.87M  | 1.04G | 256   | 75.56    |
  | MobileViT_V2_100   | 4.90M  | 1.83G | 256   | 78.09    |
  | MobileViT_V2_125   | 7.48M  | 2.84G | 256   | 79.65    |
  | MobileViT_V2_150   | 10.6M  | 4.07G | 256   | 80.38    |
  | - imagenet22k      | 10.6M  | 4.07G | 256   | 81.46    |
  | - imagenet22k, 384 | 10.6M  | 9.15G | 384   | 82.60    |
  | MobileViT_V2_175   | 14.3M  | 5.52G | 256   | 80.84    |
  | - imagenet22k      | 14.3M  | 5.52G | 256   | 81.94    |
  | - imagenet22k, 384 | 14.3M  | 12.4G | 384   | 82.93    |
  | MobileViT_V2_200   | 18.4M  | 7.12G | 256   | 81.17    |
  | - imagenet22k      | 18.4M  | 7.12G | 256   | 82.36    |
  | - imagenet22k, 384 | 18.4M  | 16.2G | 384   | 83.41    |
"""

MobileViT_XXS.__doc__ = __v1_head_doc__ + """
Args:
""" + __tail_doc__.format(pretrained="imagenet")

MobileViT_XS.__doc__ = MobileViT_XXS.__doc__
MobileViT_S.__doc__ = MobileViT_XXS.__doc__

MobileViT_V2_050.__doc__ = __v2_head_doc__ + """
Args:
""" + __tail_doc__.format(pretrained="one of [imagenet, imagenet22k]")

MobileViT_V2_075.__doc__ = MobileViT_V2_050.__doc__
MobileViT_V2_100.__doc__ = MobileViT_V2_050.__doc__
MobileViT_V2_125.__doc__ = MobileViT_V2_050.__doc__
MobileViT_V2_150.__doc__ = MobileViT_V2_050.__doc__
MobileViT_V2_175.__doc__ = MobileViT_V2_050.__doc__
MobileViT_V2_200.__doc__ = MobileViT_V2_050.__doc__

linear_self_attention.__doc__ = __v2_head_doc__ + """
Linear Self Attention. Defined as function, not layer.

Args:
  inputs: input tensor.
  qkv_bias: Boolean, whether the qkv dense layer use bias vectors/matrices.
  out_bias: Boolean, whether the ouput dense layer use bias vectors/matrices.
  attn_axis: axis for applying attention softmax operation. Default 2 for MobileViT_V2 applying on `patch_hh * patch_ww` dimension.
  attn_dropout: Dropout probability for attention.

Examples:
>>> from keras_cv_attention_models import attention_layers
>>> inputs = keras.layers.Input([14, 16, 256])
>>> nn = attention_layers.linear_self_attention(inputs, attn_axis=-1)
>>> print(f"{nn.shape = }")
# nn.shape = TensorShape([None, 14, 16, 512])

>>> mm = keras.models.Model(inputs, nn)
>>> mm.summary()
>>> print({ii.name: ii.shape for ii in mm.weights})
# {'conv2d_2/kernel:0': TensorShape([1, 1, 256, 513]),
#  'conv2d_3/kernel:0': TensorShape([1, 1, 256, 256])}
"""
