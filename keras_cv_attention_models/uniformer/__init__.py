from keras_cv_attention_models.uniformer.uniformer import (
    Uniformer,
    UniformerSmall32,
    UniformerSmall64,
    UniformerSmallPlus32,
    UniformerSmallPlus64,
    UniformerBase32,
    UniformerBase64,
    UniformerLarge64,
    multi_head_self_attention,
)

__head_doc__ = """
Keras implementation of [UniFormer](https://github.com/Sense-X/UniFormer/tree/main/image_classification).
Paper [PDF 2201.09450 UniFormer: Unifying Convolution and Self-attention for Visual Recognition](https://arxiv.org/pdf/2201.09450.pdf).
"""

__tail_doc__ = """  block_types: block types for each stack,
      - `conv` or any `c` / `C` starts word, means `res_MBConv` block.
      - `transfrom` or any not `c` / `C` starts word, means `res_mhsa` + `res_ffn` block.
      value could be in format like `"cctt"` or `"CCTT"` or `["conv", "conv", "transfrom", "transform"]`.
  stem_width: output dimension for stem block. Default -1 means using `out_channels[0]`
  qkv_bias: boolean value if useing bias for `qkv` in `mhsa` block, default `True`.
  mlp_ratio: channel expansion ratio for mlp hidden layers, default 4.
  layer_scale: layer scale init value. Default `-1` means not applying, any value `>=0` will add a scale value for each block output.
      [Going deeper with Image Transformers](https://arxiv.org/pdf/2103.17239.pdf).
  mix_token: boolean value if using mix token augment, should work together with `token_label_top`, default `False`.
  token_label_top: boolean value if output `class + token header`, Default `False`.
      [All Tokens Matter: Token Labeling for Training Better Vision Transformers](https://arxiv.org/pdf/2104.10858.pdf).
  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation used in whole model, default `relu`.
  mlp_drop_rate: dropout rate for mlp blocks.
  attn_drop_rate: dropout rate for mhsa attention scores.
  drop_connect_rate: is used for [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382).
      Can be value like `0.2`, indicates the drop probability linearly changes from `0 --> 0.2` for `top --> bottom` layers.
      A higher value means a higher probability will drop the deep branch.
      or `0` to disable (default).
  dropout: dropout rate if top layers is included.
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
  pretrained: None or one of ["imagenet", "token_label"].

Returns:
    A `keras.Model` instance.
"""

Uniformer.__doc__ = __head_doc__ + """
Args:
  num_blocks: number of blocks in each stack.
  out_channels: output channels for each stack.
  head_dimension: heads number for transformer block. `32` for `xxx32` models and `64` for `xxx64` models.
  use_conv_stem: boolean value if using `Conv` stem or `Patch` stem. `True` for `UniformerSmallPlus*`, `False` for others.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model                 | Params | FLOPs  | Input | Top1 Acc |
  | --------------------- | ------ | ------ | ----- | -------- |
  | UniformerSmall32 + TL | 22M    | 3.66G  | 224   | 83.4     |
  | UniformerSmall64      | 22M    | 3.66G  | 224   | 82.9     |
  | - Token Labeling      | 22M    | 3.66G  | 224   | 83.4     |
  | UniformerSmallPlus32  | 24M    | 4.24G  | 224   | 83.4     |
  | - Token Labeling      | 24M    | 4.24G  | 224   | 83.9     |
  | UniformerSmallPlus64  | 24M    | 4.23G  | 224   | 83.4     |
  | - Token Labeling      | 24M    | 4.23G  | 224   | 83.6     |
  | UniformerBase32 + TL  | 50M    | 8.32G  | 224   | 85.1     |
  | UniformerBase64       | 50M    | 8.31G  | 224   | 83.8     |
  | - Token Labeling      | 50M    | 8.31G  | 224   | 84.8     |
  | UniformerLarge64 + TL | 100M   | 19.79G | 224   | 85.6     |
  | UniformerLarge64 + TL | 100M   | 63.11G | 384   | 86.3     |
"""

UniformerSmall32.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

UniformerSmall64.__doc__ = UniformerSmall32.__doc__
UniformerSmallPlus32.__doc__ = UniformerSmall32.__doc__
UniformerSmallPlus64.__doc__ = UniformerSmall32.__doc__
UniformerBase32.__doc__ = UniformerSmall32.__doc__
UniformerBase64.__doc__ = UniformerSmall32.__doc__
UniformerLarge64.__doc__ = UniformerSmall32.__doc__

multi_head_self_attention.__doc__ = """
Typical multi head self attention block, should work similar with `keras.layers.MultiHeadAttention`. Defined as function, not layer.

Args:
  inputs: input tensor.
  num_heads: Number of attention heads.
  key_dim: Size of each attention head for query and key. Default `0` for `key_dim = inputs.shape[-1] // num_heads`.
  out_shape: The expected shape of an output tensor. If not specified, projects back to the input dim.
  out_weight: Boolean, whether use an ouput dense.
  qkv_bias: Boolean, whether the qkv dense layer use bias vectors/matrices.
  out_bias: Boolean, whether the ouput dense layer use bias vectors/matrices.
  attn_dropout: Dropout probability for attention scores.
  output_dropout: Dropout probability for attention output.

Examples:

>>> from keras_cv_attention_models import attention_layers
>>> inputs = keras.layers.Input([14, 16, 256])
>>> nn = attention_layers.multi_head_self_attention(inputs, num_heads=4, out_shape=512)
>>> print(f"{nn.shape = }")
# nn.shape = TensorShape([None, 14, 16, 512])

>>> mm = keras.models.Model(inputs, nn)
>>> mm.summary()
>>> print({ii.name: ii.shape for ii in mm.weights})
# {'dense/kernel:0': TensorShape([256, 1024]), 'dense_1/kernel:0': TensorShape([512, 512])}
"""
