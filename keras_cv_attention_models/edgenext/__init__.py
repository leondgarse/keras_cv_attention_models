from keras_cv_attention_models.edgenext.edgenext import (
    EdgeNeXt,
    EdgeNeXt_XX_Small,
    EdgeNeXt_X_Small,
    EdgeNeXt_Small,
    PositionalEncodingFourier,
    cross_covariance_attention,
)

__head_doc__ = """
Keras implementation of [Github mmaaz60/EdgeNeXt](https://github.com/mmaaz60/EdgeNeXt).
Paper [PDF 2206.10589 EdgeNeXt: Efficiently Amalgamated CNN-Transformer Architecture for Mobile Vision Applications](https://arxiv.org/pdf/2206.10589.pdf).
"""

__tail_doc__ = """  mlp_ratio: expand ratio for mlp blocks hidden channel.
  stem_patch_size: stem patch size for stem strides.
  layer_scale: layer scale init value. Default `-1` means not applying, any value `>=0` will add a scale value for each block output.
      [Going deeper with Image Transformers](https://arxiv.org/pdf/2103.17239.pdf).
  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation used in whole model, default `gelu`.
  drop_connect_rate: is used for [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382).
      Can be value like `0.2`, indicates the drop probability linearly changes from `0 --> 0.2` for `top --> bottom` layers.
      A higher value means a higher probability will drop the deep branch.
      or `0` to disable (default).
  dropout: dropout rate if top layers is included.
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
  pretrained: None or "imagenet" or one of ["imagenet", "usi"] for `EdgeNeXt_Small`.

Returns:
    A `keras.Model` instance.
"""

EdgeNeXt.__doc__ = __head_doc__ + """
Args:
  num_blocks: number of blocks in each stack.
  out_channels: output channels for each stack.
  num_heads: num heads for each stack.
  num_stda_layers: number of `split_depthwise_transpose_attention` blocks used in each stack tail.
  stda_split: number of inputs split for `sdta` blocks in each stack.
  stda_use_pos_emb: list of boolean value if use `PositionalEncodingFourier` positional embedding for `sdta` blocks in each stack.
  conv_kernel_size: list of int numbers, kernel_size for `conv_encoder` blocks in each stack.
  stem_width: output dimension for stem block, default `-1` for using out_channels[0]
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model             | Params | FLOPs  | Input | Top1 Acc |
  | ----------------- | ------ | ------ | ----- | -------- |
  | EdgeNeXt_XX_Small | 1.33M  | 266M   | 256   | 71.23    |
  | EdgeNeXt_X_Small  | 2.34M  | 547M   | 256   | 74.96    |
  | EdgeNeXt_Small    | 5.59M  | 1.27G  | 256   | 79.41    |
  | - usi             | 5.59M  | 1.27G  | 256   | 81.07    |
"""

EdgeNeXt_XX_Small.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

EdgeNeXt_X_Small.__doc__ = EdgeNeXt_XX_Small.__doc__
EdgeNeXt_Small.__doc__ = EdgeNeXt_XX_Small.__doc__

PositionalEncodingFourier.__doc__ = __head_doc__ + """
Positional Encoding Fourier layer.
Layer weight shape depends on parameter `filters` and input channel dimension only.

Args:
  filters: int number, encoded positional dimension for each point.
  temperature: temperature.

Examples:

>>> from keras_cv_attention_models import attention_layers
>>> aa = attention_layers.PositionalEncodingFourier(filters=16)
>>> print(f"{aa(tf.ones([1, 12, 27, 64])).shape = }")
# aa(tf.ones([1, 12, 27, 64])).shape = TensorShape([1, 12, 27, 64])

>>> print({ii.name: ii.shape for ii in aa.weights})
# {'positional_encoding_fourier_1/ww:0': TensorShape([32, 64]), 'positional_encoding_fourier_1/bb:0': TensorShape([64])}
"""

cross_covariance_attention.__doc__ = __head_doc__ + """
Cross Covariance Attention block. Defined as function, not layer.
It's different from traditional MHSA. This is using `attention_scores` shape `[batch, num_heads, key_dim, key_dim]`,
while traditional MHSA `attention_scores` shape `[batch, num_heads, hh * ww, hh * ww]`.
Also using cosine distance between `query` and `key` calculating `attention_scores`.

Args:
  inputs: input tensor.
  num_heads: Number of attention heads.
  key_dim: Size of each attention head for query, key and value. Default `0` for `key_dim = inputs.shape[-1] // num_heads`.
  qkv_bias: Boolean, whether the qkv dense layer use bias vectors/matrices.
  out_bias: Boolean, whether the ouput dense layer use bias vectors/matrices.
  attn_dropout: Dropout probability for attention.
  out_dropout: Dropout probability for output.

Examples:

>>> from keras_cv_attention_models import attention_layers
>>> inputs = keras.layers.Input([14, 16, 256])
>>> nn = attention_layers.cross_covariance_attention(inputs, num_heads=4, name="attn_")
>>> print(f"{nn.shape = }")
# nn.shape = TensorShape([None, 14, 16, 256])

>>> mm = keras.models.Model(inputs, nn)
>>> mm.summary()
>>> print({ii.name: ii.shape for ii in mm.weights})
# {'attn_qkv/kernel:0': TensorShape([256, 768]),
#  'attn_qkv/bias:0': TensorShape([768]),
#  'attn_temperature/no_weight_decay/weight:0': TensorShape([4, 1, 1]),
#  'attn_output/kernel:0': TensorShape([256, 256]),
#  'attn_output/bias:0': TensorShape([256])}
"""
