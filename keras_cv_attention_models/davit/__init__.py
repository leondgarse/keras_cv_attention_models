from keras_cv_attention_models.davit.davit import (
    DaViT,
    DaViT_T,
    DaViT_S,
    DaViT_B,
    DaViT_L,
    DaViT_H,
    DaViT_G,
    multi_head_self_attention_channel,
    window_attention,
)

__head_doc__ = """
Keras implementation of [Github dingmyu/davit](https://github.com/dingmyu/davit).
Paper  [PDF 2204.03645 DaViT: Dual Attention Vision Transformers](https://arxiv.org/pdf/2204.03645.pdf).
"""

__tail_doc__ = """  window_ratio: window_size ratio, window_size = [input_shape[0] // window_ratio, input_shape[1] // window_ratio].
  mlp_ratio: expand ratio for mlp blocks hidden channel.
  stem_patch_size: stem patch size for stem strides.
  layer_scale: layer scale init value. Default `-1` means not applying, any value `>=0` will add a scale value for each block output.
      [Going deeper with Image Transformers](https://arxiv.org/pdf/2103.17239.pdf).
  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
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

DaViT.__doc__ = __head_doc__ + """
Args:
  num_blocks: number of blocks in each stack.
  out_channels: output channels for each stack.
  num_heads: num heads for each stack.
  stem_width: output dimension for stem block, default `-1` for using out_channels[0]
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model         | Params | FLOPs  | Input | Top1 Acc |
  | ------------- | ------ | ------ | ----- | -------- |
  | DaViT_T       | 28.36M | 4.56G  | 224   | 82.8     |
  | DaViT_S       | 49.75M | 8.83G  | 224   | 84.2     |
  | DaViT_B       | 87.95M | 15.55G | 224   | 84.6     |
  | DaViT_L, 21k  | 196.8M | 103.2G | 384   | 87.5     |
  | DaViT_H, 1.5B | 348.9M | 327.3G | 512   | 90.2     |
  | DaViT_G, 1.5B | 1.406B | 1.022T | 512   | 90.4     |
"""

DaViT_T.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

DaViT_S.__doc__ = DaViT_T.__doc__
DaViT_B.__doc__ = DaViT_T.__doc__
DaViT_L.__doc__ = DaViT_T.__doc__
DaViT_H.__doc__ = DaViT_T.__doc__
DaViT_G.__doc__ = DaViT_T.__doc__

multi_head_self_attention_channel.__doc__ = __head_doc__ + """
Multi head self attention on channel dimension. Defined as function, not layer.
It's different from traditional MHSA, that using `attention_scores` shape `[batch, num_heads, key_dim, key_dim]`,
while traditional MHSA `attention_scores` shape `[batch, num_heads, hh * ww, hh * ww]`

Args:
  inputs: input tensor.
  num_heads: Number of attention heads.
  key_dim: Size of each attention head for query, key and value. Default `0` for `key_dim = inputs.shape[-1] // num_heads`.
  out_shape: The expected shape of an output tensor. If not specified, projects back to the input dim.
  out_weight: Boolean, whether use an ouput dense.
  qkv_bias: Boolean, whether the qkv dense layer use bias vectors/matrices.
  out_bias: Boolean, whether the ouput dense layer use bias vectors/matrices.
  attn_dropout: Dropout probability for attention.

Examples:

>>> from keras_cv_attention_models import attention_layers
>>> inputs = keras.layers.Input([14, 16, 256])
>>> nn = attention_layers.multi_head_self_attention_channel(inputs, num_heads=4, out_shape=512, name="attn_")
>>> print(f"{nn.shape = }")
# nn.shape = TensorShape([None, 14, 16, 512])

>>> mm = keras.models.Model(inputs, nn)
>>> mm.summary()
>>> print({ii.name: ii.shape for ii in mm.weights})
# {'attn_qkv/kernel:0': TensorShape([256, 768]), 'attn_output/kernel:0': TensorShape([256, 512])}
"""

window_attention.__doc__ = __head_doc__ + """
Window multi head self attention. Defined as function, not layer.
Typical MHSA with `window_partition` process ahead and `window_reverse` process after.
Also works like a wrapper, perform `window_partition -> attention_block -> window_reverse`.

Args:
  inputs: input tensor.
  window_size: window partition size.
  num_heads: Number of attention heads.
  attention_block: specific callable attention block, instead of preset MHSA.
      3 parameters are required supporting: `inputs, num_heads, name`
  kwargs: any additional kwargs for `attention_block`.

Examples:

>>> from keras_cv_attention_models import attention_layers
>>> inputs = keras.layers.Input([14, 16, 256])
>>> nn = attention_layers.window_attention(inputs, 7, num_heads=4, name="attn_")
>>> print(f"{nn.shape = }")
# nn.shape = TensorShape([None, 14, 16, 256])

>>> mm = keras.models.Model(inputs, nn)
>>> mm.summary()
>>> print({ii.name: ii.shape for ii in mm.weights})
# {'attn_qkv/kernel:0': TensorShape([256, 768]),
#  'attn_qkv/bias:0': TensorShape([768]),
#  'attn_output/kernel:0': TensorShape([256, 256]),
#  'attn_output/bias:0': TensorShape([256])}
"""
