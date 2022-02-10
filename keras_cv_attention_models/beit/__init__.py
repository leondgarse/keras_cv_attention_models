from keras_cv_attention_models.beit.beit import (
    Beit,
    BeitBasePatch16,
    BeitLargePatch16,
    MultiHeadRelativePositionalEmbedding,
)

__head_doc__ = """
Keras implementation of [beit](https://github.com/microsoft/unilm/tree/master/beit).
Paper [PDF 2106.08254 BEIT: BERT Pre-Training of Image Transformers](https://arxiv.org/pdf/2106.08254.pdf).
"""

__tail_doc__ = """  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation used in whole model, default `gelu`.
  drop_connect_rate: is used for [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382).
      Can be a constant value like `0.2`,
      or a tuple value like `(0, 0.2)` indicates the drop probability linearly changes from `0 --> 0.2` for `top --> bottom` layers.
      A higher value means a higher probability will drop the deep branch.
      or `0` to disable (default).
  use_mean_pooling: boolean value if use mean output or `class_token` output. Default `True`.
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
      Default is `None`.
  pretrained: one of `None` (random initialization) or 'imagenet21k-ft1k' (pre-training on ImageNet21k and fine-tuned ImageNet).
      Will try to download and load pre-trained model weights if not None.

Returns:
    A `keras.Model` instance.

Model architectures:
  | Model            | Params  | Image resolution | Top1 Acc |
  | ---------------- | ------- | ---------------- | -------- |
  | BeitBasePatch16  | 86.53M  | 224              | 85.240   |
  |                  | 86.74M  | 384              | 86.808   |
  | BeitLargePatch16 | 304.43M | 224              | 87.476   |
  |                  | 305.00M | 384              | 88.382   |
  |                  | 305.67M | 512              | 88.584   |
"""

Beit.__doc__ = __head_doc__ + """
Args:
  depth: number of blocks. Default `12`.
  embed_dim: channel dimension for stem and all blocks. Default `768`.
  num_heads: heads number for transformer blocks. Default `12`.
  mlp_ratio: dimension expansion ration for `mlp_block`s. Default `4`.
  patch_size: stem patch size. Default `16`.
  attn_key_dim: key dimension for transformer blocks. Default `0`.
  attn_qv_bias: boolean value if use `bias` for `query` and `value` in transformer. Default `True`.
  attn_out_weight: boolean value if use output dense for transformer. Default `True`.
  attn_out_bias: boolean value if output dense use bias for transformer. Default `True`.
  attn_dropout: `attention_score` dropout rate. Default `0`.
  gamma_init_value: init value for `attention` and `mlp` branch `gamma`. Default `0.1`.
  model_name: string, model name.
""" + __tail_doc__

BeitBasePatch16.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

BeitLargePatch16.__doc__ = BeitBasePatch16.__doc__

MultiHeadRelativePositionalEmbedding.__doc__ = __head_doc__ + """
Multi Head Relative Positional Embedding layer.

input (with_cls_token=True): `[batch, num_heads, attn_blocks, attn_blocks]`. where `attn_blocks = attn_height * attn_width + class_token`
input (with_cls_token=False): `[batch, num_heads, attn_blocks, attn_blocks]`. where `attn_blocks = attn_height * attn_width`
output: `[batch, num_heads, attn_blocks, attn_blocks] + positional_bias`.
conditions: attn_height == attn_width

Args:
  with_cls_token: boolean value if input is with class_token.
  attn_height: specify `height` for `attn_blocks` if not square `attn_height != attn_width`.
  num_heads: specify num_heads, or using `input.shape[1]`.

Examples:

>>> from keras_cv_attention_models import attention_layers
>>> aa = attention_layers.MultiHeadRelativePositionalEmbedding()
>>> print(f"{aa(tf.ones([1, 8, 29 * 29 + 1, 29 * 29 + 1])).shape = }")
# aa(tf.ones([1, 8, 29 * 29 + 1, 29 * 29 + 1])).shape = TensorShape([1, 8, 842, 842])
>>> print({ii.name:ii.numpy().shape for ii in aa.weights})
# {'multi_head_relative_positional_embedding/pos_emb:0': (3252, 8)}

>>> aa = attention_layers.MultiHeadRelativePositionalEmbedding(with_cls_token=False)
>>> print(f"{aa(tf.ones([1, 8, 29 * 29, 29 * 29])).shape = }")
# aa(tf.ones([1, 8, 29 * 29, 29 * 29])).shape = TensorShape([1, 8, 841, 841])
>>> print({ii.name:ii.numpy().shape for ii in aa.weights})
# {'multi_head_relative_positional_embedding_1/pos_emb:0': (3249, 8)}

>>> plt.imshow(aa.relative_position_index)
"""
