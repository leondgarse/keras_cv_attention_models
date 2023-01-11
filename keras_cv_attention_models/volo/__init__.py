from keras_cv_attention_models.volo.volo import (
    VOLO,
    VOLO_d1,
    VOLO_d2,
    VOLO_d3,
    VOLO_d4,
    VOLO_d5,
    outlook_attention,
    outlook_attention_simple,
    BiasLayer,
    PositionalEmbedding,
    ClassToken,
)

__head_doc__ = """
Keras implementation of [Github sail-sg/volo](https://github.com/sail-sg/volo).
Paper [PDF 2106.13112 VOLO: Vision Outlooker for Visual Recognition](https://arxiv.org/pdf/2106.13112.pdf).
"""

__tail_doc__ = """  patch_size: model architecture parameter, patch size extracted from input.
  input_shape: it should have exactly 3 inputs channels like `(224, 224, 3)`.
      Mostly pre-trained input resolution is in [224, 384, 448, 512].
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  drop_connect_rate: is used for [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382).
      Can be value like `0.2`, indicates the drop probability linearly changes from `0 --> 0.2` for `top --> bottom` layers.
      A higher value means a higher probability will drop the deep branch.
      or `0` to disable (default).
  classfiers: number of classfier attension layers.
      Will try to download and load pre-trained model weights if not None.
  mix_token: set True for training using `mix_token`. Should better with `token_label_top=True`.
  token_classifier_top: set True for using tokens[0] for classify when `mix_token==False`.
  mean_classifier_top: set True for using all tokens for classify when `mix_token==False`.
  token_label_top: Set True for returning both `token_head` and `aux_head`.
      If `token_classifier_top` / `mean_classifier_top` / `token_label_top` all `False`, will return head layer as evaluation,
      which is the default behavior.
  pretrained: one of `None` (random initialization) or 'imagenet' (pre-training on ImageNet).
  **kwargs: other parameters if available.

Returns:
    A `keras.Model` instance.
"""

VOLO.__doc__ = __head_doc__ + """
Args:
  num_blocks: model architecture parameter, number of each blocks.
  embed_dims: model architecture parameter, hidden dim for each blocks.
  num_heads: model architecture parameter, num heads for each blocks.
  mlp_ratios: model architecture parameter, mlp width expansion.
  stem_hidden_dim: model architecture parameter, 128 for VOLO_d5, 64 for others.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model   | Params | FLOPs   | Input | Top1 Acc |
  | ------- | ------ | ------- | ----- | -------- |
  | VOLO_d1 | 27M    | 4.82G   | 224   | 84.2     |
  | - 384   | 27M    | 14.22G  | 384   | 85.2     |
  | VOLO_d2 | 59M    | 9.78G   | 224   | 85.2     |
  | - 384   | 59M    | 28.84G  | 384   | 86.0     |
  | VOLO_d3 | 86M    | 13.80G  | 224   | 85.4     |
  | - 448   | 86M    | 55.50G  | 448   | 86.3     |
  | VOLO_d4 | 193M   | 29.39G  | 224   | 85.7     |
  | - 448   | 193M   | 117.81G | 448   | 86.8     |
  | VOLO_d5 | 296M   | 53.34G  | 224   | 86.1     |
  | - 448   | 296M   | 213.72G | 448   | 87.0     |
  | - 512   | 296M   | 279.36G | 512   | 87.1     |
"""

VOLO_d1.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

VOLO_d2.__doc__ = VOLO_d1.__doc__
VOLO_d3.__doc__ = VOLO_d1.__doc__
VOLO_d4.__doc__ = VOLO_d1.__doc__
VOLO_d5.__doc__ = VOLO_d1.__doc__

outlook_attention.__doc__ = __head_doc__ + """
outlook attention. Callable function, NOT defined as a layer.
Extract patches with a `kernel_size` from `value` as an enlarged attention area, then matmul with `attention_scores` and fold back.

Args:
  inputs: input tensor.
  embed_dim: hidden dim for attention, value and ouput.
  num_head: number of attention heads.
  kernel_size: kernel size for extracting patches from input and calculating attention weights.
      For extracting patches from input, it's simmilar with `Conv2D`.
      For calculating attention weights, it affects the attention output shape.
  padding: set if apply padding, simmilar with `Conv2D`.
  strides: simmilar with `Conv2D`. If it's smaller then `kernel_size`, will have overlap area.
      Currently only works if overlap happened once, like `kernel_size=3, strides=2`.
  attn_dropout: dropout probability for attention.
  output_dropout: dropout probability for output.

Examples:

>>> from keras_cv_attention_models import attention_layers
>>> inputs = keras.layers.Input([28, 28, 192])
>>> nn = attention_layers.outlook_attention(inputs, embed_dim=192, num_head=4)
>>> cc = keras.models.Model(inputs, nn)
>>> cc.summary()
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_2 (InputLayer)            [(None, 28, 28, 192) 0
__________________________________________________________________________________________________
average_pooling2d_1 (AveragePoo (None, 14, 14, 192)  0           input_2[0][0]
__________________________________________________________________________________________________
attn (Dense)                    (None, 14, 14, 324)  62532       average_pooling2d_1[0][0]
__________________________________________________________________________________________________
tf.math.truediv_1 (TFOpLambda)  (None, 14, 14, 324)  0           attn[0][0]
__________________________________________________________________________________________________
tf.reshape_1 (TFOpLambda)       (None, 14, 14, 4, 9, 0           tf.math.truediv_1[0][0]
__________________________________________________________________________________________________
v (Dense)                       (None, 28, 28, 192)  36864       input_2[0][0]
__________________________________________________________________________________________________
tf.nn.softmax_1 (TFOpLambda)    (None, 14, 14, 4, 9, 0           tf.reshape_1[0][0]
__________________________________________________________________________________________________
unfold_matmul_fold_1 (UnfoldMat (None, 28, 28, 192)  0           v[0][0]
                                                               tf.nn.softmax_1[0][0]
__________________________________________________________________________________________________
out (Dense)                     (None, 28, 28, 192)  37056       unfold_matmul_fold_1[0][0]
==================================================================================================
Total params: 136,452
Trainable params: 136,452
Non-trainable params: 0
__________________________________________________________________________________________________
"""

outlook_attention_simple.__doc__ = __head_doc__ + """
A simplidied version of outlook attention that not using unfold and fold. Callable function, NOT defined as a layer.

Args:
  inputs: input tensor.
  embed_dim: hidden dim for attention, value and ouput.
  num_head: number of attention heads.
  kernel_size: kernel size for extracting patches from input and calculating attention weights.
      For extracting patches from input, it's simmilar with `Conv2D`.
      For calculating attention weights, it affects the attention output shape.
  attn_dropout: dropout probability for attention.

Examples:

>>> from keras_cv_attention_models import attention_layers
>>> inputs = keras.layers.Input([28, 28, 192])
>>> nn = attention_layers.outlook_attention_simple(inputs, embed_dim=192, num_head=4)
>>> cc = keras.models.Model(inputs, nn)
>>> cc.summary()
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to
==================================================================================================
input_1 (InputLayer)            [(None, 28, 28, 192) 0
__________________________________________________________________________________________________
zero_padding2d (ZeroPadding2D)  (None, 30, 30, 192)  0           input_1[0][0]
__________________________________________________________________________________________________
average_pooling2d (AveragePooli (None, 10, 10, 192)  0           zero_padding2d[0][0]
__________________________________________________________________________________________________
attn (Dense)                    (None, 10, 10, 324)  62532       average_pooling2d[0][0]
__________________________________________________________________________________________________
v (Dense)                       (None, 30, 30, 192)  36864       zero_padding2d[0][0]
__________________________________________________________________________________________________
tf.math.truediv (TFOpLambda)    (None, 10, 10, 324)  0           attn[0][0]
__________________________________________________________________________________________________
tf.reshape (TFOpLambda)         (None, 10, 3, 10, 3, 0           v[0][0]
__________________________________________________________________________________________________
tf.reshape_2 (TFOpLambda)       (None, 10, 10, 4, 9, 0           tf.math.truediv[0][0]
__________________________________________________________________________________________________
tf.compat.v1.transpose (TFOpLam (None, 10, 10, 4, 3, 0           tf.reshape[0][0]
__________________________________________________________________________________________________
tf.nn.softmax (TFOpLambda)      (None, 10, 10, 4, 9, 0           tf.reshape_2[0][0]
__________________________________________________________________________________________________
tf.reshape_1 (TFOpLambda)       (None, 10, 10, 4, 9, 0           tf.compat.v1.transpose[0][0]
__________________________________________________________________________________________________
tf.linalg.matmul (TFOpLambda)   (None, 10, 10, 4, 9, 0           tf.nn.softmax[0][0]
                                                                 tf.reshape_1[0][0]
__________________________________________________________________________________________________
tf.reshape_3 (TFOpLambda)       (None, 10, 10, 4, 3, 0           tf.linalg.matmul[0][0]
__________________________________________________________________________________________________
tf.compat.v1.transpose_1 (TFOpL (None, 10, 3, 10, 3, 0           tf.reshape_3[0][0]
__________________________________________________________________________________________________
tf.reshape_4 (TFOpLambda)       (None, 30, 30, 192)  0           tf.compat.v1.transpose_1[0][0]
__________________________________________________________________________________________________
tf.__operators__.getitem (Slici (None, 28, 28, 192)  0           tf.reshape_4[0][0]
__________________________________________________________________________________________________
out (Dense)                     (None, 28, 28, 192)  37056       tf.__operators__.getitem[0][0]
==================================================================================================
Total params: 136,452
Trainable params: 136,452
Non-trainable params: 0
__________________________________________________________________________________________________
"""

ClassToken.__doc__ = __head_doc__ + """
Attach class token on head.

input: `[batch, blocks, channel]`
output: `[batch, 1 * class_token + blocks, channel]`

Examples:

>>> from keras_cv_attention_models import attention_layers
>>> aa = attention_layers.ClassToken()
>>> print(f"{aa(tf.ones([2, 14 * 14, 192])).shape = }")
aa(tf.ones([2, 14 * 14, 192])).shape = TensorShape([2, 197, 192])

>>> print({ii.name:ii.numpy().shape for ii in aa.weights})
{'class_token/tokens:0': (1, 1, 192)}
"""

BiasLayer.__doc__ = __head_doc__ + """
Bias only layer on channel dimension.

input: `[..., channel]`
output: `[..., channel]`

Examples:

>>> from keras_cv_attention_models import attention_layers
>>> aa = attention_layers.BiasLayer()
>>> print(f"{aa(tf.ones([2, 14, 14, 192])).shape = }")
aa(tf.ones([2, 14, 14, 192])).shape = TensorShape([2, 14, 14, 192])

>>> print({ii.name:ii.numpy().shape for ii in aa.weights})
{'bias_layer/bias:0': (192,)}
"""

PositionalEmbedding.__doc__ = __head_doc__ + """
Absolute Positional embedding layer.
Positional embedding shape is `[1, height, width, channel]`, then adds directly with input.

input: `[batch, height, width, channel]`
output: `[batch, height, width, channel]`

Args:
  input_height: specify `height` for inputs if not square `input_height != input_width`.

Examples:

>>> from keras_cv_attention_models import attention_layers
>>> aa = attention_layers.PositionalEmbedding()
>>> print(f"{aa(tf.ones([2, 14, 14, 192])).shape = }")
aa(tf.ones([2, 14, 14, 192])).shape = TensorShape([2, 14, 14, 192])

>>> print({ii.name:ii.numpy().shape for ii in aa.weights})
{'positional_embedding/positional_embedding:0': (1, 14, 14, 192)}
"""
