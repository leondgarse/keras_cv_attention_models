from keras_cv_attention_models.volo.volo import VOLO, VOLO_d1, VOLO_d2, VOLO_d3, VOLO_d4, VOLO_d5, outlook_attention, outlook_attention_simple

__head_doc__ = """
Keras implementation of [Github sail-sg/volo](https://github.com/sail-sg/volo).
Paper [PDF 2106.13112 VOLO: Vision Outlooker for Visual Recognition](https://arxiv.org/pdf/2106.13112.pdf).
"""

__tail_doc__ = """  input_shape: it should have exactly 3 inputs channels like `(224, 224, 3)`.
      Mostly pre-trained input resolution is in [224, 384, 448, 512].
  survivals: is used for [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382).
      Can be value like `0.5` or `0.8`, indicates the survival probability linearly changes from `1 --> 0.8` for `top --> bottom` layers.
      A higher value means a higher probability will keep the deep branch.
  classfiers: number of classfier attension layers.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  pretrained: one of `None` (random initialization) or 'imagenet' (pre-training on ImageNet).
      Will try to download and load pre-trained model weights if not None.
  mix_token: set True for training using `mix_token`. Should better with `token_label_top=True`.
  token_classifier_top: set True for using tokens[0] for classify when `mix_token==False`.
  mean_classifier_top: set True for using all tokens for classify when `mix_token==False`.
  token_label_top: Set True for returning both `token_head` and `aux_head`.
      If `token_classifier_top` / `mean_classifier_top` / `token_label_top` all `False`, will return head layer as evaluation,
      which is the default behavior.
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
  patch_size: model architecture parameter, patch size extracted from input.
""" + __tail_doc__ + """
Model architectures:
  | Model        | params | Image  resolution | Top1 Acc |
  | ------------ | ------ | ----------------- | -------- |
  | volo_d1      | 27M    | 224               | 84.2     |
  | volo_d1 ↑384 | 27M    | 384               | 85.2     |
  | volo_d2      | 59M    | 224               | 85.2     |
  | volo_d2 ↑384 | 59M    | 384               | 86.0     |
  | volo_d3      | 86M    | 224               | 85.4     |
  | volo_d3 ↑448 | 86M    | 448               | 86.3     |
  | volo_d4      | 193M   | 224               | 85.7     |
  | volo_d4 ↑448 | 193M   | 448               | 86.8     |
  | volo_d5      | 296M   | 224               | 86.1     |
  | volo_d5 ↑448 | 296M   | 448               | 87.0     |
  | volo_d5 ↑512 | 296M   | 512               | 87.1     |
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

As the name `fold_overlap_1` indicates, works only if overlap happens once in fold, like `kernel_size=3, strides=2`.
For `kernel_size=3, strides=1`, overlap happens twice, will NOT work...

Examples:

>>> from keras_cv_attention_models import attention_layers
>>> inputs = keras.layers.Input([28, 28, 192])
>>> nn = attention_layers.outlook_attention(inputs, 4, 192)
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
"""
