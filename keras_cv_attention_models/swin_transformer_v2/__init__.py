from keras_cv_attention_models.swin_transformer_v2.swin_transformer_v2 import (
    DivideScale,
    PairWiseRelativePositionalEmbedding,
    window_multi_head_self_attention,
    SwinTransformerV2,
    SwinTransformerV2Tiny,
    SwinTransformerV2Tiny_ns,
    SwinTransformerV2Small,
    SwinTransformerV2Base,
    SwinTransformerV2Large,
)

__head_doc__ = """
Keras implementation of [Github timm/swin_transformer_v2_cr](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/swin_transformer_v2_cr.py).
Paper [PDF 2111.09883 Swin Transformer V2: Scaling Up Capacity and Resolution](https://arxiv.org/pdf/2111.09883.pdf).
"""

__tail_doc__ = """  window_ratio: window_size ratio, window_size = [input_shape[0] // window_ratio, input_shape[1] // window_ratio].
  stem_patch_size: stem patch size for stem kernel_size and strides.
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

SwinTransformerV2.__doc__ = __head_doc__ + """
Args:
  num_blocks: number of blocks in each stack.
  num_heads: num heads for each stack.
  embed_dim: basic hidden dims, expand * 2 for each stack.
  use_stack_norm: boolean value if apply an additional layer_norm after each stack.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model                       | Params | Image resolution | Top1 Acc |
  | --------------------------- | ------ | ---------------- | -------- |
  | SwinTransformerV2Tiny_ns    | 28.3M  | 224              | 81.8     |
  | SwinTransformerV2Small      | 49.7M  | 224              | 83.13    |
  | SwinTransformerV2Base, 22k  | 87.9M  | 384              | 87.1     |
  | SwinTransformerV2Large, 22k | 196.7M | 384              | 87.7     |
"""

SwinTransformerV2Tiny.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

SwinTransformerV2Tiny_ns.__doc__ = SwinTransformerV2Tiny.__doc__
SwinTransformerV2Small.__doc__ = SwinTransformerV2Tiny.__doc__
SwinTransformerV2Base.__doc__ = SwinTransformerV2Tiny.__doc__
SwinTransformerV2Large.__doc__ = SwinTransformerV2Tiny.__doc__
