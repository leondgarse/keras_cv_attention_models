from keras_cv_attention_models.efficientvit.efficientvit import (
    EfficientViT,
    EfficientViT_M0,
    EfficientViT_M1,
    EfficientViT_M2,
    EfficientViT_M3,
    EfficientViT_M4,
    EfficientViT_M5,
    grouped_mhsa_with_multi_head_position,
)

__head_doc__ = """
Keras implementation of [Github microsoft/Cream/EfficientViT/classification](https://github.com/microsoft/Cream/tree/main/EfficientViT/classification).
Paper [PDF 2205.14756 EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention](https://arxiv.org/pdf/2205.14756.pdf).
"""

__tail_doc__ = """  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation used in whole model, default `relu`.
  drop_connect_rate: is used for [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382).
      Can be a constant value like `0.2`,
      or a tuple value like `(0, 0.2)` indicates the drop probability linearly changes from `0 --> 0.2` for `top --> bottom` layers.
      A higher value means a higher probability will drop the deep branch.
      or `0` to disable (default).
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
      Default is `None`.
  pretrained: one of `None` (random initialization) or 'imagenet'.
      Will try to download and load pre-trained model weights if not None.

Returns:
    A `keras.Model` instance.
"""

EfficientViT.__doc__ = __head_doc__ + """
Args:
  num_blocks: number of blocks in each stack.
  out_channels: output channels for each stack.
  window_size: int value indicates window partition size for attention blocks.
  num_heads: int or list value indicates num_heads for attention blocks in each stack.
  key_dim: int value indicates key dim for attention blocks query and key.
  kernel_sizes: int or list value indicates kernel_size for each head in attention blocks. Length must larger than max of num_heads.
  mlp_ratio: int value indicates expand ratio for mlp blocks hidden channels.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model           | Params | FLOPs | Input | Top1 Acc |
  | --------------- | ------ | ----- | ----- | -------- |
  | EfficientViT_M0 | 2.35M  | 794M  | 224   | 63.2     |
  | EfficientViT_M1 | 2.98M  | 167M  | 224   | 68.4     |
  | EfficientViT_M2 | 4.19M  | 201M  | 224   | 70.8     |
  | EfficientViT_M3 | 6.90M  | 263M  | 224   | 73.4     |
  | EfficientViT_M4 | 8.80M  | 299M  | 224   | 74.3     |
  | EfficientViT_M5 | 12.47M | 522M  | 224   | 77.1     |
"""

EfficientViT_M0.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

EfficientViT_M1.__doc__ = EfficientViT_M0.__doc__
EfficientViT_M2.__doc__ = EfficientViT_M0.__doc__
EfficientViT_M3.__doc__ = EfficientViT_M0.__doc__
EfficientViT_M4.__doc__ = EfficientViT_M0.__doc__
EfficientViT_M5.__doc__ = EfficientViT_M0.__doc__

grouped_mhsa_with_multi_head_position.__doc__ = __head_doc__ + """
Args:
 [TODO]
"""
