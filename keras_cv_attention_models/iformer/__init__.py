from keras_cv_attention_models.iformer.iformer import InceptionTransformer, IFormerSmall, IFormerBase, IFormerLarge

__head_doc__ = """
Keras implementation of [Github sail-sg/iFormer](https://github.com/sail-sg/iFormer).
Paper [PDF 2205.12956 Inception Transformer](https://arxiv.org/pdf/2205.12956.pdf).
"""

__tail_doc__ = """  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation used in whole model, default `gelu`.
  drop_connect_rate: is used for [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382).
      Can be a constant value like `0.2`,
      or a tuple value like `(0, 0.2)` indicates the drop probability linearly changes from `0 --> 0.2` for `top --> bottom` layers.
      A higher value means a higher probability will drop the deep branch.
      or `0` to disable (default).
  layer_scales: int or list of int, indicates layer scale init value for each stack. Default `[0, 0, 1e-6, 1e-6]`, 0 for not using.
      [Going deeper with Image Transformers](https://arxiv.org/abs/2103.17239).
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
      Default is `None`.
  pretrained: one of `None` (random initialization) or 'imagenet21k-ft1k' (pre-training on ImageNet21k and fine-tuned ImageNet).
      Will try to download and load pre-trained model weights if not None.

Returns:
    A `keras.Model` instance.
"""

InceptionTransformer.__doc__ = __head_doc__ + """
Args:
  num_blocks: number of blocks in each stack.
  embed_dims: output channels for each stack.
  num_heads: int or list value indicates heads number for `conv_attention_mixer` blocks in each stack.
  num_attn_low_heads: int or list value indicates attention heads number for `attention_low_frequency_mixer` blocks in each stack.
  pool_sizes: int or list value indicates attention blocks key_value downsample rate in each stack.
  mlp_ratios: int or list value indicates expand ratio for mlp blocks hidden channel in each stack.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model        | Params | FLOPs  | Input | Top1 Acc |
  | ------------ | ------ | ------ | ----- | -------- |
  | IFormerSmall | 19.9M  | 4.88G  | 224   | 83.4     |
  |              | 20.9M  | 16.29G | 384   | 84.6     |
  | IFormerBase  | 47.9M  | 9.44G  | 224   | 84.6     |
  |              | 48.9M  | 30.86G | 384   | 85.7     |
  | IFormerLarge | 86.6M  | 14.12G | 224   | 84.6     |
  |              | 87.7M  | 45.74G | 384   | 85.8     |
"""

IFormerSmall.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

IFormerBase.__doc__ = IFormerSmall.__doc__
IFormerLarge.__doc__ = IFormerSmall.__doc__
