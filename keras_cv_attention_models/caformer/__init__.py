from keras_cv_attention_models.caformer.caformer import (
    CAFormer,
    CAFormerS18,
    CAFormerS36,
    CAFormerM36,
    CAFormerB36,
    ConvFormerS18,
    ConvFormerS36,
    ConvFormerM36,
    ConvFormerB36,
)

__head_doc__ = """
Keras implementation of [Github sail-sg/metaformer](https://github.com/sail-sg/metaformer).
Paper [PDF 2210.13452 MetaFormer Baselines for Vision](https://arxiv.org/pdf/2210.13452.pdf).

`CAFormer` is using 2 transformer stacks by `block_types=["conv", "conv", "transform", "transform"]`.
`ConvFormer` is all conv blocks by `block_types=["conv", "conv", "conv", "conv"]`.
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
  residual_scales: int or list of int, indicates residual short branch scale init value for each stack.
      Default `[0, 0, 1, 1]`, 0 for not using.
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
      Default is `None`.
  pretrained: one of `None` (random initialization) or 'imagenet' or 'imagenet21k-ft1k' (pre-training on ImageNet21k and fine-tuned ImageNet).
      Will try to download and load pre-trained model weights if not None.

Returns:
    A `keras.Model` instance.
"""

CAFormer.__doc__ = __head_doc__ + """
Args:
  num_blocks: number of blocks in each stack.
  out_channels: output channels for each stack.
  block_types: block types for each stack,
      - `conv` or any `c` / `C` starts word, means `mlp_block_with_depthwise_conv` block.
      - `transfrom` or any `t` / `T` starts word, means `multi_head_self_attention` block.
      value could be in format like `"cctt"` or `"CCTT"` or `["conv", "conv", "transfrom", "transform"]`.
  head_dim: heads number for transformer block.
  mlp_ratios: int or list value indicates expand ratio for mlp blocks hidden channel in each stack.
  head_filter: numer of feature block channels before output block. 0 for not using.
  head_filter_activation: feature block activation. None for same with activation.
  num_attn_low_heads: int or list value indicates attention heads number for `attention_low_frequency_mixer` blocks in each stack.
  pool_sizes: int or list value indicates attention blocks key_value downsample rate in each stack.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model              | Params | FLOPs | Input | Top1 Acc |
  | ------------------ | ------ | ----- | ----- | -------- |
  | CAFormerS18        | 26M    | 4.1G  | 224   | 83.6     |
  | - imagenet21k-ft1k | 26M    | 4.1G  | 224   | 84.1     |
  |                    | 26M    | 13.4G | 384   | 85.0     |
  | - imagenet21k-ft1k | 26M    | 13.4G | 384   | 85.4     |
  | CAFormerS36        | 39M    | 8.0G  | 224   | 84.5     |
  | - imagenet21k-ft1k | 39M    | 8.0G  | 224   | 85.8     |
  |                    | 39M    | 26.0G | 384   | 85.7     |
  | - imagenet21k-ft1k | 39M    | 26.0G | 384   | 86.9     |
  | CAFormerM36        | 56M    | 13.2G | 224   | 85.2     |
  | - imagenet21k-ft1k | 56M    | 13.2G | 224   | 86.6     |
  |                    | 56M    | 42.0G | 384   | 86.2     |
  | - imagenet21k-ft1k | 56M    | 42.0G | 384   | 87.5     |
  | CAFormerB36        | 99M    | 23.2G | 224   | 85.5     |
  | - imagenet21k-ft1k | 99M    | 23.2G | 224   | 87.4     |
  |                    | 99M    | 72.2G | 384   | 86.4     |
  | - imagenet21k-ft1k | 99M    | 72.2G | 384   | 88.1     |

  | Model              | Params | FLOPs | Input | Top1 Acc |
  | ------------------ | ------ | ----- | ----- | -------- |
  | ConvFormerS18      | 27M    | 3.9G  | 224   | 83.0     |
  | - imagenet21k-ft1k | 27M    | 3.9G  | 224   | 83.7     |
  |                    | 27M    | 11.6G | 384   | 84.4     |
  | - imagenet21k-ft1k | 27M    | 11.6G | 384   | 85.0     |
  | ConvFormerS36      | 40M    | 7.6G  | 224   | 84.1     |
  | - imagenet21k-ft1k | 40M    | 7.6G  | 224   | 85.4     |
  |                    | 40M    | 22.4G | 384   | 85.4     |
  | - imagenet21k-ft1k | 40M    | 22.4G | 384   | 86.4     |
  | ConvFormerM36      | 57M    | 12.8G | 224   | 84.5     |
  | - imagenet21k-ft1k | 57M    | 12.8G | 224   | 86.1     |
  |                    | 57M    | 37.7G | 384   | 85.6     |
  | - imagenet21k-ft1k | 57M    | 37.7G | 384   | 86.9     |
  | ConvFormerB36      | 100M   | 22.6G | 224   | 84.8     |
  | - imagenet21k-ft1k | 100M   | 22.6G | 224   | 87.0     |
  |                    | 100M   | 66.5G | 384   | 85.7     |
  | - imagenet21k-ft1k | 100M   | 66.5G | 384   | 87.6     |
"""

CAFormerS18.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

CAFormerS36.__doc__ = CAFormerS18.__doc__
CAFormerM36.__doc__ = CAFormerS18.__doc__
CAFormerB36.__doc__ = CAFormerS18.__doc__
ConvFormerS18.__doc__ = CAFormerS18.__doc__
ConvFormerS36.__doc__ = CAFormerS18.__doc__
ConvFormerM36.__doc__ = CAFormerS18.__doc__
ConvFormerB36.__doc__ = CAFormerS18.__doc__
