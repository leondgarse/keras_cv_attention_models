from keras_cv_attention_models.tinyvit.tinyvit import TinyViT, TinyViT_5M, TinyViT_11M, TinyViT_21M

__head_doc__ = """
Keras implementation of [Github microsoft/TinyViT](https://github.com/microsoft/Cream/tree/main/TinyViT).
Paper [PDF 2207.10666 TinyViT: Fast Pretraining Distillation for Small Vision Transformers](https://arxiv.org/pdf/2207.10666.pdf).
"""

__tail_doc__ = """  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation used in whole model, default `gelu`.
  drop_connect_rate: is used for [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382).
      Can be a constant value like `0.2`,
      or a tuple value like `(0, 0.2)` indicates the drop probability linearly changes from `0 --> 0.2` for `top --> bottom` layers.
      A higher value means a higher probability will drop the deep branch.
      or `0` to disable (default).
  layer_scale: int value indicates layer scale init value for each stack. Default 0 for not using.
      [Going deeper with Image Transformers](https://arxiv.org/abs/2103.17239).
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
      Default is `None`.
  pretrained: one of `None` (random initialization) or 'imagenet' or 'imagenet21k-ft1k' (pre-training on ImageNet21k and fine-tuned ImageNet).
      Will try to download and load pre-trained model weights if not None.

Returns:
    A `keras.Model` instance.
"""

TinyViT.__doc__ = __head_doc__ + """
Args:
  num_blocks: number of blocks in each stack.
  out_channels: output channels for each stack.
  block_types: block types for each stack,
      - `conv` or any `c` / `C` starts word, means `mlp_block_with_depthwise_conv` block.
      - `transfrom` or any `t` / `T` starts word, means `multi_head_self_attention` block.
      value could be in format like `"cctt"` or `"CCTT"` or `["conv", "conv", "transfrom", "transform"]`.
  num_heads: int or list of int value indicates attention heads number for each transformer stack.
  window_ratios: int or list of int value indicates attention heads window ratio number for each transformer stack.
      Actual using `window_size = ceil(cur_input_shape / window_ratio)`.
      For `input_shape=(224, 224, 3)` will be window_sizes=[7, 7, 14, 7], for `(384, 384, 3)` will be `[12, 12, 24, 12]`.
  mlp_ratio: int value indicates expand ratio for mlp blocks hidden channel in each stack.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model                | Params | FLOPs | Input | Top1 Acc |
  | -------------------- | ------ | ----- | ----- | -------- |
  | TinyViT_5M, distill  | 5.4M   | 1.3G  | 224   | 79.1     |
  | - imagenet21k-ft1k   | 5.4M   | 1.3G  | 224   | 80.7     |
  | TinyViT_11M, distill | 11M    | 2.0G  | 224   | 81.5     |
  | - imagenet21k-ft1k   | 11M    | 2.0G  | 224   | 83.2     |
  | TinyViT_21M, distill | 21M    | 4.3G  | 224   | 83.1     |
  | - imagenet21k-ft1k   | 21M    | 4.3G  | 224   | 84.8     |
  |                      | 21M    | 13.8G | 384   | 86.2     |
  |                      | 21M    | 27.0G | 512   | 86.5     |
"""

TinyViT_5M.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

TinyViT_11M.__doc__ = TinyViT_5M.__doc__
TinyViT_21M.__doc__ = TinyViT_5M.__doc__
