from keras_cv_attention_models.maxvit.maxvit import MaxViT, MaxViT_Tiny, MaxViT_Small, MaxViT_Base, MaxViT_Large, MaxViT_XLarge

__head_doc__ = """
Keras implementation of [Github google-research/maxvit](https://github.com/google-research/maxvit).
Paper [PDF 2204.01697 MaxViT: Multi-Axis Vision Transformer](https://arxiv.org/pdf/2204.01697.pdf).
"""

__tail_doc__ = """  strides: int or list of int, for strides in the first block of each stack.
  expansion: filter expansion in each block. The larger the wider.
  se_ratio: value in `(0, 1)`, where `0` means not using `se_module`. Should be a `number` or `list`, indicates `se_ratio` for each stack.
      Each element can also be a `number` or `list`, indicates `se_ratio` for each block.
  head_dimension: heads number for transformer block.
  window_ratio: window_size ratio, window_size = [input_shape[0] // window_ratio, input_shape[1] // window_ratio].
  output_filter: additional `Dense + tanh` block before ouput block. Default -1 for `out_channels[-1]`, 0 to disable.
  use_torch_mode: whether use torch model or not.
      True: use_torch_padding = True, epsilon = 1e-5, momentum = 0.9
      False: use_torch_padding = False, epsilon = 0.001, momentum = 0.99
  layer_scale: layer scale init value. `-1` means not applying, any value `>=0` will add a scale value for each block output.
      [Going deeper with Image Transformers](https://arxiv.org/pdf/2103.17239.pdf). Default -1.
  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
      Actual `num_classes` for `"imagenet21k"` models is `21843`.
  activation: activation used in whole model, default "gelu/app" means `tf.nn.gelu(approximate=True)`.
  drop_connect_rate: is used for [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382).
      Can be value like `0.2`, indicates the drop probability linearly changes from `0 --> 0.2` for `top --> bottom` layers.
      A higher value means a higher probability will drop the deep branch.
      or `0` to disable (default).
  dropout: dropout rate if top layers is included.
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
  pretrained: One of `[None, "imagenet", "imagenet21k", "imagenet21k-ft1k"]`.
  **kwargs: other parameters if available.

Returns:
  A `keras.Model` instance.
"""

MaxViT.__doc__ = __head_doc__ + """
Args:
  num_blocks: number of blocks in each stack.
  out_channels: output channels for each stack.
  stem_width: output dimension for stem block.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model                           | Params | FLOPs  | Input | Top1 Acc |
  | ------------------------------- | ------ | ------ | ----- | -------- |
  | MaxViT_Tiny                     | 31M    | 5.6G   | 224   | 83.62    |
  | MaxViT_Tiny                     | 31M    | 17.7G  | 384   | 85.24    |
  | MaxViT_Tiny                     | 31M    | 33.7G  | 512   | 85.72    |
  | MaxViT_Small                    | 69M    | 11.7G  | 224   | 84.45    |
  | MaxViT_Small                    | 69M    | 36.1G  | 384   | 85.74    |
  | MaxViT_Small                    | 69M    | 67.6G  | 512   | 86.19    |
  | MaxViT_Base                     | 119M   | 24.2G  | 224   | 84.95    |
  | - imagenet21k                   | 135M   | 24.2G  | 224   |          |
  | MaxViT_Base                     | 119M   | 74.2G  | 384   | 86.34    |
  | - imagenet21k-ft1k              | 119M   | 74.2G  | 384   | 88.24    |
  | MaxViT_Base                     | 119M   | 138.5G | 512   | 86.66    |
  | - imagenet21k-ft1k              | 119M   | 138.5G | 512   | 88.38    |
  | MaxViT_Large                    | 212M   | 43.9G  | 224   | 85.17    |
  | - imagenet21k                   | 233M   | 43.9G  | 224   |          |
  | MaxViT_Large                    | 212M   | 133.1G | 384   | 86.40    |
  | - imagenet21k-ft1k              | 212M   | 133.1G | 384   | 88.32    |
  | MaxViT_Large                    | 212M   | 245.4G | 512   | 86.70    |
  | - imagenet21k-ft1k              | 212M   | 245.4G | 512   | 88.46    |
  | MaxViT_XLarge, imagenet21k      | 507M   | 97.7G  | 224   |          |
  | MaxViT_XLarge, imagenet21k-ft1k | 475M   | 293.7G | 384   | 88.51    |
  | MaxViT_XLarge, imagenet21k-ft1k | 475M   | 535.2G | 512   | 88.70    |
"""

MaxViT_Tiny.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

MaxViT_Small.__doc__ = MaxViT_Tiny.__doc__
MaxViT_Base.__doc__ = MaxViT_Tiny.__doc__
MaxViT_Large.__doc__ = MaxViT_Tiny.__doc__
MaxViT_XLarge.__doc__ = MaxViT_Tiny.__doc__
