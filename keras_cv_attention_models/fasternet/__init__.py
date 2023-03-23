from keras_cv_attention_models.fasternet.fasternet import FasterNet, FasterNetT0, FasterNetT1, FasterNetT2, FasterNetS, FasterNetM, FasterNetL

__head_doc__ = """
Keras implementation of [Github JierunChen/FasterNet](https://github.com/JierunChen/FasterNet).
Paper [PDF 2303.03667 Run, Don’t Walk: Chasing Higher FLOPS for Faster Neural Networks ](https://arxiv.org/pdf/2303.03667.pdf).
"""

__tail_doc__ = """  window_ratios: window split ratio. Each stack will calculate `window_size = (height // window_ratio, width // window_ratio)` .
  layer_scale: layer scale init value. Default `-1` means not applying, any value `>=0` will add a scale value for each block output.
      [Going deeper with Image Transformers](https://arxiv.org/pdf/2103.17239.pdf).
  head_init_scale: init head layer scale value if create model from scratch or fine-tune. Default `1`.
  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation used in whole model, default `gelu`.
  drop_connect_rate: is used for [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382).
      Can be value like `0.2`, indicates the drop probability linearly changes from `0 --> 0.2` for `top --> bottom` layers.
      A higher value means a higher probability will drop the deep branch.
      or `0` to disable (default).
  dropout: dropout rate if top layers is included.
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
  pretrained: one of None or "imagenet".
      Will try to download and load pre-trained model weights if not None.

Returns:
    A `keras.Model` instance.
"""

FasterNet.__doc__ = __head_doc__ + """
Args:
  num_blocks: number of blocks in each stack.
  embed_dim: basic hidden dims, expand * 2 for each stack.
  patch_size: int value for stem kernel size and strides.
  mlp_ratio: expand ratio for mlp blocks hidden channel.
  partial_conv_ratio: float value for partial channles appling `Conv2D` in each block.
  output_conv_filter: int value for filters of `Conv2D` block before output block.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model       | Params | FLOPs  | Input | Top1 Acc |
  | ----------- | ------ | ------ | ----- | -------- |
  | FasterNetT0 | 3.9M   | 0.34G  | 224   | 71.9     |
  | FasterNetT1 | 7.6M   | 0.85G  | 224   | 76.2     |
  | FasterNetT2 | 15.0M  | 1.90G  | 224   | 78.9     |
  | FasterNetS  | 31.1M  | 4.55G  | 224   | 81.3     |
  | FasterNetM  | 53.5M  | 8.72G  | 224   | 83.0     |
  | FasterNetL  | 93.4M  | 15.49G | 224   | 83.5     |
"""

FasterNetT0.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

FasterNetT1.__doc__ = FasterNetT0.__doc__
FasterNetT2.__doc__ = FasterNetT0.__doc__
FasterNetS.__doc__ = FasterNetT0.__doc__
FasterNetM.__doc__ = FasterNetT0.__doc__
FasterNetL.__doc__ = FasterNetT0.__doc__
