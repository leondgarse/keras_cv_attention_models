from keras_cv_attention_models.fastervit.fastervit import FasterViT, FasterViT0, FasterViT1, FasterViT2, FasterViT3, FasterViT4, FasterViT5, FasterViT6

__head_doc__ = """
Keras implementation of [Github NVlabs/FasterViT](https://github.com/NVlabs/FasterViT).
Paper [PDF 2306.06189 FasterViT: Fast Vision Transformers with Hierarchical Attention](https://arxiv.org/pdf/2306.06189.pdf).
"""

__tail_doc__ = """  window_ratios: window split ratio. Each stack will calculate `window_size = (height // window_ratio, width // window_ratio)` .
  layer_scale: layer scale init value. Default `-1` means not applying, any value `>=0` will add a scale value for each block output.
      [Going deeper with Image Transformers](https://arxiv.org/pdf/2103.17239.pdf).
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

FasterViT.__doc__ = __head_doc__ + """
Args:
  num_blocks: number of blocks in each stack.
  num_heads: num heads for each stack.
  embed_dim: basic hidden dims, expand * 2 for each stack.
  mlp_ratio: expand ratio for mlp blocks hidden channel.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model      | Params   | FLOPs   | Input | Top1 Acc |
  | ---------- | -------- | ------- | ----- | -------- |
  | FasterViT0 | 31.40M   | 3.51G   | 224   | 82.1     |
  | FasterViT1 | 53.37M   | 5.52G   | 224   | 83.2     |
  | FasterViT2 | 75.92M   | 9.00G   | 224   | 84.2     |
  | FasterViT3 | 159.55M  | 18.75G  | 224   | 84.9     |
  | FasterViT4 | 351.12M  | 41.57G  | 224   | 85.4     |
  | FasterViT5 | 957.52M  | 114.08G | 224   | 85.6     |
  | FasterViT6 | 1360.33M | 144.13G | 224   | 85.8     |
"""

FasterViT0.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

FasterViT1.__doc__ = FasterViT0.__doc__
FasterViT2.__doc__ = FasterViT0.__doc__
FasterViT3.__doc__ = FasterViT0.__doc__
FasterViT4.__doc__ = FasterViT0.__doc__
FasterViT5.__doc__ = FasterViT0.__doc__
FasterViT6.__doc__ = FasterViT0.__doc__
