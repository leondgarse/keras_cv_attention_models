from keras_cv_attention_models.inceptionnext.inceptionnext import InceptionNeXt, InceptionNeXtTiny, InceptionNeXtSmall, InceptionNeXtBase

__head_doc__ = """
Keras implementation of [Github sail-sg/inceptionnext](https://github.com/sail-sg/inceptionnext).
Paper [PDF 2303.16900 InceptionNeXt: When Inception Meets ConvNeXt](https://arxiv.org/pdf/2303.16900.pdf).
"""

__tail_doc__ = """  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation used in whole model, default `gelu`.
  drop_connect_rate: is used for [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382).
      Can be a constant value like `0.2`,
      or a tuple value like `(0, 0.2)` indicates the drop probability linearly changes from `0 --> 0.2` for `top --> bottom` layers.
      A higher value means a higher probability will drop the deep branch.
      or `0` to disable (default).
  layer_scale: int value indicates layer scale init value for each stack. Default `[0, 0, 1e-6, 1e-6]`, 0 for not using.
      [Going deeper with Image Transformers](https://arxiv.org/abs/2103.17239).
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
      Default is `None`.
  pretrained: one of `None` (random initialization) or 'imagenet21k-ft1k' (pre-training on ImageNet21k and fine-tuned ImageNet).
      Will try to download and load pre-trained model weights if not None.

Returns:
    A `keras.Model` instance.
"""

InceptionNeXt.__doc__ = __head_doc__ + """
Args:
  num_blocks: number of blocks in each stack.
  embed_dims: output channels for each stack.
  mlp_ratios: int or list value indicates expand ratio for mlp blocks hidden channel in each stack.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model              | Params | FLOP s | Input | Top1 Acc |
  | ------------------ | ------ | ------ | ----- | -------- |
  | InceptionNeXtTiny  | 28.05M | 4.21G  | 224   | 82.3     |
  | InceptionNeXtSmall | 49.37M | 8.39G  | 224   | 83.5     |
  | InceptionNeXtBase  | 86.67M | 14.88G | 224   | 84.0     |
  |                    | 86.67M | 43.73G | 384   | 85.2     |
"""

InceptionNeXtTiny.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

InceptionNeXtSmall.__doc__ = InceptionNeXtTiny.__doc__
InceptionNeXtBase.__doc__ = InceptionNeXtTiny.__doc__
