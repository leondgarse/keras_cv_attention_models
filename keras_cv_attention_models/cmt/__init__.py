from keras_cv_attention_models.cmt.cmt import CMT, CMTTiny, CMTXS, CMTSmall, CMTBig

__head_doc__ = """
Keras implementation of CMT.
Paper [PDF 2107.06263 CMT: Convolutional Neural Networks Meet Vision Transformers](https://arxiv.org/pdf/2107.06263.pdf).
"""

__tail_doc__ = """  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation used in whole model, default `relu`.
  drop_connect_rate: is used for [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382).
      Can be value like `0.2`, indicates the drop probability linearly changes from `0 --> 0.2` for `top --> bottom` layers.
      A higher value means a higher probability will drop the deep branch.
      or `0` to disable (default).
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
  dropout: dropout rate if top layers is included.
  pretrained: None or "imagenet". Only CMTTiny pretrained available.
  **kwargs: other parameters if available.

Returns:
    A `keras.Model` instance.
"""

CMT.__doc__ = __head_doc__ + """
Args:
  num_blocks: number of blocks in each stack.
  out_channels: output channels for each stack.
  stem_width: output dimension for stem block.
  num_heads: heads number for transformer block.
  sr_ratios: attenstion blocks key_value downsample rate.
  ffn_expansion: IRFFN blocks hidden expansion rate.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model    | Params | Image resolution | Top1 Acc |
  | -------- | ------ | ---------------- | -------- |
  | CMTTiny  | 9.5M   | 160              | 79.2     |
  | CMTXS    | 15.2M  | 192              | 81.8     |
  | CMTSmall | 25.1M  | 224              | 83.5     |
  | CMTBig   | 45.7M  | 256              | 84.5     |
"""

CMTTiny.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

CMTXS.__doc__ = CMTTiny.__doc__
CMTSmall.__doc__ = CMTTiny.__doc__
CMTBig.__doc__ = CMTTiny.__doc__
