from keras_cv_attention_models.coat.coat import CoaT, CoaTLiteTiny, CoaTLiteMini, CoaTLiteSmall, CoaTTiny, CoaTMini

__head_doc__ = """
Keras implementation of [Github mlpc-ucsd/CoaT](https://github.com/mlpc-ucsd/CoaT).
Paper [PDF 2104.06399 CoaT: Co-Scale Conv-Attentional Image Transformers](http://arxiv.org/abs/2104.06399).
"""

__tail_doc__ = """  head_splits:
  head_kernel_size:
  use_share_cpe:
  use_share_crpe:
  out_features:
  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation used in whole model, default `relu`.
  drop_connect_rate:
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
  pretrained: None available.
  **kwargs: other parameters if available.

Returns:
    A `keras.Model` instance.
"""

CoaT.__doc__ = __head_doc__ + """
Args:
  serial_depths:
  embed_dims:
  mlp_ratios:
  parallel_depth:
  patch_size:
  num_heads: heads number for transformer block.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model         | Params | Top1 Acc |
  | ------------- | ------ | -------- |
  | CoaTLiteTiny  | 5.7M   | 77.5     |
  | CoaTLiteMini  | 11M    | 79.1     |
  | CoaTLiteSmall | 20M    | 81.9     |
  | CoaTTiny      | 5.5M   | 78.3     |
  | CoaTMini      | 10M    | 81.0     |
"""

CoaTLiteTiny.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

CoaTLiteMini.__doc__ = CoaTLiteTiny.__doc__
CoaTLiteSmall.__doc__ = CoaTLiteTiny.__doc__
CoaTTiny.__doc__ = CoaTLiteTiny.__doc__
CoaTMini.__doc__ = CoaTLiteTiny.__doc__
