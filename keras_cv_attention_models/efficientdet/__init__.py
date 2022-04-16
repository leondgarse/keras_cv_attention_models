from keras_cv_attention_models.efficientdet.efficientdet import (
    EfficientDet,
    EfficientDetD0,
    EfficientDetD1,
    EfficientDetD2,
    EfficientDetD3,
    EfficientDetD4,
    EfficientDetD5,
    EfficientDetD6,
    EfficientDetD7,
    EfficientDetD7X,
    EfficientDetLite0,
    EfficientDetLite1,
    EfficientDetLite2,
    EfficientDetLite3,
    EfficientDetLite3X,
    EfficientDetLite4,
)

__head_doc__ = """
Keras implementation of [google/automl/efficientdet](https://github.com/google/automl/tree/master/efficientdet).
Paper [Paper 1911.09070 EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/pdf/1911.09070.pdf).

Args:
  backbone: backbone model, could be any model with pyramid stage structure. {}
"""

__tail_doc__ = """  anchors_mode: one of ["efficientdet", "anchor_free", "yolor"], controls which anchor to use.
      - efficientdet anchors default settings: use_object_scores=False, num_anchors=9, anchor_scale=4,
          aspect_ratios=[1, 2, 0.5], num_scales=3, grid_zero_start=False.
      - anchor_free default settings: use_object_scores=True, num_anchors=1, anchor_scale=1,
          aspect_ratios=[1], num_scales=1, grid_zero_start=True.
      - yolor default settings: use_object_scores=True, num_anchors=3.
      Default "efficientdet".
  num_anchors: number of anchors for a single grid point, should be same with dataset used value.
      Default "auto" means: anchors_mode=="anchor_free" -> 1, anchors_mode=="yolor" -> 3, else 9.
  use_object_scores: bollean value if model header output includes `object_scores`.
      Default "auto" means: True if anchors_mode=="anchor_free" or anchors_mode=="yolor", else False.
  num_classes: total output classes. `90` for EfficientDet pretrained, `80` for `tfds.coco`.
      Set `0` to disable `classifier` output.
  use_sep_conv: set `False` for using `Conv2D` instead of `SeparableConv2D`.
  activation: activation used in whole model, default `swish`. Default "swish".
  classifier_activation: The activation function to use for classifier output if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer. Default `sigmoid`.
  freeze_backbone: set `True` for `backbone.trainable = False`. Default `False`.
  pretrained: one of `None` (random initialization) or 'coco' (pre-training on COCO).
      Will try to download and load pre-trained model weights if not None. Default `coco`.
  pyramid_levels_min: anchors inititial parameter for model prediction, not affecting model architecture. Default `3`.
      pyramid_levels_max is calculated as `pyramid_levels_min + len(features_pick) + additional_features - 1`.
  anchor_scale: anchors inititial parameter for model prediction, not affecting model architecture.
      Default "auto" means: 1 if (anchors_mode=="anchor_free" or anchors_mode=="yolor"), else 4.
  rescale_mode: model precessing input, not for model structure. Defulat "torch".

Returns:
    A `keras.Model` instance.
"""

EfficientDet.__doc__ = __head_doc__.format("") + """  features_pick: specific `layer names` or `pyramid feature indexes` from backbone model.
        Default `[-3, -2, -1]` means using the last 3 pyramid feature output from backbone.
  additional_features: number of generated features from last backbone feature. `3` for EfficientDetD7X, `2` for others.
  fpn_depth: number of `bi_fpn` repeats.
  head_depth: depth for `bbox_regressor` and `classifier`.
  num_channels: channel filters for `bi_fpn`, `bbox_regressor` and `classifier`.
  use_weighted_sum: if use weighted sum in `bi_fpn`. `False` for `EfficientDetD6, D7, D7X`, `True` for others.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model              | Params | FLOPs   | Input | COCO val AP | test AP |
  | ------------------ | ------ | ------- | ----- | ----------- | ------- |
  | EfficientDetD0     | 3.9M   | 2.55G   | 512   | 34.3        | 34.6    |
  | EfficientDetD1     | 6.6M   | 6.13G   | 640   | 40.2        | 40.5    |
  | EfficientDetD2     | 8.1M   | 11.03G  | 768   | 43.5        | 43.9    |
  | EfficientDetD3     | 12.0M  | 24.95G  | 896   | 46.8        | 47.2    |
  | EfficientDetD4     | 20.7M  | 55.29G  | 1024  | 49.3        | 49.7    |
  | EfficientDetD5     | 33.7M  | 135.62G | 1280  | 51.2        | 51.5    |
  | EfficientDetD6     | 51.9M  | 225.93G | 1280  | 52.1        | 52.6    |
  | EfficientDetD7     | 51.9M  | 325.34G | 1536  | 53.4        | 53.7    |
  | EfficientDetD7X    | 77.0M  | 410.87G | 1536  | 54.4        | 55.1    |
  | EfficientDetLite0  | 3.2M   | 0.98G   | 320   | 27.5        | 26.41   |
  | EfficientDetLite1  | 4.2M   | 1.97G   | 384   | 32.6        | 31.50   |
  | EfficientDetLite2  | 5.3M   | 3.38G   | 448   | 36.2        | 35.06   |
  | EfficientDetLite3  | 8.4M   | 7.50G   | 512   | 39.9        | 38.77   |
  | EfficientDetLite3X | 9.3M   | 14.01G  | 640   | 44.0        | 42.64   |
  | EfficientDetLite4  | 15.1M  | 20.20G  | 640   | 44.4        | 43.18   |
"""

__model_doc__ = __head_doc__ + __tail_doc__
EfficientDetD0.__doc__ = __model_doc__.format("Default None for EfficientNetV1B0.")
EfficientDetD1.__doc__ = __model_doc__.format("Default None for EfficientNetV1B1.")
EfficientDetD2.__doc__ = __model_doc__.format("Default None for EfficientNetV1B2.")
EfficientDetD3.__doc__ = __model_doc__.format("Default None for EfficientNetV1B3.")
EfficientDetD4.__doc__ = __model_doc__.format("Default None for EfficientNetV1B4.")
EfficientDetD5.__doc__ = __model_doc__.format("Default None for EfficientNetV1B5.")
EfficientDetD6.__doc__ = __model_doc__.format("Default None for EfficientNetV1B6.")
EfficientDetD7.__doc__ = __model_doc__.format("Default None for EfficientNetV1B6.")
EfficientDetD7X.__doc__ = __model_doc__.format("Default None for EfficientNetV1B7.")
EfficientDetLite0.__doc__ = __model_doc__.format("Default None for EfficientNetV1Lite0.")
EfficientDetLite1.__doc__ = __model_doc__.format("Default None for EfficientNetV1Lite1.")
EfficientDetLite2.__doc__ = __model_doc__.format("Default None for EfficientNetV1Lite2.")
EfficientDetLite3.__doc__ = __model_doc__.format("Default None for EfficientNetV1Lite3.")
EfficientDetLite3X.__doc__ = __model_doc__.format("Default None for EfficientNetV1Lite3X.")
EfficientDetLite4.__doc__ = __model_doc__.format("Default None for EfficientNetV1Lite4.")
