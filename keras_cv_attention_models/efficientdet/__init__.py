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
    DecodePredictions,
)

__head_doc__ = """
Keras implementation of [google/automl/efficientdet](https://github.com/google/automl/tree/master/efficientdet).
Paper [Paper 1911.09070 EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/pdf/1911.09070.pdf).
"""

__tail_doc__ = """  num_anchors: number of anchors for a single grid point, should be same with dataset used value.
  num_classes: total output classes. `90` for EfficientDet pretrained, `80` for `tfds.coco`. Set `0` to disable `classifier` output.
  num_heads: heads number for transformer block. `64` for `CoAtNet5`, `32` for others.
  activation: activation used in whole model, default `swish`. Default
  freeze_backbone: set `True` for `backbone.trainable = False`. Default `False`.
  anchor_scale: anchors inititial parameter for model prediction, not affecting model architecture. Default 4.
  pyramid_levels: anchors inititial parameter for model prediction, not affecting model architecture. Default `[3, 4, 5, 6, 7]`.
  pretrained: one of `None` (random initialization) or 'coco' (pre-training on COCO).
      Will try to download and load pre-trained model weights if not None. Defailt `coco`.
  **kwargs: other parameters if available.

Returns:
    A `keras.Model` instance.
"""

EfficientDet.__doc__ = __head_doc__ + """
Args:
  backbone: backbone model, could be any model with pyramid stage structure.
  features_pick: specific `layer names` or `pyramid feature indexes` from backbone model.
        Default `[-3, -2, -1]` means using the last 3 pyramid feature output from backbone.
  additional_features: number of generated features from last backbone feature. `3` for EfficientDetD7X, `2` for others.
  fpn_depth: number of `bi_fpn` repeats.
  head_depth: depth for `bbox_regressor` and `classifier`.
  num_channels: channel filters for `bi_fpn`, `bbox_regressor` and `classifier`.
  use_weighted_sum: if use weighted sum in `bi_fpn`. `False` for `EfficientDetD6, D7, D7X`, `True` for others.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model            | Params | Image resolution | COCO test AP |
  | ---------------- | ------ | ---------------- | ------------ |
  | EfficientDet-D0  | 3.9M   | 512              | 34.6         |
  | EfficientDet-D1  | 6.6M   | 640              | 40.5         |
  | EfficientDet-D2  | 8.1M   | 768              | 43.9         |
  | EfficientDet-D3  | 12.0M  | 896              | 47.2         |
  | EfficientDet-D4  | 20.7M  | 1024             | 49.7         |
  | EfficientDet-D5  | 33.7M  | 1280             | 51.5         |
  | EfficientDet-D6  | 51.9M  | 1280             | 52.6         |
  | EfficientDet-D7  | 51.9M  | 1536             | 53.7         |
  | EfficientDet-D7x | 77.0M  | 1536             | 55.1         |
"""

EfficientDetD0.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

EfficientDetD1.__doc__ = EfficientDetD0.__doc__
EfficientDetD2.__doc__ = EfficientDetD0.__doc__
EfficientDetD3.__doc__ = EfficientDetD0.__doc__
EfficientDetD4.__doc__ = EfficientDetD0.__doc__
EfficientDetD5.__doc__ = EfficientDetD0.__doc__
EfficientDetD6.__doc__ = EfficientDetD0.__doc__
EfficientDetD7.__doc__ = EfficientDetD0.__doc__
EfficientDetD7X.__doc__ = EfficientDetD0.__doc__
