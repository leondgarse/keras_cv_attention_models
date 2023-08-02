from keras_cv_attention_models.yolov8.yolov8 import (
    YOLOV8Backbone,
    YOLOV8,
    YOLOV8_N,
    YOLOV8_S,
    YOLOV8_M,
    YOLOV8_L,
    YOLOV8_X,
    YOLOV8_X6,
    YOLOV8_N_CLS,
    YOLOV8_S_CLS,
    YOLOV8_M_CLS,
    YOLOV8_L_CLS,
    YOLOV8_X_CLS,
    switch_to_deploy,
)
from keras_cv_attention_models.yolov8.yolo_nas import YOLO_NAS, YOLO_NAS_S, YOLO_NAS_M, YOLO_NAS_L

__head_doc__ = """
Keras implementation of [Github ultralytics/ultralytics](https://github.com/ultralytics/ultralytics).
"""

__detector_head_doc__ = """
Args:
  backbone: backbone model, could be any model with pyramid stage structure.
      Default None for YOLOV8Backbone.
"""

__detector_tail_doc__ = """  features_pick: specific `layer names` or `pyramid feature indexes` from backbone model.
        Default `[-3, -2, -1]` means using the last 3 pyramid feature output from backbone.
  regression_len: bbox output len, typical value is 4, for yolov8 reg_max=16 -> regression_len = 16 * 4 == 64.
      For YOLO_NAS, it's 17 * 4 == 68.
  use_reparam_conv: boolean value if using `reparam_conv_bn` instead of `conv_bn` block in all `csp_block`s.
      Roughly, `reparam_conv_bn` is a block with `Conv_3x3+BN(inputs) + Conv_1x1(inputs) + inputs`,
      and can be reparametered to a single `Conv_3x3` layer after training.
  paf_parallel_mode: False for YOLOV8_X6 and True for others.
      If False, only concat `short` and the last `deep` one in `path_aggregation_fpn` module.
  anchors_mode: one of ["efficientdet", "anchor_free", "yolor", "yolov8"], controls which anchor to use.
      - efficientdet anchors default settings: use_object_scores=False, num_anchors=9, anchor_scale=4,
          aspect_ratios=[1, 2, 0.5], num_scales=3, grid_zero_start=False.
      - anchor_free default settings: use_object_scores=True, num_anchors=1, anchor_scale=1,
          aspect_ratios=[1], num_scales=1, grid_zero_start=True.
      - yolor default settings: use_object_scores=True, num_anchors=3.
      - yolov8 default settings: use_object_scores=False, num_anchors=1, anchor_scale=1,
          aspect_ratios=[1], num_scales=1, grid_zero_start=False.
      Default "yolov8".
  num_anchors: number of anchors for a single grid point, should be same with dataset used value.
      Default "auto" means: anchors_mode=="anchor_free" / "yolov8" -> 1, anchors_mode=="yolor" -> 3, else 9.
  use_object_scores: bollean value if model header output includes `object_scores`.
      Default "auto" means: True if anchors_mode=="anchor_free" or anchors_mode=="yolor", else False.
  input_shape: input shape if backbone is None, else will use input_shape from backbone.
  num_classes: total output classes. Set `0` to disable `classifier` output. Default 80 for COCO.
  activation: activation used in whole model, default `swish`. Default "swish".
  classifier_activation: The activation function to use for classifier output if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer. Default `sigmoid`.
  freeze_backbone: set `True` for `backbone.trainable = False`. Default `False`.
  pretrained: one of `None` (random initialization) or 'coco' (pre-training on COCO).
      Will try to download and load pre-trained model weights if not None. Default `coco`.
  pyramid_levels_min: anchors inititial parameter for model prediction, not affecting model architecture. Default `3`.
      pyramid_levels_max is calculated as `pyramid_levels_min + len(features_pick) - 1`.
  anchor_scale: anchors inititial parameter for model prediction, not affecting model architecture.
      Default "auto" means: 1 if (anchors_mode=="anchor_free" or anchors_mode=="yolor"), else 4.
  rescale_mode: model precessing input, not for model structure. Defulat "raw01" means input value in range `[0, 1]`.

Returns:
    A `keras.Model` instance.
"""

YOLOV8.__doc__ = __head_doc__ + __detector_head_doc__ + """
[YOLOV8Backbone parameters]
  csp_channels: YOLOV8Backbone backbone channel for each block, default: `[32, 64, 128, 256]`.
  csp_depthes: YOLOV8Backbone backbone depth for each block, default: `[1, 2, 2, 1]`.

[YOLOV8 parameters]
  model_name: string, model name.

[Detector parameters]
""" + __detector_tail_doc__ + """
Model architectures:
  | Model     | Params | FLOPs  | Input | COCO val AP | test AP |
  | --------- | ------ | ------ | ----- | ----------- | ------- |
  | YOLOV8_N  | 3.16M  | 4.39G  | 640   | 37.3        |         |
  | YOLOV8_S  | 11.17M | 14.33G | 640   | 44.9        |         |
  | YOLOV8_M  | 25.90M | 39.52G | 640   | 50.2        |         |
  | YOLOV8_L  | 43.69M | 82.65G | 640   | 52.9        |         |
  | YOLOV8_X  | 68.23M | 129.0G | 640   | 53.9        |         |
  | YOLOV8_X6 | 97.42M | 522.6G | 1280  | 56.7 ?      |         |

  | Model                    | Params | FLOPs  | Input | COCO val AP | test AP |
  | ------------------------ | ------ | ------ | ----- | ----------- | ------- |
  | YOLO_NAS_S               | 12.88M | 16.96G | 640   | 47.5        |         |
  | - use_reparam_conv=False | 12.18M | 15.92G | 640   | 47.5        |         |
  | YOLO_NAS_M               | 33.86M | 47.12G | 640   | 51.55       |         |
  | - use_reparam_conv=False | 31.92M | 43.91G | 640   | 51.55       |         |
  | YOLO_NAS_L               | 44.53M | 64.53G | 640   | 52.22       |         |
  | - use_reparam_conv=False | 42.02M | 59.95G | 640   | 52.22       |         |
"""

YOLOV8_N.__doc__ = __head_doc__ + __detector_head_doc__ + __detector_tail_doc__
YOLOV8_S.__doc__ = YOLOV8_N.__doc__
YOLOV8_M.__doc__ = YOLOV8_N.__doc__
YOLOV8_L.__doc__ = YOLOV8_N.__doc__
YOLOV8_X.__doc__ = YOLOV8_N.__doc__
YOLOV8_X6.__doc__ = YOLOV8_N.__doc__

YOLO_NAS.__doc__ = YOLOV8.__doc__
YOLO_NAS_S.__doc__ = YOLOV8_N.__doc__
YOLO_NAS_M.__doc__ = YOLOV8_N.__doc__
YOLO_NAS_L.__doc__ = YOLOV8_N.__doc__

__classifier_tail_doc__ = """  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation used in whole stacks, default `gelu`.
  dropout: top dropout rate if top layers is included. Default 0.
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
      Default is `None`.
  pretrained: one of `None` (random initialization) or 'imagenet' (pre-training on ImageNet).
      Will try to download and load pre-trained model weights if not None.
  **kwargs: other parameters if available.
Returns:
    A `keras.Model` instance.
"""

YOLOV8Backbone.__doc__ = __head_doc__ + """
Args:
  channels: channel for each block, default: `[32, 64, 128, 256]`.
  depthes: depth for each block, default: `[1, 2, 2, 1]`.
""" + __classifier_tail_doc__ + """
Model architectures:
  | Model        | Params | FLOPs@640 | FLOPs@224 | Input | Top1 Acc |
  | ------------ | ------ | --------- | --------- | ----- | -------- |
  | YOLOV8_N_CLS | 2.72M  | 1.65G     | 203.7M    | 224   | 66.6     |
  | YOLOV8_S_CLS | 6.36M  | 6.24G     | 765.7M    | 224   | 72.3     |
  | YOLOV8_M_CLS | 17.05M | 20.85G    | 2.56G     | 224   | 76.4     |
  | YOLOV8_L_CLS | 37.48M | 49.41G    | 6.05G     | 224   | 78.0     |
  | YOLOV8_X_CLS | 57.42M | 76.96G    | 9.43G     | 224   | 78.4     |
"""

YOLOV8_N_CLS.__doc__ = __head_doc__ + """
Args:
""" + __classifier_tail_doc__

YOLOV8_S_CLS.__doc__ = YOLOV8_N_CLS.__doc__
YOLOV8_M_CLS.__doc__ = YOLOV8_N_CLS.__doc__
YOLOV8_L_CLS.__doc__ = YOLOV8_N_CLS.__doc__
YOLOV8_X_CLS.__doc__ = YOLOV8_N_CLS.__doc__
