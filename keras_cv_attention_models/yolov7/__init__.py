from keras_cv_attention_models.yolov7.yolov7 import (
    YOLOV7Backbone,
    YOLOV7,
    YOLOV7_Tiny,
    YOLOV7_CSP,
    YOLOV7_X,
    YOLOV7_W6,
    YOLOV7_E6,
    YOLOV7_D6,
    YOLOV7_E6E,
)

__head_doc__ = """
Keras implementation of [Github WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7).
Paper [Paper 2207.02696 YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/pdf/2207.02696.pdf).

Args:
  backbone: backbone model, could be any model with pyramid stage structure.
      Default None for YOLOV7Backbone.
"""

__tail_doc__ = """  features_pick: specific `layer names` or `pyramid feature indexes` from backbone model.
        Default `[-3, -2, -1]` means using the last 3 pyramid feature output from backbone.
  anchors_mode: one of ["efficientdet", "anchor_free", "yolor"], controls which anchor to use.
      - efficientdet anchors default settings: use_object_scores=False, num_anchors=9, anchor_scale=4,
          aspect_ratios=[1, 2, 0.5], num_scales=3, grid_zero_start=False.
      - anchor_free default settings: use_object_scores=True, num_anchors=1, anchor_scale=1,
          aspect_ratios=[1], num_scales=1, grid_zero_start=True.
      - yolor default settings: use_object_scores=True, num_anchors=3.
      Default "yolor".
  num_anchors: number of anchors for a single grid point, should be same with dataset used value.
      Default "auto" means: anchors_mode=="anchor_free" -> 1, anchors_mode=="yolor" -> 3, else 9.
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

YOLOV7.__doc__ = __head_doc__ + """
[YOLOV7Backbone parameters]
  csp_channels: YOLOV7Backbone backbone channel for each block, default: `[64, 128, 256, 256]`.
  stack_concats: .
  stack_depth: .
  stack_out_ratio: .
  use_additional_stack: .
  stem_width: CSPDarknet backbone stem width, default -1 means using csp_channels[0] // 2.
  stem_type: One of ["conv3", "focus", "conv1"], "focus" for YOLOV7_*6 models, "conv1" for YOLOV7_Tiny. Default "conv3".
  csp_downsample_ratios: list value controls down sample block channel ratios for each stack.
  spp_depth: .

[FPN parameters]
  fpn_hidden_channels: .
  fpn_channel_ratio: .
  fpn_stack_concats: .
  fpn_stack_depth: .
  fpn_mid_ratio: .
  fpn_csp_downsample_ratio: .
  use_reparam_conv_head: .
  use_csp_downsample: boolean value if CSPDarknet backbone and FPN downsample using `csp_conv_downsample`
      or a conv layer with `kernel_size=3, strides=2`. Default False.
  model_name: string, model name.

[Detector parameters]
""" + __tail_doc__ + """
Model architectures:
  | Model       | Params | FLOPs  | Input | COCO val AP | test AP |
  | ----------- | ------ | ------ | ----- | ----------- | ------- |
  | YOLOV7_Tiny | 6.23M  | 2.90G  | 416   | 33.3        |         |
  | YOLOV7_CSP  | 37.67M | 53.0G  | 640   | 51.4        |         |
  | YOLOV7_X    | 71.41M | 95.0G  | 640   | 53.1        |         |
  | YOLOV7_W6   | 70.49M | 180.1G | 1280  | 54.9        |         |
  | YOLOV7_E6   | 97.33M | 257.6G | 1280  | 56.0        |         |
  | YOLOV7_D6   | 133.9M | 351.4G | 1280  | 56.6        |         |
  | YOLOV7_E6E  | 151.9M | 421.7G | 1280  | 56.8        |         |
"""

YOLOV7_Tiny.__doc__ = __head_doc__ + __tail_doc__
YOLOV7_CSP.__doc__ = YOLOV7_Tiny.__doc__
YOLOV7_X.__doc__ = YOLOV7_Tiny.__doc__
YOLOV7_W6.__doc__ = YOLOV7_Tiny.__doc__
YOLOV7_E6.__doc__ = YOLOV7_Tiny.__doc__
YOLOV7_D6.__doc__ = YOLOV7_Tiny.__doc__
YOLOV7_E6E.__doc__ = YOLOV7_Tiny.__doc__
