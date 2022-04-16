from keras_cv_attention_models.yolox.yolox import (
    CSPDarknet,
    YOLOX,
    YOLOXNano,
    YOLOXTiny,
    YOLOXS,
    YOLOXM,
    YOLOXL,
    YOLOXX,
)

__head_doc__ = """
Keras implementation of [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX).
Paper [Paper 2107.08430 YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/pdf/2107.08430.pdf).

Args:
  backbone: backbone model, could be any model with pyramid stage structure.
      Default None for CSPDarknet with depth_mul={depth_mul}, width_mul={width_mul}.
"""

__tail_doc__ = """  features_pick: specific `layer names` or `pyramid feature indexes` from backbone model.
        Default `[-3, -2, -1]` means using the last 3 pyramid feature output from backbone.
  anchors_mode: one of ["efficientdet", "anchor_free", "yolor"], controls which anchor to use.
      - efficientdet anchors default settings: use_object_scores=False, num_anchors=9, anchor_scale=4,
          aspect_ratios=[1, 2, 0.5], num_scales=3, grid_zero_start=False.
      - anchor_free default settings: use_object_scores=True, num_anchors=1, anchor_scale=1,
          aspect_ratios=[1], num_scales=1, grid_zero_start=True.
      - yolor default settings: use_object_scores=True, num_anchors=3.
      Default "anchor_free".
  num_anchors: number of anchors for a single grid point, should be same with dataset used value.
      Default "auto" means: anchors_mode=="anchor_free" -> 1, anchors_mode=="yolor" -> 3, else 9.
  use_object_scores: bollean value if model header output includes `object_scores`.
      Default "auto" means: True if anchors_mode=="anchor_free" or anchors_mode=="yolor", else False.
  num_classes: total output classes. Set `0` to disable `classifier` output. Default 80 for COCO.
  input_shape: input shape if backbone is None, else will use input_shape from backbone.
  activation: activation used in whole model, default `swish`. Default "swish".
  freeze_backbone: set `True` for `backbone.trainable = False`. Default `False`.
  pretrained: one of `None` (random initialization) or 'coco' (pre-training on COCO).
      Will try to download and load pre-trained model weights if not None. Default `coco`.
  pyramid_levels_min: anchors inititial parameter for model prediction, not affecting model architecture. Default `3`.
      pyramid_levels_max is calculated as `pyramid_levels_min + len(features_pick) - 1`.
  anchor_scale: anchors inititial parameter for model prediction, not affecting model architecture.
      Default "auto" means: 1 if (anchors_mode=="anchor_free" or anchors_mode=="yolor"), else 4.
  rescale_mode: model precessing input, not for model structure. Defulat "raw" means input value in range `[0, 255]`.

Returns:
    A `keras.Model` instance.
"""

YOLOX.__doc__ = __head_doc__.format(depth_mul=1, width_mul=1) + """  depth_mul: CSPDarknet backbone and FPN blocks depth expansion ratio.
      - For CSPDarknet, base_depth = max(round(depth_mul * 3), 1)
      - For FPN blocks, csp_depth = max(round(depth_mul * 3), 1)
  width_mul: CSPDarknet backbone and output header width expansion ratio.
      - For CSPDarknet, base_channels = int(width_mul * 64)
      - For output header, out_channel = int(256 * width_mul)
      Default -1 means: `min([ii.shape[-1] for ii in features]) / 256` for custom backbones.
  use_depthwise_conv: boolean value if using additional depthwise conv. True for YOLOXNano, False for others.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model     | Params | FLOPs   | Input | COCO val AP | test AP |
  | --------- | ------ | ------- | ----- | ----------- | ------- |
  | YOLOXNano | 0.91M  | 0.53G   | 416   | 25.8        |         |
  | YOLOXTiny | 5.06M  | 3.22G   | 416   | 32.8        |         |
  | YOLOXS    | 9.0M   | 13.39G  | 640   | 40.5        | 40.5    |
  | YOLOXM    | 25.3M  | 36.84G  | 640   | 46.9        | 47.2    |
  | YOLOXL    | 54.2M  | 77.76G  | 640   | 49.7        | 50.1    |
  | YOLOXX    | 99.1M  | 140.87G | 640   | 51.5        | 51.5    |
"""

YOLOXNano.__doc__ = __head_doc__.format(depth_mul=0.33, width_mul=0.25) + __tail_doc__
YOLOXTiny.__doc__ = __head_doc__.format(depth_mul=0.33, width_mul=0.375) + __tail_doc__
YOLOXS.__doc__ = __head_doc__.format(depth_mul=0.33, width_mul=0.5) + __tail_doc__
YOLOXM.__doc__ = __head_doc__.format(depth_mul=0.67, width_mul=0.75) + __tail_doc__
YOLOXL.__doc__ = __head_doc__.format(depth_mul=1.0, width_mul=1.0) + __tail_doc__
YOLOXX.__doc__ = __head_doc__.format(depth_mul=1.33, width_mul=1.25) + __tail_doc__
