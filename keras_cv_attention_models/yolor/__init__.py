from keras_cv_attention_models.yolor.yolor import (
    CSPDarknet,
    YOLOR,
    YOLOR_CSP,
    YOLOR_CSPX,
    YOLOR_P6,
    YOLOR_W6,
    YOLOR_E6,
    YOLOR_D6,
)

__head_doc__ = """
Keras implementation of [Github WongKinYiu/yolor](https://github.com/WongKinYiu/yolor).
Paper [Paper 2105.04206 You Only Learn One Representation: Unified Network for Multiple Tasks](https://arxiv.org/pdf/2105.04206.pdf).

Args:
  backbone: backbone model, could be any model with pyramid stage structure.
      Default None for CSPDarknet with csp_depthes={csp_depthes}, csp_channels={csp_channels}.
"""

__tail_doc__ = """  features_pick: specific `layer names` or `pyramid feature indexes` from backbone model.
        Default `[-3, -2, -1]` means using the last 3 pyramid feature output from backbone.
  use_anchor_free_mode: boolean value if use anchor free mode. Default False.
  use_yolor_anchors_mode: boolean value if use yolor anchors mode. Default True.
      Currently 3 types anchors supported:
      - efficientdet anchors default settings: use_object_scores=False, num_anchors=9, anchor_scale=4,
          aspect_ratios=[1, 2, 0.5], num_scales=3, grid_zero_start=False.
      - anchor_free_mode default settings: use_object_scores=True, num_anchors=1, anchor_scale=1,
          aspect_ratios=[1], num_scales=1, grid_zero_start=True.
      - yolor_anchors_mode default settings: use_object_scores=True, num_anchors=3.
  num_anchors: number of anchors for a single grid point, should be same with dataset used value.
      Default "auto" means 1 if use_anchor_free_mode else 3
  use_object_scores: bollean value if model header output includes `object_scores`. Default True.
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
      Default "auto" means 1 if use_anchor_free_mode else 4.
  rescale_mode: model precessing input, not for model structure. Defulat "raw01" means input value in range `[0, 1]`.

Returns:
    A `keras.Model` instance.
"""

YOLOR.__doc__ = __head_doc__.format(csp_depthes=[2, 8, 8, 4], csp_channels=[128, 256, 512, 1024]) + """  csp_depthes: CSPDarknet backbone depth for each block, default `[2, 8, 8, 4]`.
  csp_channels: CSPDarknet backbone channel for each block, default: `[128, 256, 512, 1024]`.
  stem_width: CSPDarknet backbone stem width, default -1 means using csp_channels[0] // 2.
  use_focus_stem: boolean value if CSPDarknet backbone using focus_stem or conv one, default False.
  ssp_depth: CSPDarknet backbone spatial_pyramid_pooling depth, default 2.
  csp_use_pre: boolean value if CSPDarknet backbone blocks using pre-conv for deep branch, default False.
  csp_use_post: boolean value if CSPDarknet backbone blocks using post-conv for deep branch, default True.
  use_csp_downsample: boolean value if CSPDarknet backbone and FPN downsample using `csp_conv_downsample`
      or a conv layer with `kernel_size=3, strides=2`. Default False.
  fpn_depth: depth for FPN headers, default 2.
  use_depthwise_conv: boolean value if using additional depthwise conv.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model      | Params | Image resolution | COCO test AP |
  | ---------- | ------ | ---------------- | ------------ |
  | YOLOR_CSP  | 52.9M  | 640              | 52.8         |
  | YOLOR_CSPX | 99.8M  | 640              | 54.8         |
  | YOLOR_P6   | 37.3M  | 1280             | 55.7         |
  | YOLOR_W6   | 79.9M  | 1280             | 56.9         |
  | YOLOR_E6   | 115.9M | 1280             | 57.6         |
  | YOLOR_D6   | 151.8M | 1280             | 58.2         |
"""

YOLOR_CSP.__doc__ = __head_doc__.format(csp_depthes=[2, 8, 8, 4], csp_channels=[128, 256, 512, 1024]) + __tail_doc__
YOLOR_CSP.__doc__ = __head_doc__.format(csp_depthes=[3, 10, 10, 5], csp_channels=[160, 320, 640, 1280]) + __tail_doc__
YOLOR_P6.__doc__ = __head_doc__.format(csp_depthes=[3, 7, 7, 3, 3], csp_channels=[128, 256, 384, 512, 640]) + __tail_doc__
YOLOR_W6.__doc__ = __head_doc__.format(csp_depthes=[3, 7, 7, 3, 3], csp_channels=[128, 256, 512, 768, 1024]) + __tail_doc__
YOLOR_E6.__doc__ = __head_doc__.format(csp_depthes=[3, 7, 7, 3, 3], csp_channels=[160, 320, 640, 960, 1280]) + __tail_doc__
YOLOR_D6.__doc__ = __head_doc__.format(csp_depthes=[3, 15, 15, 7, 7], csp_channels=[160, 320, 640, 960, 1280]) + __tail_doc__
