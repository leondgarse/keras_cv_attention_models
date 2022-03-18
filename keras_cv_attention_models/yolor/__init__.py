from keras_cv_attention_models.yolor.yolor import (
    CSPDarknet,
    YOLOR,
    YOLOR_CSP,
    YOLOR_CSPX,
)

__head_doc__ = """
Keras implementation of [Github WongKinYiu/yolor](https://github.com/WongKinYiu/yolor).
Paper [Paper 2105.04206 You Only Learn One Representation: Unified Network for Multiple Tasks](https://arxiv.org/pdf/2105.04206.pdf).

Args:
  backbone: backbone model, could be any model with pyramid stage structure.
      Default None for CSPDarknet with depth_mul={depth_mul}, width_mul={width_mul}.
"""

__tail_doc__ = """  features_pick: specific `layer names` or `pyramid feature indexes` from backbone model.
        Default `[-3, -2, -1]` means using the last 3 pyramid feature output from backbone.
  use_anchor_free_mode: boolean value if use anchor free mode. num_anchors = 1 if use_anchor_free_mode else 3. Default False.
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

YOLOR.__doc__ = __head_doc__.format(depth_mul=1, width_mul=1) + """  depth_mul: CSPDarknet backbone and FPN blocks depth expansion ratio.
      - For CSPDarknet, base_depth = max(round(depth_mul * 2), 1)
      - For FPN blocks, csp_depth = max(round(depth_mul * 2), 1)
  width_mul: CSPDarknet backbone width expansion ratio, base_channels = int(width_mul * 64).
  use_depthwise_conv: boolean value if using additional depthwise conv.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model      | Params | Image resolution | COCO val AP |
  | ---------- | ------ | ---------------- | ----------- |
  | YOLOR_CSP  | 52.9M  | 640              | 50.0        |
  | YOLOR_CSPX | 99.8M  | 640              | 51.5        |
"""

YOLOR_CSP.__doc__ = __head_doc__.format(depth_mul=1, width_mul=1) + __tail_doc__
YOLOR_CSP.__doc__ = __head_doc__.format(depth_mul=1.3, width_mul=1.25) + __tail_doc__
