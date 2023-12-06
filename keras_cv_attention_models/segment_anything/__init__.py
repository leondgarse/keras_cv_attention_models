from keras_cv_attention_models.segment_anything.sam import SAM, MobileSAM, EfficientViT_SAM_L0

__head_doc__ = """
Keras implementation of [Github facebookresearch/segment-anything](https://github.com/facebookresearch/segment-anything).
Paper [PDF 2304.02643 Segment Anything](https://arxiv.org/abs/2304.02643).
"""

__call_doc__ = """
Call args:
  image: raw input image. np.array value in shape `[height, width, 3]`, value range in `[0, 255]`.
  points: combinging with `labels`, specific points coordinates as background or foreground.
      np.array value in shape `[None, 2]`, `2` means `[left, top]`.
      left / top value range in `[0, 1]` or `[0, width]` / `[0, height]`.
  labels: combinging with `points`, specific points coordinates as background or foreground.
      np.array value in shape `[None]`, value in `[0, 1]`, where 0 means relative point being background, and 1 foreground.
  boxes: specific box area performing segmentation.
      np.array value in shape `[None, 4]`, `4` means `[left, top, right, bottom]`.
      left and right / top and bottom value range in `[0, 1]` or `[0, width]` / `[0, height]`.
  masks: NOT tested.
  mask_threshold: float value for regading model output where `masks > mask_threshold` as True.
  return_logits: boolean value if returning boolean mask or logits mask. Default False for boolean mask.
"""

__tail_doc__ = """  image_shape: int or list of 2 int like [1024, 1024].
  embed_dims: inner channels for prompt encoder.
  mask_hidden_dims: `MaskEncoder` hidden channels.
  pretrained: one of `None` (random initialization) or 'sam' (pre-training on SA-1B from Segment Anything paper).
      Will try to download and load pre-trained model weights if not None.
""" + __call_doc__ + """
Returns:
    A `keras.Model` instance.
"""

SAM.__doc__ = __head_doc__ + """
Init args:
  image_encoder: string or built image encoder model. Currently string can be one of ["TinyViT_5M", "EfficientViT_L0"].
  name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model               | Params | FLOPs | Input | COCO val mIoU |
  | ------------------- | ------ | ----- | ----- | ------------- |
  | MobileSAM           | 5.75M  | 39.4G | 1024  | 72.8          |
  | EfficientViT_SAM_L0 | 30.73M | 35.4G | 512   | 74.45         |
"""

MobileSAM.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

EfficientViT_SAM_L0.__doc__ = MobileSAM.__doc__
