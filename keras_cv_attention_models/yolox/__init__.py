from keras_cv_attention_models.yolox.yolox import (
    CSPDarknet,
    YOLOXHead,
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
"""

__tail_doc__ = """

Returns:
    A `keras.Model` instance.
"""

YOLOX.__doc__ = __head_doc__.format("") + """
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model     | Params | Image resolution | COCO test AP |
  | --------- | ------ | ---------------- | ------------ |
  | YOLOXNano | 0.91M  | 416              | 25.8         |
  | YOLOXTiny | 5.06M  | 416              | 32.8         |
  | YOLOXS    | 9.0M   | 640              | 40.5         |
  | YOLOXM    | 25.3M  | 640              | 47.2         |
  | YOLOXL    | 54.2M  | 640              | 50.1         |
  | YOLOXX    | 99.1M  | 640              | 51.5         |
"""
