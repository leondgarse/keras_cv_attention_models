# Keras_CMT
***

## Summary
- CMT article: [PDF 2107.06263 CoAtNet: CMT: Convolutional Neural Networks Meet Vision Transformers](https://arxiv.org/pdf/2107.06263.pdf)
- [Github FlyEgle/CMT-pytorch](https://github.com/FlyEgle/CMT-pytorch)
- No pretraind available.
***

## Models
  | Model    | Params | Image resolution | Top1 Acc |
  | -------- | ------ | ---------------- | -------- |
  | CMTTiny  | 9.5M   | 160              | 79.2     |
  | CMTXS    | 15.2M  | 192              | 81.8     |
  | CMTSmall | 25.1M  | 224              | 83.5     |
  | CMTBig   | 45.7M  | 256              | 84.5     |
## Usage
  ```py
  from keras_cv_attention_models import cmt

  # No pretraind available.
  mm = cmt.CMTTiny()
  mm.summary()
  ```
***
