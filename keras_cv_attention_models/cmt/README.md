# Keras_CMT
***

## Summary
- CMT article: [PDF 2107.06263 CoAtNet: CMT: Convolutional Neural Networks Meet Vision Transformers](https://arxiv.org/pdf/2107.06263.pdf)
- [Github FlyEgle/CMT-pytorch](https://github.com/FlyEgle/CMT-pytorch)
- No pretraind available.
***

## Models
  | Model  | Params | Image resolution | Top1 Acc |
  | ------ | ------ | ---------------- | -------- |
  | CMT-T  | 11.3M  | 160x160          | 79.2     |
  | CMT-XS |        | 192x192          | 81.8     |
  | CMT-S  |        | 224x224          | 83.5     |
  | CMT-L  |        | 256x256          | 84.5     |
## Usage
  ```py
  from keras_cv_attention_models import cmt

  # No pretraind available.
  mm = cmt.CMTTiny()
  mm.summary()
  ```
***
