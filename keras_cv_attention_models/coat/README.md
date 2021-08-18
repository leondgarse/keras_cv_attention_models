# ___Keras CoaT___
***

## Summary
  - Coat article: [PDF 2104.06399 CoaT: Co-Scale Conv-Attentional Image Transformers](http://arxiv.org/abs/2104.06399).
  - [Github mlpc-ucsd/CoaT](https://github.com/mlpc-ucsd/CoaT).
***

## Models
| Model         | Params | Image resolution | Top1 Acc |
| ------------- | ------ | ---------------- | -------- |
| CoaTLiteTiny  | 5.7M   | 224              | 77.5     |
| CoaTLiteMini  | 11M    | 224              | 79.1     |
| CoaTLiteSmall | 20M    | 224              | 81.9     |
| CoaTTiny      | 5.5M   | 224              | 78.3     |
| CoaTMini      | 10M    | 224              | 81.0     |
## Usage
  ```py
  from keras_cv_attention_models import coat

  mm = coat.CoaTLiteTiny()
  mm.summary()
  ```
***
