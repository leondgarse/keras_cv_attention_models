# ___Keras NFNets___
***

- **In progress**

## Summary
  - Keras implementation of [Github deepmind/nfnets](https://github.com/deepmind/deepmind-research/tree/master/nfnets). Paper [PDF 2102.06171 High-Performance Large-Scale Image Recognition Without Normalization](https://arxiv.org/pdf/2102.06171.pdf).
  - Model weights reloaded from official publication.
***

## Models
  | Model       | Params | Image  resolution | Top1 Acc | Download |
  | ----------- | ------ | ----------------- | -------- | -------- |
  | NFNetF0     | 71.5M  | 256               | 83.6     |          |
  | NFNetF1     | 132.6M | 320               | 84.7     |          |
  | NFNetF2     | 193.8M | 352               | 85.1     |          |
  | NFNetF3     | 254.9M | 416               | 85.7     |          |
  | NFNetF4     | 316.1M | 512               | 85.9     |          |
  | NFNetF5     | 377.2M | 544               | 86.0     |          |
  | NFNetF6 SAM | 438.4M | 576               | 86.5     |          |
  | NFNetF7     | 499.5M |                   |          |          |
## Usage
  ```py
  from keras_cv_attention_models import nfnets

  mm = nfnets.NFNetF0()
  mm.summary()
  ```
