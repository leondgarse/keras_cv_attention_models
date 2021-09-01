# ___Keras HaloNet___
***

## Summary
  - [Github lucidrains/halonet-pytorch](https://github.com/lucidrains/halonet-pytorch).
  - HaloAttention article: [PDF 2103.12731 Scaling Local Self-Attention for Parameter Efficient Visual Backbones](https://arxiv.org/pdf/2103.12731.pdf).
  - No pretraind available. Architecture is guessed from article, so it's NOT certain.
## Models
  ![](halo_attention.png)

  | Model     | Params | Image resolution | Top1 Acc |
  | --------- | ------ | ---------------- | -------- |
  | HaloNetH0 | 6.6M   | 256              |          |
  | HaloNetH1 | 9.1M   | 256              |          |
  | HaloNetH2 | 10.3M  | 256              |          |
  | HaloNetH3 | 12.5M  | 320              |          |
  | HaloNetH4 | 19.5M  | 384              | 85.5     |
  | HaloNetH5 | 31.6M  | 448              |          |
  | HaloNetH6 | 44.3M  | 512              |          |
  | HaloNetH7 | 67.9M  | 640              |          |

  Comparing `halo_b7` accuracy by replacing Conv layers with Attention in each stage:

  | Conv Stages | Attention Stages | Top-1 Acc (%) | Norm. Train Time |
  |:-----------:|:----------------:|:-------------:|:----------------:|
  |      -      |    1, 2, 3, 4    |     84.9      |       1.9        |
  |      1      |     2, 3, 4      |     84.6      |       1.4        |
  |    1, 2     |       3, 4       |     84.7      |       1.0        |
  |   1, 2, 3   |        4         |     83.8      |       0.5        |
## Usage
  ```py
  from keras_cv_attention_models import halonet

  # No pretraind available.
  mm = halonet.HaloNetH1()
  mm.summary()
  ```
***
