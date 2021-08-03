# Keras_halonet
***

## Summary
  - [Github lucidrains/halonet-pytorch](https://github.com/lucidrains/halonet-pytorch)
  - HaloAttention article: [PDF 2103.12731 Scaling Local Self-Attention for Parameter Efficient Visual Backbones](https://arxiv.org/pdf/2103.12731.pdf)
  - No pretraind available.
## Models
  | Model   | params | Image resolution | Top1 Acc |
  | ------- | ------ | ---------------- | -------- |
  | halo_b0 | 4.6M   | 256              |          |
  | halo_b1 | 8.8M   | 256              |          |
  | halo_b2 | 11.04M | 256              |          |
  | halo_b3 | 15.1M  | 320              |          |
  | halo_b4 | 31.4M  | 384              | 85.5%    |
  | halo_b5 | 34.4M  | 448              |          |
  | halo_b6 | 47.98M | 512              |          |
  | halo_b7 | 68.4M  | 600              |          |

  Comparing `halo_b7` accuracy by replacing Conv layers with Attention in each stage:

  | Conv Stages | Attention Stages | Top-1 Acc (%) | Norm. Train Time |
  | ----------- | ---------------- | ------------- | ---------------- |
  | -           | 1, 2, 3, 4       | 84.9          | 1.9              |
  | 1           | 2, 3, 4          | 84.6          | 1.4              |
  | 1, 2        | 3, 4             | 84.7          | 1.0              |
  | 1, 2, 3     | 4                | 83.8          | 0.5              |
## Usage
  ```py
  from keras_cv_attention_models import halonet

  # No pretraind available.
  mm = halonet.HaloNetB0()
  mm.summary()
  ```
***
