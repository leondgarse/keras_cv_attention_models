# ___Keras CoAtNet___
***

## Summary
- CoAtNet article: [PDF 2106.04803 CoAtNet: Marrying Convolution and Attention for All Data Sizes](https://arxiv.org/pdf/2106.04803.pdf)
- [Github comvex/coatnet/model.py](https://github.com/blakechi/ComVEX/blob/master/comvex/coatnet/model.py)
- No pretraind available. Architecture is guessed from article, so it's NOT certain.
***

## Models
  ![](coatnet.png)

  | Model                               | Params | Image resolution | Top1 Acc |
  | ----------------------------------- | ------ | ---------------- | -------- |
  | CoAtNet0                            | 25M    | 224              | 81.6     |
  | CoAtNet0                            | 25M    | 384              | 83.9     |
  | CoAtNet1                            | 42M    | 224              | 83.3     |
  | CoAtNet1                            | 42M    | 384              | 85.1     |
  | CoAtNet2                            | 75M    | 224              | 84.1     |
  | CoAtNet2                            | 75M    | 384              | 85.7     |
  | CoAtNet2                            | 75M    | 512              | 85.9     |
  | CoAtNet2, ImageNet-21k pretrain     | 75M    | 224              | 87.1     |
  | CoAtNet2, ImageNet-21k pretrain     | 75M    | 384              | 87.1     |
  | CoAtNet2, ImageNet-21k pretrain     | 75M    | 512              | 87.3     |
  | CoAtNet3                            | 168M   | 224              | 84.5     |
  | CoAtNet3                            | 168M   | 384              | 85.8     |
  | CoAtNet3                            | 168M   | 512              | 86.0     |
  | CoAtNet3, ImageNet-21k pretrain     | 168M   | 224              | 87.6     |
  | CoAtNet3, ImageNet-21k pretrain     | 168M   | 384              | 87.6     |
  | CoAtNet3, ImageNet-21k pretrain     | 168M   | 512              | 87.9     |
  | CoAtNet4, ImageNet-21k pretrain     | 275M   | 384              | 87.9     |
  | CoAtNet4, ImageNet-21k pretrain     | 275M   | 512              | 88.1     |
  | CoAtNet4, ImageNet-21K + PT-RA-E150 | 275M   | 384              | 88.4     |
  | CoAtNet4, ImageNet-21K + PT-RA-E150 | 275M   | 512              | 88.56    |

  **JFT pre-trained models accuracy**

  | Model    | Image resolution | Reported Params | self-defined Params | Top1 Acc |
  | -------- | ---------------- | --------------- | ------------------- | -------- |
  | CoAtNet3 | 384              | 168M            | 162.96M             | 88.52    |
  | CoAtNet3 | 512              | 168M            | 163.57M             | 88.81    |
  | CoAtNet4 | 512              | 275M            | 273.10M             | 89.11    |
  | CoAtNet5 | 512              | 688M            | 680.47M             | 89.77    |
  | CoAtNet6 | 512              | 1.47B           | 1.340B              | 90.45    |
  | CoAtNet7 | 512              | 2.44B           | 2.422B              | 90.88    |
## Usage
  ```py
  from keras_cv_attention_models import coatnet

  # No pretraind available.
  mm = coatnet.CoAtNet0()
  mm.summary()
  ```
***
- L denotes the number of blocks and D denotes the hidden dimension (#channels).
- For all Conv and MBConv blocks, we always use the kernel size 3.
- For all Transformer blocks, we set the size of each attention head to 32, following [22].
- The expansion rate for the inverted bottleneck is always 4 and the expansion (shrink) rate for the SE is always 0.25.

| Stages    | Size | CoAtNet-0 | CoAtNet-1  | CoAtNet-2  | CoAtNet-3  | CoAtNet-4  |
| --------- | ---- | --------- | ---------- | ---------- | ---------- | ---------- |
| S0-Conv   | 1/2  | L=2 D=64  | L=2 D=64   | L=2 D=128  | L=2 D=192  | L=2 D=192  |
| S1-MbConv | 1/4  | L=2 D=96  | L=2 D=96   | L=2 D=128  | L=2 D=192  | L=2 D=192  |
| S2-MBConv | 1/8  | L=3 D=192 | L=6 D=192  | L=6 D=256  | L=6 D=384  | L=12 D=384 |
| S3-TFMRel | 1/16 | L=5 D=384 | L=14 D=384 | L=14 D=512 | L=14 D=768 | L=28 D=768 |
| S4-TFMRel | 1/32 | L=2 D=768 | L=2 D=768  | L=2 D=1024 | L=2 D=1536 | L=2 D=1536 |

- At the beginning of each stage, we always reduce the spatial size by 2x and increase the number of channels.
- The first stage S0 is a simple 2-layer convolutional Stem.
- S1 always employs MBConv blocks with squeeze-excitation (SE), as the spatial size is too large for global attention.
- Starting from S2 through S4, we consider either the MBConv or the Transformer block, with a constraint that convolution stages must appear before Transformer stages.
- The constraint is based on the prior that convolution is better at processing local patterns that are more common in early stages.
- This leads to 4 variants with increasingly more Transformer stages, C-C-C-C, C-C-C-T, C-C-T-T and C-T-T-T, where C and T denote Convolution and Transformer respectively.
- Also, for simplicity, when increasing the depth of the network, we only scale the number of blocks in S2 and S3.
- Down-sampling in the MBConv block is achieved by stride-2 Depthwise Convolution.

- Lastly, we study two choices of model details, namely the dimension of each attention (default to 32) head as well as the type of normalization (default to BatchNorm) used in MBConv blocks.
- From Table 8, we can see increasing head size from 32 to 64 can slightly hurt performance, though it actually improves the TPU speed by a significant amount.
- In practice, this will be a quality-speed trade-off one can make.
- On the other hand, BatchNorm and LayerNorm have almost the same performance, while BatchNorm is 10 - 20% faster on TPU depending on the per-core batch size.

- while Norm corresponds to BatchNorm for MBConv and LayerNorm for Self-Attention and FFN.
- We have experimented with using LayerNorm in the MBConv block, which achieves the same performance while being significantly slower on our accelerator (TPU).
- In general, we recommend whichever is faster on your device.
- Following the same spirit, Gaussian Error Linear Units (GELUs) [50] is used as the activation function in both the MBConv blocks and Transformer blocks.

- Relative attention. Under the general name of relative attention, there have been various variants in literature [29, 37, 38, 33, 39, 30]. Generally speaking, we can separate them into two categories:
- (a) the input-dependent version where the extra relative attention score is a function of the input states f(xi , xj , i − j)
- and (b) the input-independent version f(i − j).
- The variant in CoAtNet belongs to the input-independent version.

- PT-RA denotes applying RandAugment during 21K pre-training, and E150 means 150 epochs of 21K pre-training,
