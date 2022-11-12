# ___Keras CoAtNet___
***

## Summary
  - CoAtNet article: [PDF 2106.04803 CoAtNet: Marrying Convolution and Attention for All Data Sizes](https://arxiv.org/pdf/2106.04803.pdf)
  - [Github comvex/coatnet/model.py](https://github.com/blakechi/ComVEX/blob/master/comvex/coatnet/model.py)
  - Architecture is guessed from article, so it's NOT certain.
  - **`CoAtNet` is using `vv_dim = key_dim` instead of previous `vv_dim = out_shape // num_heads` now, and pretrained weights updated, be caution of this update if wanna reload earlier models**. - `2022.04.24`

  ![](https://user-images.githubusercontent.com/5744524/151656800-1baab0ad-a31b-4ef0-bada-483c83108670.png)
***

## Usage
  ```py
  from keras_cv_attention_models import coatnet

  # Only CoAtNet0 pretrained available.
  mm = coatnet.CoAtNet0()

  # Run prediction
  import tensorflow as tf
  from skimage.data import chelsea
  imm = tf.keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(tf.keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.99324566), ('n02123159', 'tiger_cat', 0.00381939), ... ]
  ```
## Training
  - As model structure not certain, these are most tests.
  - **Model structure**
    - `V1` means `ResNetV1` like. Conv shortcut branch: `output = conv_shortcut(input) + block(prenorm(input))`. Identity branch: `output = input + block(prenorm(input))`.
    - `V2` means `ResNetV2` like. Conv shortcut branch: `prenorm_input = prenorm(input), output = conv_shortcut(prenorm_input) + block(prenorm_input)`. Identity branch: `output = input + block(prenorm(input))`.
    - `wd exc pos_emb` means `optimizer weight decay excluding positional embedding`.

    | Model                    | stem              | res_MBConv block   | res_mhsa block     | res_ffn block     | Best top1  |
    | ------------------------ | ----------------- | ------------------ | ------------------ | ----------------- | ---------- |
    | CoAtNet0_8               | conv,bn,gelu,conv | prenorm bn+gelu,V2 | prenorm bn+gelu,V2 | bn,conv,gelu,conv | 0.8010     |
    | CoAtNet0_11              | conv,bn,gelu,conv | prenorm bn,V2      | prenorm bn,V2      | bn,conv,gelu,conv | 0.8016     |
    | CoAtNet0_15              | conv,bn,gelu,conv | prenorm bn,V2      | prenorm ln,V2      | ln,conv,gelu,conv | 0.7999     |
    | CoAtNet0_16              | conv,bn,gelu,conv | prenorm bn,V1      | prenorm ln,V1      | ln,conv,gelu,conv | 0.8019     |
    | - drop_connect 0.05      | conv,bn,gelu,conv | prenorm bn,V1      | prenorm ln,V1      | ln,conv,gelu,conv | 0.8017     |
    | - wd exc pos_emb         | conv,bn,gelu,conv | prenorm bn,V1      | prenorm ln,V1      | ln,conv,gelu,conv | **0.8048** |
    | - wd exc pos_emb, mag 10 | conv,bn,gelu,conv | prenorm bn,V1      | prenorm ln,V1      | ln,conv,gelu,conv | 0.8024     |
  - **Training**. Using `A3` recipe with `batch_size=128, input_shape=(160, 160)`. Weight decay excluding positional embedding is default behavior now.
    ```sh
    CUDA_VISIBLE_DEVICES='0' TF_XLA_FLAGS="--tf_xla_auto_jit=2" ./train_script.py -m coatnet.CoAtNet0 \
    --seed 0 --batch_size 128 -s CoAtNet0_160
    ```
    Evaluating on 224 input resolution without fine-tune:
    ```sh
    CUDA_VISIBLE_DEVICES='1' ./eval_script.py -m coatnet.CoAtNet0 --pretrained checkpoints/CoAtNet0_160_latest.h5 -b 8 -i 224
    # >>>> Accuracy top1: 0.81056 top5: 0.95362
    ```
  - **Plot**
    ![coatnet0_160](https://user-images.githubusercontent.com/5744524/154603658-a96aa137-167a-47a8-987f-ec5599a289f8.png)
  - **Fine-tuning 160 -> 224, 37 epochs**
    ```sh
    CUDA_VISIBLE_DEVICES='0' TF_XLA_FLAGS='--tf_xla_auto_jit=2' ./train_script.py --seed 0 \
    -m coatnet.CoAtNet0 --pretrained checkpoints/CoAtNet0_160_latest.h5 -i 224 --batch_size 64 \
    --lr_decay_steps 32 --lr_warmup_steps 0 --lr_base_512 0.004 \
    --additional_model_kwargs '{"drop_connect_rate": 0.05}' --magnitude 15 \
    -s coatnet.CoAtNet0_ft_224_lr_steps_32_lr4e3_drc005_magnitude_15
    ```
    | magnitude          | drop_connect_rate | Best val loss, acc                                                              |
    | ------------------ | ----------------- | ------------------------------------------------------------------------------- |
    | 6                  | 0                 | Epoch 35/37 loss: 0.0023 - acc: 0.7288 - val_loss: 0.0012 - val_acc: 0.8160     |
    | 7                  | 0                 | Epoch 34/37 loss: 0.0024 - acc: 0.7218 - val_loss: 0.0012 - val_acc: 0.8161     |
    | 7                  | 0.05              | Epoch 36/37 loss: 0.0026 - acc: 0.7026 - val_loss: 0.0011 - val_acc: 0.8193     |
    | 7                  | 0.2               | Epoch 34/37 loss: 0.0030 - acc: 0.6658 - val_loss: 0.0011 - val_acc: 0.8176     |
    | 10                 | 0.05              | Epoch 36/37 loss: 0.0028 - acc: 0.6783 - val_loss: 0.0011 - val_acc: 0.8199     |
    | 10, wd exc pos_emb | 0.05              | Epoch 35/37 loss: 0.0028 - acc: 0.6811 - val_loss: 0.0011 - val_acc: 0.8206     |
    | 15, wd exc pos_emb | 0.05              | Epoch 36/37 loss: 0.0028 - acc: 0.6796 - val_loss: 0.0011 - val_acc: **0.8221** |

    ![coatnet0_ft_224](https://user-images.githubusercontent.com/5744524/157171155-5eacb713-62c0-420a-bb63-57644ab9f0ec.png)
  - **Training 224 for 305 epochs**
    ```sh
    CUDA_VISIBLE_DEVICES='1' TF_XLA_FLAGS='--tf_xla_auto_jit=2' ./train_script.py --seed 0 \
    -m coatnet.CoAtNet0 --batch_size 128 -i 224 --lr_decay_steps 300 \
    --magnitude 15 --additional_model_kwargs'{"drop_connect_rate": 0.1}' -s CoAtNet0_224_decay_300
    ```
    ![coatnet0_300](https://user-images.githubusercontent.com/5744524/201463307-7e991fb5-a745-414a-930c-f623412533d9.png)
## Models
  - Self defined models are using `Stride-2 DConv2D` by default. Set `use_dw_strides=False` for using `strides=2` in `Conv2D` layer instead.

  | Model                               | Params | FLOPs  | Input | Top1 Acc | Download |
  | ----------------------------------- | ------ | ------ | ----- | -------- | -------- |
  | CoAtNet0 (Self trained 105 epochs)  | 23.8M  | 2.17G  | 160   | 80.48    | [coatnet0_160_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/coatnet/coatnet0_160_imagenet.h5) |
  | CoAtNet0 (Self trained 305 epochs)  | 23.8M  | 4.22G  | 224   | 82.79    | [coatnet0_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/coatnet/coatnet0_224_imagenet.h5) |
  | CoAtNet0                            | 25M    | 4.2G   | 224   | 81.6     |          |
  | CoAtNet0, Stride-2 DConv2D          | 25M    | 4.6G   | 224   | 82.0     |          |
  | CoAtNet0                            | 25M    | 13.4G  | 384   | 83.9     |          |
  | CoAtNet1                            | 42M    | 8.4G   | 224   | 83.3     |          |
  | CoAtNet1, Stride-2 DConv2D          | 42M    | 8.8G   | 224   | 83.5     |          |
  | CoAtNet1                            | 42M    | 27.4G  | 384   | 85.1     |          |
  | CoAtNet2                            | 75M    | 15.7G  | 224   | 84.1     |          |
  | CoAtNet2, Stride-2 DConv2D          | 75M    | 16.6G  | 224   | 84.1     |          |
  | CoAtNet2                            | 75M    | 49.8G  | 384   | 85.7     |          |
  | CoAtNet2                            | 75M    | 96.7G  | 512   | 85.9     |          |
  | CoAtNet2, ImageNet-21k pretrain     | 75M    | 16.6G  | 224   | 87.1     |          |
  | CoAtNet2, ImageNet-21k pretrain     | 75M    | 49.8G  | 384   | 87.1     |          |
  | CoAtNet2, ImageNet-21k pretrain     | 75M    | 96.7G  | 512   | 87.3     |          |
  | CoAtNet3                            | 168M   | 34.7G  | 224   | 84.5     |          |
  | CoAtNet3                            | 168M   | 107.4G | 384   | 85.8     |          |
  | CoAtNet3                            | 168M   | 203.1G | 512   | 86.0     |          |
  | CoAtNet3, ImageNet-21k pretrain     | 168M   | 34.7G  | 224   | 87.6     |          |
  | CoAtNet3, ImageNet-21k pretrain     | 168M   | 107.4G | 384   | 87.6     |          |
  | CoAtNet3, ImageNet-21k pretrain     | 168M   | 203.1G | 512   | 87.9     |          |
  | CoAtNet4, ImageNet-21k pretrain     | 275M   | 189.5G | 384   | 87.9     |          |
  | CoAtNet4, ImageNet-21k pretrain     | 275M   | 360.9G | 512   | 88.1     |          |
  | CoAtNet4, ImageNet-21K + PT-RA-E150 | 275M   | 189.5G | 384   | 88.4     |          |
  | CoAtNet4, ImageNet-21K + PT-RA-E150 | 275M   | 360.9G | 512   | 88.56    |          |

  **JFT pre-trained models accuracy**

  | Model                      | Input | Reported Params    | self-defined Params    | Top1 Acc |
  | -------------------------- | ----- | ------------------ | ---------------------- | -------- |
  | CoAtNet3, Stride-2 DConv2D | 384   | 168M, FLOPs 114G   | 160.64M, FLOPs 109.67G | 88.52    |
  | CoAtNet3, Stride-2 DConv2D | 512   | 168M, FLOPs 214G   | 161.24M, FLOPs 205.06G | 88.81    |
  | CoAtNet4                   | 512   | 275M, FLOPs 361G   | 270.69M, FLOPs 359.77G | 89.11    |
  | CoAtNet5                   | 512   | 688M, FLOPs 812G   | 676.23M, FLOPs 807.06G | 89.77    |
  | CoAtNet6                   | 512   | 1.47B, FLOPs 1521G | 1.336B, FLOPs 1470.56G | 90.45    |
  | CoAtNet7                   | 512   | 2.44B, FLOPs 2586G | 2.413B, FLOPs 2537.56G | 90.88    |
## Article detail info
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

  | Finetuning Hyper-parameter | ImageNet-1K (CoAtNet-0/1/2/3) | ImageNet-21K (CoAtNet-2/3/4) | JFT (CoAtNet-3/4/5)             |
  | -------------------------- | ----------------------------- | ---------------------------- | ------------------------------- |
  | Stochastic depth rate      | 0.2 / 0.3 / 0.5 / 0.7         | 0.3 / 0.5 / 0.7              | 0.1 / 0.3 / 0.2                 |
  | RandAugment                | 2, 15                         | 2, 5                         | 2, 5                            |
  | Mixup alpha                | 0.8                           | None                         | None                            |

  - At the beginning of each stage, we always reduce the spatial size by 2x and increase the number of channels.
  - The first stage S0 is a simple 2-layer convolutional Stem.
  - S1 always employs MBConv blocks with squeeze-excitation (SE), as the spatial size is too large for global attention.
  - Starting from S2 through S4, we consider either the MBConv or the Transformer block, with a constraint that convolution stages must appear before Transformer stages.
  - The constraint is based on the prior that convolution is better at processing local patterns that are more common in early stages.
  - This leads to 4 variants with increasingly more Transformer stages, C-C-C-C, C-C-C-T, C-C-T-T and C-T-T-T, where C and T denote Convolution and Transformer respectively.
  - Also, for simplicity, when increasing the depth of the network, we only scale the number of blocks in S2 and S3.
  - Down-sampling in the MBConv block is achieved by stride-2 Depthwise Convolution.
  - Lastly, we study two choices of model details, namely the dimension of each attention (default to 32) head as well as the type of normalization (default to BatchNorm) used in MBConv blocks.
  - On the other hand, BatchNorm and LayerNorm have almost the same performance, while BatchNorm is 10 - 20% faster on TPU depending on the per-core batch size.
  - while Norm corresponds to BatchNorm for MBConv and LayerNorm for Self-Attention and FFN.
  - We have experimented with using LayerNorm in the MBConv block, which achieves the same performance while being significantly slower on our accelerator (TPU).
  - In general, we recommend whichever is faster on your device.
  - Following the same spirit, Gaussian Error Linear Units (GELUs) is used as the activation function in both the MBConv blocks and Transformer blocks.
  - Relative attention. Under the general name of relative attention, there have been various variants in literature. Generally speaking, we can separate them into two categories:
  - (a) the input-dependent version where the extra relative attention score is a function of the input states f(xi , xj , i − j)
  - and (b) the input-independent version f(i − j).
  - The variant in CoAtNet belongs to the input-independent version.
  - PT-RA denotes applying RandAugment during 21K pre-training, and E150 means 150 epochs of 21K pre-training,
