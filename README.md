# Keras_cv_attention_models
<!-- TOC depthFrom:1 depthTo:6 withLinks:1 updateOnSave:1 orderedList:0 -->

- [Keras_cv_attention_models](#kerascvattentionmodels)
- [Usage](#usage)
	- [Basic Usage](#basic-usage)
	- [Layers](#layers)
	- [Model surgery](#model-surgery)
	- [AotNet](#aotnet)
	- [BotNet](#botnet)
	- [VOLO](#volo)
	- [ResNeSt](#resnest)
	- [ResNeXt](#resnext)
	- [HaloNet](#halonet)
	- [CoTNet](#cotnet)
	- [CoAtNet](#coatnet)
	- [CoaT](#coat)
	- [MLP mixer](#mlp-mixer)
	- [ResMLP](#resmlp)
	- [GMLP](#gmlp)
- [Other implemented keras models](#other-implemented-keras-models)

<!-- /TOC -->
***

# Usage
## Basic Usage
  - Install as pip package:
    ```sh
    pip install -U git+https://github.com/leondgarse/keras_cv_attention_models
    ```
    Refer to each sub directory for detail usage.
  - **Basic model prediction**
    ```py
    from keras_cv_attention_models import volo
    mm = volo.VOLO_d1(pretrained="imagenet")

    """ Run predict """
    import tensorflow as tf
    from tensorflow import keras
    from skimage.data import chelsea
    img = chelsea() # Chelsea the cat
    imm = keras.applications.imagenet_utils.preprocess_input(img, mode='torch')
    pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
    pred = tf.nn.softmax(pred).numpy()  # If classifier activation is not softmax
    print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
    # [('n02124075', 'Egyptian_cat', 0.9692954),
    #  ('n02123045', 'tabby', 0.020203391),
    #  ('n02123159', 'tiger_cat', 0.006867502),
    #  ('n02127052', 'lynx', 0.00017674894),
    #  ('n02123597', 'Siamese_cat', 4.9493494e-05)]
    ```
  - **Exclude model top layers by set `num_classes=0`**
    ```py
    from keras_cv_attention_models import resnest
    mm = resnest.ResNest50(num_classes=0)
    print(mm.output_shape)
    # (None, 7, 7, 2048)
    ```
## Layers
  - [attention_layers](keras_cv_attention_models/attention_layers) is `__init__.py` only, which imports core layers defined in model architectures. Like `MHSAWithPositionEmbedding` from `botnet`, `HaloAttention` from `halonet`.
  ```py
  from keras_cv_attention_models import attention_layers
  aa = attention_layers.MHSAWithPositionEmbedding(num_heads=4, key_dim=128, relative=True)
  print(f"{aa(tf.ones([1, 14, 16, 256])).shape = }")
  # aa(tf.ones([1, 14, 16, 256])).shape = TensorShape([1, 14, 16, 512])
  ```
## Model surgery
  - [model_surgery](keras_cv_attention_models/model_surgery) including functions used to change model parameters after built.
  ```py
  from keras_cv_attention_models import model_surgery
  # Replace all ReLU with PReLU
  mm = model_surgery.replace_ReLU(keras.applications.ResNet50(), target_activation='PReLU')
  ```
## AotNet
  - [Keras AotNet](keras_cv_attention_models/aotnet) is just a `ResNet` / `ResNetV2` like framework, that set parameters like `attn_types` and `se_ratio` and others, which is used to apply different types attention layer.
    ```py
    # Mixing se and outlook and halo and mhsa and cot_attention, 21M parameters
    # 50 is just a picked number that larger than the relative `num_block`
    from keras_cv_attention_models import aotnet
    attn_types = [None, "outlook", ["mhsa", "halo"] * 50, "cot"]
    se_ratio = [0.25, 0, 0, 0]
    mm = aotnet.AotNet50V2(attn_types=attn_types, se_ratio=se_ratio, deep_stem=True, strides=1)
    ```
## BotNet
  - [Keras BotNet](keras_cv_attention_models/botnet) is for [PDF 2101.11605 Bottleneck Transformers for Visual Recognition](https://arxiv.org/pdf/2101.11605.pdf).

  | Model        | Params | Image resolution | Top1 Acc | Download            |
  | ------------ | ------ | ---------------- | -------- | ------------------- |
  | botnet50     | 21M    | 224              | 77.604   | [botnet50.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/botnet/botnet50.h5)  |
  | botnet101    | 41M    | 224              |          |  |
  | botnet152    | 56M    | 224              |          |  |
## VOLO
  - [Keras VOLO](keras_cv_attention_models/volo) is for [PDF 2106.13112 VOLO: Vision Outlooker for Visual Recognition](https://arxiv.org/pdf/2106.13112.pdf).

  | Model        | Params | Image resolution | Top1 Acc | Download            |
  | ------------ | ------ | ---------------- | -------- | ------------------- |
  | volo_d1      | 27M    | 224              | 84.2     | [volo_d1_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d1_224.h5)  |
  | volo_d1 ↑384 | 27M    | 384              | 85.2     | [volo_d1_384.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d1_384.h5)  |
  | volo_d2      | 59M    | 224              | 85.2     | [volo_d2_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d2_224.h5)  |
  | volo_d2 ↑384 | 59M    | 384              | 86.0     | [volo_d2_384.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d2_384.h5)  |
  | volo_d3      | 86M    | 224              | 85.4     | [volo_d3_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d3_224.h5)  |
  | volo_d3 ↑448 | 86M    | 448              | 86.3     | [volo_d3_448.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d3_448.h5)  |
  | volo_d4      | 193M   | 224              | 85.7     | [volo_d4_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d4_224.h5)  |
  | volo_d4 ↑448 | 193M   | 448              | 86.8     | [volo_d4_448.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d4_448.h5) |
  | volo_d5      | 296M   | 224              | 86.1     | [volo_d5_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d5_224.h5) |
  | volo_d5 ↑448 | 296M   | 448              | 87.0     | [volo_d5_448.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d5_448.h5) |
  | volo_d5 ↑512 | 296M   | 512              | 87.1     | [volo_d5_512.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d5_512.h5) |
## ResNeSt
  - [Keras ResNeSt](keras_cv_attention_models/resnest) is for [PDF 2004.08955 ResNeSt: Split-Attention Networks](https://arxiv.org/pdf/2004.08955.pdf).

  | Model          | Params | Image resolution | Top1 Acc | Download            |
  | -------------- | ------ | ----------------- | -------- | ------------------- |
  | resnest50      | 28M    | 224               | 81.03    | [resnest50.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnest/resnest50.h5)  |
  | resnest101     | 49M    | 256               | 82.83    | [resnest101.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnest/resnest101.h5)  |
  | resnest200     | 71M    | 320               | 83.84    | [resnest200.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnest/resnest200.h5)  |
  | resnest269     | 111M   | 416               | 84.54    | [resnest269.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnest/resnest269.h5)  |
## ResNeXt
  - [Keras ResNeXt](keras_cv_attention_models/resnext) is for [PDF 1611.05431 Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf).

  | Model          | Params | Image  resolution | Top1 Acc | Download            |
  | -------------- | ------ | ----------------- | -------- | ------------------- |
  | resnext50      | 25M    | 224               | 77.74    | [resnext50.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnext/resnext50.h5)  |
  | resnext101     | 42M    | 224               | 78.73    | [resnext101.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnext/resnext101.h5)  |
## HaloNet
  - [Keras HaloNet](keras_cv_attention_models/halonet) is for [PDF 2103.12731 Scaling Local Self-Attention for Parameter Efficient Visual Backbones](https://arxiv.org/pdf/2103.12731.pdf).

  | Model   | Params | Image resolution | Top1 Acc |
  | ------- | ------ | ---------------- | -------- |
  | halo_b0 | 4.6M   | 256              |          |
  | halo_b1 | 8.8M   | 256              |          |
  | halo_b2 | 11.04M | 256              |          |
  | halo_b3 | 15.1M  | 320              |          |
  | halo_b4 | 31.4M  | 384              | 85.5     |
  | halo_b5 | 34.4M  | 448              |          |
  | halo_b6 | 47.98M | 512              |          |
  | halo_b7 | 68.4M  | 600              |          |
## CoTNet
  - [Keras CoTNet](keras_cv_attention_models/cotnet) is for [PDF 2107.12292 Contextual Transformer Networks for Visual Recognition](https://arxiv.org/pdf/2107.12292.pdf).

  | Model          | Params | Image resolution | FLOPs | Top1 Acc | Download            |
  | -------------- |:------:| ---------------- | ----- |:--------:| ------------------- |
  | CoTNet-50      | 22.2M  | 224              | 3.3   |   81.3   | [cotnet50_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cotnet/cotnet50_224.h5) |
  | CoTNeXt-50     | 30.1M  | 224              | 4.3   |   82.1   |  |
  | SE-CoTNetD-50  | 23.1M  | 224              | 4.1   |   81.6   | [se_cotnetd50_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cotnet/se_cotnetd50_224.h5) |
  | CoTNet-101     | 38.3M  | 224              | 6.1   |   82.8   | [cotnet101_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cotnet/cotnet101_224.h5) |
  | CoTNeXt-101    | 53.4M  | 224              | 8.2   |   83.2   |  |
  | SE-CoTNetD-101 | 40.9M  | 224              | 8.5   |   83.2   | [se_cotnetd101_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cotnet/se_cotnetd101_224.h5) |
  | SE-CoTNetD-152 | 55.8M  | 224              | 17.0  |   84.0   | [se_cotnetd152_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cotnet/se_cotnetd152_224.h5) |
  | SE-CoTNetD-152 | 55.8M  | 320              | 26.5  |   84.6   | [se_cotnetd152_320.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cotnet/se_cotnetd152_320.h5) |
## CoAtNet
  - [Keras CoAtNet](keras_cv_attention_models/coatnet) is for [PDF 2106.04803 CoAtNet: Marrying Convolution and Attention for All Data Sizes](https://arxiv.org/pdf/2106.04803.pdf).

  | Model                                | Params | Image resolution | Top1 Acc |
  | ------------------------------------ | ------ | ---------------- | -------- |
  | CoAtNet-0                            | 25M    | 224              | 81.6     |
  | CoAtNet-1                            | 42M    | 224              | 83.3     |
  | CoAtNet-2                            | 75M    | 224              | 84.1     |
  | CoAtNet-2, ImageNet-21k pretrain     | 75M    | 224              | 87.1     |
  | CoAtNet-3                            | 168M   | 224              | 84.5     |
  | CoAtNet-3, ImageNet-21k pretrain     | 168M   | 224              | 87.6     |
  | CoAtNet-3, ImageNet-21k pretrain     | 168M   | 512              | 87.9     |
  | CoAtNet-4, ImageNet-21k pretrain     | 275M   | 512              | 88.1     |
  | CoAtNet-4, ImageNet-21K + PT-RA-E150 | 275M   | 512              | 88.56    |
## CoaT
  - [Keras CoaT](keras_cv_attention_models/coat) is for [PDF 2104.06399 CoaT: Co-Scale Conv-Attentional Image Transformers](http://arxiv.org/abs/2104.06399).

  | Model         | Params | Image resolution | Top1 Acc | Download |
  | ------------- | ------ | ---------------- | -------- | -------- |
  | CoaTLiteTiny  | 5.7M   | 224              | 77.5     | [coat_lite_tiny_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/coat/coat_lite_tiny_imagenet.h5) |
  | CoaTLiteMini  | 11M    | 224              | 79.1     | [coat_lite_mini_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/coat/coat_lite_mini_imagenet.h5) |
  | CoaTLiteSmall | 20M    | 224              | 81.9     | [coat_lite_small_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/coat/coat_lite_small_imagenet.h5) |
  | CoaTTiny      | 5.5M   | 224              | 78.3     | [coat_tiny_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/coat/coat_tiny_imagenet.h5) |
  | CoaTMini      | 10M    | 224              | 81.0     | [coat_mini_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/coat/coat_mini_imagenet.h5) |
## MLP mixer
  - [Keras mlp](keras_cv_attention_models/mlp_family#mlp-mixer) includes implementation of [PDF 2105.01601 MLP-Mixer: An all-MLP Architecture for Vision](https://arxiv.org/pdf/2105.01601.pdf).
  - **Models** `Top1 Acc` is `Pre-trained on JFT-300M` model accuray on `ImageNet 1K` from paper.

  | Model       | Params | Top1 Acc | ImageNet | Imagenet21k | ImageNet SAM |
  | ----------- | ------ | -------- | --------------- | ------------------ | ------------------- |
  | MLPMixerS32 | 19.1M  | 68.70    |                 |                    |                     |
  | MLPMixerS16 | 18.5M  | 73.83    |                 |                    |                     |
  | MLPMixerB32 | 60.3M  | 75.53    |                 |                    | [b32_imagenet_sam.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/mlp_mixer_b32_imagenet_sam.h5) |
  | MLPMixerB16 | 59.9M  | 80.00    | [b16_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/mlp_mixer_b16_imagenet.h5) | [b16_imagenet21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/mlp_mixer_b16_imagenet21k.h5) | [b16_imagenet_sam.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/mlp_mixer_b16_imagenet_sam.h5) |
  | MLPMixerL32 | 206.9M | 80.67    |  |  |                     |
  | MLPMixerL16 | 208.2M | 84.82    | [l16_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/mlp_mixer_l16_imagenet.h5) | [l16_imagenet21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/mlp_mixer_l16_imagenet21k.h5) |                     |
  | - input 448 | 208.2M | 86.78    |                 |                    |                     |
  | MLPMixerH14 | 432.3M | 86.32    |                 |                    |                     |
  | - input 448 | 432.3M | 87.94    |                 |                    |                     |
## ResMLP
  - [Keras mlp](keras_cv_attention_models/mlp_family#resmlp) includes implementation of [PDF 2105.03404 ResMLP: Feedforward networks for image classification with data-efficient training](https://arxiv.org/pdf/2105.03404.pdf)

  | Model      | Params | Image resolution | Top1 Acc | ImageNet |
  | ---------- | ------ | ---------------- | -------- | -------- |
  | ResMLP12   | 15M    | 224              | 77.8     | [resmlp12_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/resmlp12_imagenet.h5) |
  | ResMLP24   | 30M    | 224              | 80.8     | [resmlp24_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/resmlp24_imagenet.h5) |
  | ResMLP36   | 116M   | 224              | 81.1     | [resmlp36_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/resmlp36_imagenet.h5) |
  | ResMLP_B24 | 129M   | 224              | 83.6     | [resmlp_b24_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/resmlp_b24_imagenet.h5) |
  | - imagenet22k | 129M   | 224              | 84.4     | [resmlp_b24_imagenet22k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/resmlp_b24_imagenet22k.h5) |
## GMLP
  - [Keras mlp](keras_cv_attention_models/mlp_family#gmlp) includes implementation of [PDF 2105.08050 Pay Attention to MLPs](https://arxiv.org/pdf/2105.08050.pdf).

  | Model      | Params | Image resolution | Top1 Acc | ImageNet |
  | ---------- | ------ | ---------------- | -------- | -------- |
  | GMLPTiny16 | 6M     | 224              | 72.3     |          |
  | GMLPS16    | 20M    | 224              | 79.6     | [gmlp_s16_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp_family/gmlp_s16_imagenet.h5) |
  | GMLPB16    | 73M    | 224              | 81.6     |          |
***

# Other implemented keras models
  - [Github ypeleg/nfnets-keras](https://github.com/ypeleg/nfnets-keras)
  - [Github faustomorales/vit-keras](https://github.com/faustomorales/vit-keras)
  - [Github tensorflow/resnet_rs](https://github.com/tensorflow/tpu/tree/master/models/official/resnet/resnet_rs)
***
