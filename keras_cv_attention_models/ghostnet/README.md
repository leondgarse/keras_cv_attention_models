# ___Keras GhostNetV2___
***

## Summary
  - Keras implementation of [Gitee mindspore/models/ghostnetv2](https://gitee.com/mindspore/models/tree/master/research/cv/ghostnetv2). Paper [PDF GhostNetV2: Enhance Cheap Operation with Long-Range Attention](https://openreview.net/pdf/6db544c65bbd0fa7d7349508454a433c112470e2.pdf).
  - `GhostNetV2_100` model weights ported from official publication [download.mindspore ghostnetv2](https://download.mindspore.cn/model_zoo/research/cv/ghostnetv2/).
  - `GhostNet_100` model weights ported from official publication [Github huawei-noah/ghostnet_pytorch](https://github.com/huawei-noah/Efficient-AI-Backbones/tree/master/ghostnet_pytorch). Paper [PDF 1911.11907 GhostNet: More Features from Cheap Operations](https://arxiv.org/pdf/1911.11907.pdf).
  - `GhostNet_050` and `GhostNet_130` model weights ported from [Github PaddlePaddle/PaddleClas](https://github.com/PaddlePaddle/PaddleClas).

  ![ghostnetv2](https://user-images.githubusercontent.com/5744524/202699896-4c429db1-8038-4dc9-992b-d355d1cfee6e.PNG)
***

## Models
  - `GhostNetV2_100` should be same with `GhostNetV2 (1.0x)`. Weights are ported from official publication. Currently it's only weights with accuracy `74.41` provided.

  | Model             | Params | FLOPs  | Input | Top1 Acc | Download |
  | ----------------- | ------ | ------ | ----- | -------- | -------- |
  | GhostNetV2_100    | 6.12M  | 168.5M | 224   | 74.41    | [ghostnetv2_100_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/ghostnetv2/ghostnetv2_100_imagenet.h5) |
  | GhostNetV2 (1.0x) | 6.12M  | 168.5M | 224   | 75.3     |          |
  | GhostNetV2 (1.3x) | 8.96M  | 271.1M | 224   | 76.9     |          |
  | GhostNetV2 (1.6x) | 12.39M | 400.9M | 224   | 77.8     |          |

  | Model        | Params | FLOPs  | Input | Top1 Acc | Download |
  | ------------ | ------ | ------ | ----- | -------- | -------- |
  | GhostNet_050 | 2.59M  | 42.6M  | 224   | 66.88    | [ghostnet_050_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/ghostnetv2/ghostnet_050_imagenet.h5) |
  | GhostNet_100 | 5.18M  | 141.7M | 224   | 74.16    | [ghostnet_100_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/ghostnetv2/ghostnet_100_imagenet.h5) |
  | GhostNet_130 | 7.36M  | 227.7M | 224   | 75.79    | [ghostnet_130_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/ghostnetv2/ghostnet_130_imagenet.h5) |
  | - ssld       | 7.36M  | 227.7M | 224   | 79.38    | [ghostnet_130_ssld.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/ghostnetv2/ghostnet_130_ssld.h5) |
## Usage
  ```py
  from keras_cv_attention_models import ghostnet

  # Will download and load pretrained imagenet weights.
  mm = ghostnet.GhostNetV2_100(pretrained="imagenet")

  # Run prediction
  import tensorflow as tf
  from tensorflow import keras
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.81426907), ('n02123045', 'tabby', 0.07202001), ...]
  ```
  **Use dynamic input resolution** by set `input_shape=(None, None, 3)`.
  ```py
  from keras_cv_attention_models import ghostnet
  model = ghostnet.GhostNetV2_100(input_shape=(None, None, 3), num_classes=0)

  print(model(np.ones([1, 224, 224, 3])).shape)
  # (1, 7, 7, 960)
  print(model(np.ones([1, 512, 384, 3])).shape)
  # (1, 16, 12, 960)
  ```
