# ___Keras ResNet Family___
***

## Usage
  ```py
  from keras_cv_attention_models import resnet_family

  # Will download and load pretrained imagenet weights.
  mm = resnet_family.ResNeXt50(pretrained="swsl")

  # Run prediction
  import tensorflow as tf
  from tensorflow import keras
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.99826247), ('n02123045', 'tabby', 0.0009374655), ... ]
  ```
  **Set new input resolution**
  ```py
  from keras_cv_attention_models import resnet_family
  mm = resnet_family.ResNet51Q(input_shape=(320, 320, 3), num_classes=0)
  print(mm(np.ones([1, 320, 320, 3])).shape)
  # (1, 10, 10, 2048)

  mm = resnet_family.ResNet51Q(input_shape=(512, 512, 3), num_classes=0)
  print(mm(np.ones([1, 512, 512, 3])).shape)
  # (1, 16, 16, 2048)
  ```
  **Set `input_shape=(None, None, 3)` for dynamic input shape**
  ```py
  from keras_cv_attention_models import resnet_family
  mm = resnet_family.RegNetZB(input_shape=(None, None, 3), num_classes=0)
  print(mm(np.ones([1, 320, 320, 3])).shape)
  # (1, 10, 10, 1536)
  print(mm(np.ones([1, 512, 512, 3])).shape)
  # (1, 10, 10, 1536)
  ```
## Bag of Tricks for ResNet
  - [PDF 1812.01187 Bag of Tricks for Image Classification with Convolutional Neural Networks](https://arxiv.org/pdf/1812.01187.pdf)
  - [PDF 2004.08955 ResNeSt: Split-Attention Networks](https://arxiv.org/pdf/2004.08955.pdf)
    | Model                                                 | Params | Top1 Acc |
    | ----------------------------------------------------- | ------ | -------- |
    | Basic ResNet50_V1                                     | 25M    | 76.21    |
    | ResNet50_V1-B (block strides 1, 2 -> 2, 1)            | 25M    | 76.66    |
    | ResNet50_V1-C (deep stem)                             | 25M    | 76.87    |
    | ResNet50_V1-D (block shortcut avgpool + conv)         | 25M    | 77.16    |
    | ResNet50_V1-D + cosine lr decay                       | 25M    | 77.91    |
    | ResNet50_V1-D + cosine + label smoothing 0.1          | 25M    | 78.31    |
    | ResNet50_V1-D + cosine + LS 0.1 + mixup 0.2           | 25M    | 79.15    |
    | ResNet50_V1-D + cosine + LS 0.1 + mixup 0.2 + autoaug | 25M    | 79.41    |
    | ResNeSt-50                                            | 27M    | 81.13    |

  - [PDF 2103.07579 Revisiting ResNets: Improved Training and Scaling Strategies](https://arxiv.org/pdf/2103.07579.pdf)
    | Model                      | Top1 Acc | ∆    |
    | -------------------------- | -------- | ---- |
    | ResNet-200                 | 79.0     | —    |
    | + Cosine LR Decay          | 79.3     | +0.3 |
    | + Increase training epochs | 78.8 †   | -0.5 |
    | + EMA of weights           | 79.1     | +0.3 |
    | + Label Smoothing          | 80.4     | +1.3 |
    | + Stochastic Depth         | 80.6     | +0.2 |
    | + RandAugment              | 81.0     | +0.4 |
    | + Dropout on FC            | 80.7 ‡   | -0.3 |
    | + Decrease weight decay    | 82.2     | +1.5 |
    | + Squeeze-and-Excitation   | 82.9     | +0.7 |
    | + ResNet-D                 | 83.4     | +0.5 |

    | Model      | Regularization | Weight Decay 1e-4 | Weight Decay 4e-5 | ∆    |
    | ---------- | -------------- | ----------------- | ----------------- | ---- |
    | ResNet-50  | None           | 79.7              | 78.7              | -1.0 |
    | ResNet-50  | RA-LS          | 82.4              | 82.3              | -0.1 |
    | ResNet-50  | RA-LS-DO       | 82.2              | 82.7              | +0.5 |
    | ResNet-200 | None           | 82.5              | 81.7              | -0.8 |
    | ResNet-200 | RA-LS          | 85.2              | 84.9              | -0.3 |
    | ResNet-200 | RA-LS-SD-DO    | 85.3              | 85.5              | +0.2 |
## ResNeXt
  - [PDF 1611.05431 Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf)
  - Keras implementation of [Github facebookresearch/ResNeXt](https://github.com/facebookresearch/ResNeXt).
  - Model weights reloaded from [Tensorflow keras/applications](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/applications/resnet.py) and [Github rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models).
  - `SWSL` means `Semi-Weakly Supervised ResNe*t` from [Github facebookresearch/semi-supervised-ImageNet1K-models](https://github.com/facebookresearch/semi-supervised-ImageNet1K-models). **Please note the CC-BY-NC 4.0 license on theses weights, non-commercial use only**.

  | Model                     | Params | Image  resolution | Top1 Acc | Download            |
  | ------------------------- | ------ | ----------------- | -------- | ------------------- |
  | ResNeXt50 (32x4d)         | 25M    | 224               | 79.768   | [resnext50_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/resnext50_imagenet.h5)  |
  | - SWSL                    | 25M    | 224               | 82.182   | [resnext50_swsl.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/resnext50_swsl.h5)  |
  | ResNeXt50D (32x4d + deep) | 25M    | 224               | 79.676   | [resnext50d_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/resnext50d_imagenet.h5)  |
  | ResNeXt101 (32x4d)        | 42M    | 224               | 80.334   | [resnext101_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/resnext101_imagenet.h5)  |
  | - SWSL                    | 42M    | 224               | 83.230   | [resnext101_swsl.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/resnext101_swsl.h5)  |
  | ResNeXt101W (32x8d)       | 89M    | 224               | 79.308   | [resnext101_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/resnext101_imagenet.h5)  |
  | - SWSL                    | 89M    | 224               | 84.284   | [resnext101w_swsl.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/resnext101w_swsl.h5)  |
## ResNetD
  | Model      | Params | Image  resolution | Top1 Acc | Download |
  | ---------- | ------ | ----------------- | -------- | -------- |
  | ResNet50D  | 25.58M | 224               | 80.530   | [resnet50d.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/resnet50d_imagenet.h5) |
  | ResNet101D | 44.57M | 224               | 83.022   | [resnet101d.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/resnet101d_imagenet.h5) |
  | ResNet152D | 60.21M | 224               | 83.680   | [resnet152d.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/resnet152d_imagenet.h5) |
  | ResNet200D | 64.69  | 224               | 83.962   | [resnet200d.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/resnet200d_imagenet.h5) |
## ResNetQ
  - Defined and model weights loaded from [Github timm/models/resnet.py](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/resnet.py).

  | Model     | Params | Image  resolution | Top1 Acc | Download |
  | --------- | ------ | ----------------- | -------- | -------- |
  | ResNet51Q | 35.7M  | 224               | 82.36    | [resnet51q.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/resnet51q_imagenet.h5) |
## RegNetZ
  - Defined and model weights loaded from [Github timm/models/byobnet.py](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/byobnet.py)

  | Model     | Params | Image  resolution | Top1 Acc | Download |
  | --------- | ------ | ----------------- | -------- | -------- |
  | RegNetZB  | 9.72M  | 224               | 79.868   | [regnetz_b_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/regnetz_b_imagenet.h5) |
  | RegNetZC  | 13.46M | 256               | 82.164   | [regnetz_c_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/regnetz_c_imagenet.h5) |
  | RegNetZD  | 27.58M | 256               | 83.422   | [regnetz_d_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnet_family/regnetz_d_imagenet.h5) |
***
