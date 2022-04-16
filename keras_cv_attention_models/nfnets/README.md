# ___Keras NFNets___
***

## Summary
  - Keras implementation of [Github deepmind/nfnets](https://github.com/deepmind/deepmind-research/tree/master/nfnets). Paper [PDF 2102.06171 High-Performance Large-Scale Image Recognition Without Normalization](https://arxiv.org/pdf/2102.06171.pdf).
  - Model weights reloaded from official publication.
  - `ECA` and `Light` NFNets weights reloaded from timm [Github rwightman/pytorch-image-models](https://github.com/rwightman/pytorch-image-models).
***

## Models
  - `L` types models are light versions of `NFNet-F` from `timm`.
  - `ECA` type models are using `attn_type="eca"` instead of `attn_type="se"` from `timm`.

  | Model       | Params | FLOPs   | Input | Top1 Acc | Download |
  | ----------- | ------ | ------- | ----- | -------- | -------- |
  | NFNetL0     | 35.07M | 7.13G   | 288   | 82.75    | [nfnetl0_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nfnets/nfnetl0_imagenet.h5) |
  | NFNetF0     | 71.5M  | 12.58G  | 256   | 83.6     | [nfnetf0_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nfnets/nfnetf0_imagenet.h5) |
  | NFNetF1     | 132.6M | 35.95G  | 320   | 84.7     | [nfnetf1_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nfnets/nfnetf1_imagenet.h5) |
  | NFNetF2     | 193.8M | 63.24G  | 352   | 85.1     | [nfnetf2_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nfnets/nfnetf2_imagenet.h5) |
  | NFNetF3     | 254.9M | 115.75G | 416   | 85.7     | [nfnetf3_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nfnets/nfnetf3_imagenet.h5) |
  | NFNetF4     | 316.1M | 216.78G | 512   | 85.9     | [nfnetf4_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nfnets/nfnetf4_imagenet.h5) |
  | NFNetF5     | 377.2M | 291.73G | 544   | 86.0     | [nfnetf5_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nfnets/nfnetf5_imagenet.h5) |
  | NFNetF6 SAM | 438.4M | 379.75G | 576   | 86.5     | [nfnetf6_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nfnets/nfnetf6_imagenet.h5) |
  | NFNetF7     | 499.5M | 481.80G | 608   |          |          |
  | ECA_NFNetL0 | 24.14M | 7.12G   | 288   | 82.58    | [eca_nfnetl0_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nfnets/eca_nfnetl0_imagenet.h5) |
  | ECA_NFNetL1 | 41.41M | 14.93G  | 320   | 84.01    | [eca_nfnetl1_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nfnets/eca_nfnetl1_imagenet.h5) |
  | ECA_NFNetL2 | 56.72M | 30.12G  | 384   | 84.70    | [eca_nfnetl2_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nfnets/eca_nfnetl2_imagenet.h5) |
  | ECA_NFNetL3 | 72.04M | 52.73G  | 448   |          |          |
## Usage
  ```py
  from keras_cv_attention_models import nfnets

  # Will download and load pretrained imagenet weights.
  mm = nfnets.NFNetF0(pretrained="imagenet")

  # Run prediction
  import tensorflow as tf
  from tensorflow import keras
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.9195376), ('n02123159', 'tiger_cat', 0.021603014), ...]
  ```
  **Use dynamic input resolution**
  ```py
  from keras_cv_attention_models import nfnets
  mm = nfnets.NFNetF1(input_shape=(None, None, 3), num_classes=0, pretrained="imagenet")

  print(mm(np.ones([1, 320, 320, 3])).shape)
  # (1, 10, 10, 3072)
  print(mm(np.ones([1, 512, 512, 3])).shape)
  # (1, 16, 16, 3072)

  mm.save("nfnetf1_imagenet_dynamic_notop.h5")
  ```
***
