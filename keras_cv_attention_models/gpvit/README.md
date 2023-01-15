# ___Keras GPViT___
***

## Summary
  - Keras implementation of [Github ChenhongyiYang/GPViT](https://github.com/ChenhongyiYang/GPViT). Paper [PDF 2212.06795 GPVIT: A HIGH RESOLUTION NON-HIERARCHICAL VISION TRANSFORMER WITH GROUP PROPAGATION](https://arxiv.org/pdf/2212.06795.pdf).
  - Model weights ported from official publication.
***

## Models
  | Model    | Params | FLOPs  | Input | Top1 Acc | Download |
  | -------- | ------ | ------ | ----- | -------- | -------- |
  | GPViT_L1 | 9.59M  | 6.15G  | 224   | 80.5     | [gpvit_l1_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gpvit/gpvit_l1_224_imagenet.h5) |
  | GPViT_L2 | 24.2M  | 15.74G | 224   | 83.4     | [gpvit_l2_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gpvit/gpvit_l2_224_imagenet.h5) |
  | GPViT_L3 | 36.7M  | 23.54G | 224   | 84.1     | [gpvit_l3_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gpvit/gpvit_l3_224_imagenet.h5) |
  | GPViT_L4 | 75.5M  | 48.29G | 224   | 84.3     | [gpvit_l4_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gpvit/gpvit_l4_224_imagenet.h5) |
## Usage
  ```py
  from keras_cv_attention_models import gpvit

  # Will download and load pretrained imagenet weights.
  mm = gpvit.GPViT_L1(pretrained="imagenet")

  # Run prediction
  import tensorflow as tf
  from tensorflow import keras
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.7434748), ('n02123045', 'tabby', 0.089776225), ...]
  ```
  **Change input resolution**.
  ```py
  from keras_cv_attention_models import gpvit
  mm = gpvit.GPViT_L1(input_shape=(128, 192, 3), pretrained="imagenet")
  # >>>> Load pretrained from: ~/.keras/models/gp_vit_l1_224_imagenet.h5
  # >>>> Reload mismatched weights: 224 -> (128, 192)
  # >>>> Reload layer: positional_embedding

  # Run prediction
  from skimage.data import chelsea
  preds = mm(mm.preprocess_input(chelsea()))
  print(mm.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.8140152), ('n02123045', 'tabby', 0.05595901), ...]
  ```
***
