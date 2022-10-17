# ___Keras MaxViT___
***

## Summary
  - Keras implementation of [Github google-research/maxvit](https://github.com/google-research/maxvit). Paper [PDF 2204.01697 MaxViT: Multi-Axis Vision Transformer](https://arxiv.org/pdf/2204.01697.pdf).
  - Model weights ported from official publication.
***

## Models
  | Model         | Params | FLOPs  | Input | Top1 Acc | Download |
  | ------------- | ------ | ------ | ----- | -------- | -------- |
  | MaxViT_T      | 31M    | 5.6G   | 224   | 83.62    | [maxvit_t_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_t_224_imagenet.h5) |

## Usage
  ```py
  from keras_cv_attention_models import maxvit

  # Will download and load pretrained imagenet weights.
  mm = maxvit.MaxViT_T(pretrained="imagenet")

  # Run prediction
  import tensorflow as tf
  from tensorflow import keras
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.5994264), ('n02123159', 'tiger_cat', 0.085186124), ...]
  ```
  **Change input resolution**.
  ```py
  from keras_cv_attention_models import maxvit
  mm = maxvit.MaxViT_T(input_shape=(117, 393, 3), pretrained="imagenet")
  # >>>> Load pretrained from: ~/.keras/models/hornet_tiny_gf_224_imagenet.h5
  # ...
  # >>>> Reload mismatched weights: 224 -> (117, 393)
  # >>>> Reload layer: stack3_block1_gnconv_gf_complex_dense
  # ...
  # >>>> Reload layer: stack4_block2_gnconv_gf_complex_dense

  # Run prediction
  from skimage.data import chelsea
  preds = mm(mm.preprocess_input(chelsea()))
  print(mm.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.74485517), ('n02123045', 'tabby', 0.025426276), ...]
  ```
