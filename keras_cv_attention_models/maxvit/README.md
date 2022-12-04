# ___Keras MaxViT___
***

## Summary
  - Keras implementation of [Github google-research/maxvit](https://github.com/google-research/maxvit). Paper [PDF 2204.01697 MaxViT: Multi-Axis Vision Transformer](https://arxiv.org/pdf/2204.01697.pdf).
  - Model weights ported from official publication.
***

## Models
  | Model                           | Params | FLOPs  | Input | Top1 Acc | Download |
  | ------------------------------- | ------ | ------ | ----- | -------- | -------- |
  | MaxViT_Tiny                     | 31M    | 5.6G   | 224   | 83.62    | [tiny_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_tiny_224_imagenet.h5) |
  | MaxViT_Tiny                     | 31M    | 17.7G  | 384   | 85.24    | [tiny_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_tiny_384_imagenet.h5) |
  | MaxViT_Tiny                     | 31M    | 33.7G  | 512   | 85.72    | [tiny_512_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_tiny_512_imagenet.h5) |
  | MaxViT_Small                    | 69M    | 11.7G  | 224   | 84.45    | [small_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_small_224_imagenet.h5) |
  | MaxViT_Small                    | 69M    | 36.1G  | 384   | 85.74    | [small_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_small_384_imagenet.h5) |
  | MaxViT_Small                    | 69M    | 67.6G  | 512   | 86.19    | [small_512_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_small_512_imagenet.h5) |
  | MaxViT_Base                     | 119M   | 24.2G  | 224   | 84.95    | [base_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_base_224_imagenet.h5) |
  | - imagenet21k                   | 135M   | 24.2G  | 224   |          | [base_224_imagenet21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_base_224_imagenet21k.h5) |
  | MaxViT_Base                     | 119M   | 74.2G  | 384   | 86.34    | [base_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_base_384_imagenet.h5) |
  | - imagenet21k-ft1k              | 119M   | 74.2G  | 384   | 88.24    | [base_384_21k-ft1k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_base_384_imagenet21k-ft1k.h5) |
  | MaxViT_Base                     | 119M   | 138.5G | 512   | 86.66    | [base_512_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_base_512_imagenet.h5) |
  | - imagenet21k-ft1k              | 119M   | 138.5G | 512   | 88.38    | [base_512_21k-ft1k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_base_512_imagenet21k-ft1k.h5) |
  | MaxViT_Large                    | 212M   | 43.9G  | 224   | 85.17    | [large_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_large_224_imagenet.h5) |
  | - imagenet21k                   | 233M   | 43.9G  | 224   |          | [large_224_imagenet21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_large_224_imagenet21k.h5) |
  | MaxViT_Large                    | 212M   | 133.1G | 384   | 86.40    | [large_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_large_384_imagenet.h5) |
  | - imagenet21k-ft1k              | 212M   | 133.1G | 384   | 88.32    | [large_384_21k-ft1k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_large_384_imagenet21k-ft1k.h5) |
  | MaxViT_Large                    | 212M   | 245.4G | 512   | 86.70    | [large_512_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_large_512_imagenet.h5) |
  | - imagenet21k-ft1k              | 212M   | 245.4G | 512   | 88.46    | [large_512_21k-ft1k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_large_512_imagenet21k-ft1k.h5) |
  | MaxViT_XLarge, imagenet21k      | 507M   | 97.7G  | 224   |          | [xlarge_224_imagenet21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_xlarge_224_imagenet21k.h5) |
  | MaxViT_XLarge, imagenet21k-ft1k | 475M   | 293.7G | 384   | 88.51    | [xlarge_384_21k-ft1k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_xlarge_384_imagenet21k-ft1k.h5) |
  | MaxViT_XLarge, imagenet21k-ft1k | 475M   | 535.2G | 512   | 88.70    | [xlarge_512_21k-ft1k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/maxvit/maxvit_xlarge_512_imagenet21k-ft1k.h5) |
## Usage
  ```py
  from keras_cv_attention_models import maxvit

  # Will download and load pretrained imagenet weights.
  mm = maxvit.MaxViT_Tiny(pretrained="imagenet")

  # Run prediction
  import tensorflow as tf
  from tensorflow import keras
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.700786), ('n02123159', 'tiger_cat', 0.04504126), ...]
  ```
  **Change input resolution**.
  ```py
  from keras_cv_attention_models import maxvit
  mm = maxvit.MaxViT_Tiny(input_shape=(117, 393, 3), pretrained="imagenet")
  # >>>> Load pretrained from: ~/.keras/models/maxvit_tiny_224_imagenet.h5
  # ...
  # >>>> Reload mismatched weights: 224 -> (117, 393)
  # >>>> Reload layer: stack_1_block_1/block_window_mhsa/pos_emb
  # ...
  # >>>> Reload layer: stack_4_block_2/grid_window_mhsa/pos_emb

  # Run prediction
  from skimage.data import chelsea
  preds = mm(mm.preprocess_input(chelsea()))
  print(mm.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.77995235), ('n02123045', 'tabby', 0.017138876), ...]
  ```
  **Actual `num_classes` for `"imagenet21k"` models is `21843`**.
  ```py
  from keras_cv_attention_models import maxvit

  # Will download and load pretrained imagenet weights.
  mm = maxvit.MaxViT_Large(pretrained="imagenet21k", num_classes=21843)

  # Run prediction
  from skimage.data import chelsea
  preds = mm(mm.preprocess_input(chelsea()))
  print(mm.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.059920408), ('n02120997', 'feline, felid', 0.04957321), ...]
  ```
