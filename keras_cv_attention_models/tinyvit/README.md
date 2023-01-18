# ___Keras TinyViT___
***

## Summary
  - Keras implementation of [Github microsoft/TinyViT](https://github.com/microsoft/Cream/tree/main/TinyViT). Paper [PDF 2207.10666 TinyViT: Fast Pretraining Distillation for Small Vision Transformers](https://arxiv.org/pdf/2207.10666.pdf).
  - Model weights ported from official publication.
***

## Models
  | Model                | Params | FLOPs  | Input | Top1 Acc | Download |
  | -------------------- | ------ | ------ | ----- | -------- | -------- |
  | TinyViT_5M, distill  | 5.39M  | 1.27G  | 224   | 79.1     | [tiny_vit_5m_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/tinyvit/tiny_vit_5m_224_imagenet.h5)      |
  | - imagenet21k-ft1k   | 5.39M  | 1.27G  | 224   | 80.7     | [tiny_vit_5m_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/tinyvit/tiny_vit_5m_224_imagenet21k-ft1k.h5)   |
  | TinyViT_11M, distill | 11.00M | 2.04G  | 224   | 81.5     | [tiny_vit_11m_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/tinyvit/tiny_vit_11m_224_imagenet.h5)    |
  | - imagenet21k-ft1k   | 11.00M | 2.04G  | 224   | 83.2     | [tiny_vit_11m_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/tinyvit/tiny_vit_11m_224_imagenet21k-ft1k.h5) |
  | TinyViT_21M, distill | 21.2M  | 4.29G  | 224   | 83.1     | [tiny_vit_21m_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/tinyvit/tiny_vit_21m_224_imagenet.h5)    |
  | - imagenet21k-ft1k   | 21.2M  | 4.29G  | 224   | 84.8     | [tiny_vit_21m_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/tinyvit/tiny_vit_21m_224_imagenet21k-ft1k.h5) |
  |                      | 21.2M  | 13.86G | 384   | 86.2     | [tiny_vit_21m_384_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/tinyvit/tiny_vit_21m_384_imagenet21k-ft1k.h5) |
  |                      | 21.3M  | 27.29G | 512   | 86.5     | [tiny_vit_21m_512_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/tinyvit/tiny_vit_21m_512_imagenet21k-ft1k.h5) |
## Usage
  ```py
  from keras_cv_attention_models import tinyvit

  # Will download and load pretrained imagenet21k-ft1k weights.
  mm = tinyvit.TinyViT_5M(pretrained="imagenet21k-ft1k")

  # Run prediction
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.8177282), ('n02123045', 'tabby', 0.100739196), ...]
  ```
  **Change input resolution** if input_shape is not within pre-trained, will load `MultiHeadPositionalEmbedding` weights by `load_resized_weights`. **Should better be divisible by `32`, or will apply padding.**
  ```py
  from keras_cv_attention_models import tinyvit
  mm = tinyvit.TinyViT_11M(input_shape=(160, 128, 3))
  # >>>> Load pretrained from: /home/leondgarse/.keras/models/tiny_vit_11m_224_imagenet21k-ft1k.h5
  # WARNING:tensorflow:Skipping loading weights for layer #121 (named stack3_block1_attn_attn_pos) due to mismatch in shape ...
  # >>>> Reload mismatched weights: 224 -> (160, 128)
  # >>>> Reload layer: stack2_block1_attn_attn_pos
  # ...

  # Run prediction
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.53003114), ('n02123159', 'tiger_cat', 0.13526095), ...]
  ```
***
