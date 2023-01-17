# ___Keras TinyViT___
***

## Summary
  - Keras implementation of [Github microsoft/TinyViT](https://github.com/microsoft/Cream/tree/main/TinyViT). Paper [PDF 2207.10666 TinyViT: Fast Pretraining Distillation for Small Vision Transformers](https://arxiv.org/pdf/2207.10666.pdf).
  - Model weights ported from official publication.
***

## Models
  | Model                | Params | FLOPs | Input | Top1 Acc | Download |
  | -------------------- | ------ | ----- | ----- | -------- | -------- |
  | TinyViT_5M, distill  | 5.4M   | 1.3G  | 224   | 79.1     |          |
  | - imagenet21k-ft1k   | 5.4M   | 1.3G  | 224   | 80.7     | [tiny_vit_5m_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/tinyvit/tiny_vit_5m_224_imagenet21k-ft1k.h5) |
  | TinyViT_11M, distill | 11M    | 2.0G  | 224   | 81.5     |          |
  | - imagenet21k-ft1k   | 11M    | 2.0G  | 224   | 83.2     | [tiny_vit_11m_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/tinyvit/tiny_vit_11m_224_imagenet21k-ft1k.h5) |
  | TinyViT_21M, distill | 21M    | 4.3G  | 224   | 84.8     |          |
  | - imagenet21k-ft1k   | 21M    | 4.3G  | 224   | 83.1     | [tiny_vit_21m_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/tinyvit/tiny_vit_21m_224_imagenet21k-ft1k.h5) |
  |                      | 21M    | 13.8G | 384   | 86.2     |          |
  |                      | 21M    | 27.0G | 512   | 86.5     |          |
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
  **Change input resolution** if input_shape is not within pre-trained, will load `MultiHeadPositionalEmbedding` weights by `load_resized_weights`.
  ```py
  from keras_cv_attention_models import tinyvit
  mm = tinyvit.TinyViT_11M(input_shape=(112, 253, 3))
  # >>>> Load pretrained from: /home/leondgarse/.keras/models/tiny_vit_11m_224_imagenet21k-ft1k.h5
  # WARNING:tensorflow:Skipping loading weights for layer #121 (named stack3_block1_attn_attn_pos) due to mismatch in shape ...
  # >>>> Reload mismatched weights: 224 -> (112, 253)
  # >>>> Reload layer: stack2_block1_attn_attn_pos
  # ...

  # Run prediction on Chelsea with (112, 253) resolution
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.44511992), ('n02123045', 'tabby', 0.2975872), ...]
  ```
***
