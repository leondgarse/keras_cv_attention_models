# ___Keras MogaNet___
***

## Summary
  - Keras implementation of [Github Westlake-AI/MogaNet](https://github.com/Westlake-AI/MogaNet). Paper [PDF 2211.03295 Efficient Multi-order Gated Aggregation Network](https://arxiv.org/pdf/2211.03295.pdf).
  - Model weights ported from official publication.
***

## Models
  | Model        | Params | FLOPs  | Input | Top1 Acc | Download |
  | ------------ | ------ | ------ | ----- | -------- | -------- |
  | MogaNetXtiny | 2.96M  | 806M   | 224   | 76.5     | [moganet_xtiny_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/moganet/moganet_xtiny_imagenet.h5) |
  | MogaNetTiny  | 5.20M  | 1.11G  | 224   | 79.0     | [moganet_tiny_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/moganet/moganet_tiny_224_imagenet.h5) |
  |              | 5.20M  | 1.45G  | 256   | 79.6     | [moganet_tiny_256_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/moganet/moganet_tiny_256_imagenet.h5) |
  | MogaNetSmall | 25.3M  | 4.98G  | 224   | 83.4     | [moganet_small_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/moganet/moganet_small_imagenet.h5) |
  | MogaNetBase  | 43.7M  | 9.96G  | 224   | 84.2     | [moganet_base_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/moganet/moganet_base_imagenet.h5) |
  | MogaNetLarge | 82.5M  | 15.96G | 224   | 84.6     | [moganet_large_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/moganet/moganet_large_imagenet.h5) |
## Usage
  ```py
  from keras_cv_attention_models import moganet

  # Will download and load pretrained imagenet weights.
  mm = moganet.MogaNetXtiny(pretrained="imagenet")

  # Run prediction
  import tensorflow as tf
  from tensorflow import keras
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.6138564), ('n02123045', 'tabby', 0.16214457), ...]
  ```
  **Change input resolution**.
  ```py
  from keras_cv_attention_models import moganet
  mm = moganet.MogaNetXtiny(input_shape=(112, 193, 3), pretrained="imagenet")
  # >>>> Load pretrained from: ~/.keras/models/caformer_small18_224_imagenet.h5

  # Run prediction
  from skimage.data import chelsea
  preds = mm(mm.preprocess_input(chelsea()))
  print(mm.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.5223805), ('n02123045', 'tabby', 0.27944055), ...]
  ```
  **Use dynamic input resolution** by set `input_shape=(None, None, 3)`.
  ```py
  from keras_cv_attention_models import moganet
  model = moganet.MogaNetTiny(input_shape=(None, None, 3), num_classes=0)

  print(model(np.ones([1, 223, 123, 3])).shape)
  # (1, 7, 4, 256)
  print(model(np.ones([1, 32, 526, 3])).shape)
  # (1, 1, 17, 256)
  ```
***
