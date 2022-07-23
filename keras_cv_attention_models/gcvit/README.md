# ___Keras GCViT___
***

## Summary
  - Keras implementation of [Github NVlabs/GCVit](https://github.com/NVlabs/GCVit). Paper [PDF 2206.09959 Global Context Vision Transformers](https://arxiv.org/pdf/2206.09959.pdf).
  - **Note: There are 3 issues in official implementation. Here is NOT using these, thus ported weights cannot keep the accuracy**.
    - [Global query: wrong repeating #10](https://github.com/NVlabs/GCVit/issues/10). Repeating on `global_query` batch dimension seems in wrong order.
    - [Global query: wrong heads extraction #11](https://github.com/NVlabs/GCVit/issues/11). Reshaping `global_query` `[batch, channel, height, width]` -> `[batch, num_heads, height * width, channel // num_heads]`.
    - [Global query: wrong input transformation #13](https://github.com/NVlabs/GCVit/issues/13). Reshaping a channel last input `[batch, height, width, channels]` directly to ` [batch, channels, height, width]`.
***

## Models
  | Model        | Params | FLOPs | Input | Top1 Acc | Download |
  | ------------ | ------ | ----- | ----- | -------- | -------- |
  | GCViT_XXTiny | 12.0M  | 2.15G | 224   | 79.6     |          |
  | GCViT_XTiny  | 20.0M  | 2.96G | 224   | 81.9     |          |
  | GCViT_Tiny   | 28.2M  | 4.83G | 224   | 83.2     | [gcvit_tiny_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gcvit/gcvit_tiny_224_imagenet.h5) |
  | GCViT_Small  | 51.1M  | 8.63G | 224   | 83.9     | [gcvit_small_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gcvit/gcvit_small_224_imagenet.h5) |
  | GCViT_Base   | 90.3M  | 14.9G | 224   | 84.4     | [gcvit_base_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gcvit/gcvit_base_224_imagenet.h5) |

  Self tested accuracy
  ```sh
  CUDA_VISIBLE_DEVICES='0' ./eval_script.py -m gcvit.GCViT_Tiny
  ```
  | Model       | Top1 Acc |
  | ----------- | -------- |
  | GCViT_Tiny  | 81.604   |
  | GCViT_Small | 80.298   |
  | GCViT_Base  | 83.7     |
## Usage
  ```py
  from keras_cv_attention_models import gcvit

  # Will download and load pretrained imagenet weights.
  mm = gcvit.GCViT_Tiny(pretrained="imagenet")

  # Run prediction
  import tensorflow as tf
  from tensorflow import keras
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.8970233), ('n02123045', 'tabby', 0.014118814), ...]
  ```
  **Change input resolution**. `input_shape` has to be divisible by a combination of strides and `window_ratios`, for default `window_ratios=[8, 4, 1, 1]`, it should be divisible by **32**.
  ```py
  from keras_cv_attention_models import gcvit
  mm = gcvit.GCViT_Tiny(input_shape=(128, 192, 3), pretrained="imagenet")
  # >>>> Load pretrained from: ~/.keras/models/gcvit_tiny_224_imagenet.h5

  # Run prediction
  from skimage.data import chelsea
  preds = mm(mm.preprocess_input(chelsea()))
  print(mm.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.9064783), ('n02123045', 'tabby', 0.012468374), ...]
  ```
