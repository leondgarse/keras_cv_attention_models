# ___Keras DaViT___
***

## Summary
  - DaViT article: [PDF 2204.03645 DaViT: Dual Attention Vision Transformers](https://arxiv.org/pdf/2204.03645.pdf).
  - Model weights reloaded from [Github dingmyu/davit](https://github.com/dingmyu/davit).
***

## Models
  | Model         | Params | FLOPs  | Input | Top1 Acc | Download |
  | ------------- | ------ | ------ | ----- | -------- | -------- |
  | DaViT_T       | 28.36M | 4.56G  | 224   | 82.8     | [davit_t_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/davit/davit_t_imagenet.h5) |
  | DaViT_S       | 49.75M | 8.83G  | 224   | 84.2     | [davit_s_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/davit/davit_s_imagenet.h5) |
  | DaViT_B       | 87.95M | 15.55G | 224   | 84.6     | [davit_b_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/davit/davit_b_imagenet.h5) |
  | DaViT_L, 21k  | 196.8M | 103.2G | 384   | 87.5     |          |
  | DaViT_H, 1.5B | 348.9M | 327.3G | 512   | 90.2     |          |
  | DaViT_G, 1.5B | 1.406B | 1.022T | 512   | 90.4     |          |

  **Self tested accuracy**. There may be some detail differences in model output layer or evaluating process.
  ```sh
  CUDA_VISIBLE_DEVICES='0' ./eval_script.py -m davit.DaViT_T
  # >>>> Accuracy top1: 0.82276 top5: 0.96152
  ```
  | Model   | Self tested Top1 Acc |
  | ------- | -------------------- |
  | DaViT_T | 82.276               |
  | DaViT_S | 83.810               |
  | DaViT_B | 84.142               |
## Usage
  ```py
  from keras_cv_attention_models import davit

  # Will download and load pretrained imagenet weights.
  mm = davit.DaViT_T(pretrained="imagenet")

  # Run prediction
  import tensorflow as tf
  from tensorflow import keras
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.39985177), ('n02123159', 'tiger_cat', 0.036589254), ...]
  ```
  **Change input resolution**. Note if `input_shape` is not divisible by `window_ratio`, which default is `32`, will pad for `window_attention`.
  ```py
  from keras_cv_attention_models import davit
  mm = davit.DaViT_T(input_shape=(376, 227, 3), pretrained="imagenet")
  # >>>> Load pretrained from: ~/.keras/models/davit_t_imagenet.h5

  # Run prediction
  from skimage.data import chelsea
  preds = mm(mm.preprocess_input(chelsea()))
  print(mm.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.17319576), ('n02123159', 'tiger_cat', 0.017631555), ...]
  ```
  Reloading weights with new input_shape not divisible by default `window_ratio` works in some cases, like `input_shape` and `window_ratio` both downsample half:
  ```py
  from keras_cv_attention_models import davit
  mm = davit.DaViT_T(input_shape=(112, 112, 3), window_ratio=16, pretrained="imagenet")
  # >>>> Load pretrained from: ~/.keras/models/davit_t_imagenet.h5

  # Run prediction
  from skimage.data import chelsea
  preds = mm(mm.preprocess_input(chelsea()))
  print(mm.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.7279274), ('n02123045', 'tabby', 0.021591123), ...]
  ```
***
