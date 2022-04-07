# ___Keras SwinTransformerV2___
***

## Summary
  - Keras implementation of [Github timm/swin_transformer_v2_cr](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/swin_transformer_v2_cr.py). Paper [PDF 2111.09883 Swin Transformer V2: Scaling Up Capacity and Resolution](https://arxiv.org/pdf/2111.09883.pdf).
  - Model weights reloaded from [Github timm/swin_transformer_v2_cr](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/swin_transformer_v2_cr.py).
***

## Models
  | Model                           | Params | Image resolution | Top1 Acc | Download |
  | ------------------------------- | ------ | ---------------- | -------- | -------- |
  | SwinTransformerV2Tiny_ns        | 28.3M  | 224              | 81.8     | [v2_tiny_ns_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_tiny_ns_224_imagenet.h5) |
  | SwinTransformerV2Small          | 49.7M  | 224              | 83.13    | [v2_small_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_small_224_imagenet.h5) |
  | SwinTransformerV2Base, 22k      | 87.9M  | 384              | 87.1     |          |
  | SwinTransformerV2Large, 22k     | 196.7M | 384              | 87.7     |          |
  | SwinTransformerV2Giant, 22k+ext | 2.60B  | 640              | 90.17    |          |
## Usage
  ```py
  from keras_cv_attention_models import swin_transformer_v2

  # Will download and load pretrained imagenet weights.
  mm = swin_transformer_v2.SwinTransformerV2Tiny_ns(pretrained="imagenet")

  # Run prediction
  import tensorflow as tf
  from tensorflow import keras
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.72440475), ('n02123159', 'tiger_cat', 0.0824333), ...]
  ```
  **Change input resolution** `input_shape` should be divisible by `window_ratio`, default is `32`.
  ```py
  from keras_cv_attention_models import swin_transformer_v2
  mm = swin_transformer_v2.SwinTransformerV2Tiny_ns(input_shape=(512, 256, 3), pretrained="imagenet")
  # >>>> Load pretrained from: ~/.keras/models/swin_transformer_v2_tiny_ns_224_imagenet.h5

  # Run prediction
  from skimage.data import chelsea
  preds = mm(mm.preprocess_input(chelsea()))
  print(mm.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.4695489), ('n02123159', 'tiger_cat', 0.15133126), ...]
  ```
  Reloading weights with new input_shape not divisible by default `window_ratio` works in some cases, like `input_shape` and `window_ratio` both downsample half:
  ```py
  from keras_cv_attention_models import swin_transformer_v2
  mm = swin_transformer_v2.SwinTransformerV2Tiny_ns(input_shape=(112, 112, 3), window_ratio=16, pretrained="imagenet")
  # >>>> Load pretrained from: ~/.keras/models/swin_transformer_v2_tiny_ns_224_imagenet.h5

  # Run prediction
  from skimage.data import chelsea
  preds = mm(mm.preprocess_input(chelsea()))
  print(mm.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.8370753), ('n02123045', 'tabby', 0.04485862), ...]
  ```
## Verification with PyTorch version
  ```py
  inputs = np.random.uniform(size=(1, 224, 224, 3)).astype("float32")

  """ PyTorch swin_v2_cr_tiny_ns_224 """
  sys.path.append("../pytorch-image-models")
  import timm
  import torch
  torch_model = timm.models.swin_v2_cr_tiny_ns_224(pretrained=True)
  _ = torch_model.eval()
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()

  """ Keras SwinTransformerV2Tiny_ns """
  from keras_cv_attention_models import swin_transformer_v2
  mm = swin_transformer_v2.SwinTransformerV2Tiny_ns(pretrained="imagenet", classifier_activation=None)
  keras_out = mm(inputs).numpy()

  """ Verification """
  print(f"{np.allclose(torch_out, keras_out, atol=1e-5) = }")
  # np.allclose(torch_out, keras_out, atol=1e-5) = True
  ```
***
