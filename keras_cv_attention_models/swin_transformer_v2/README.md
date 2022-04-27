# ___Keras SwinTransformerV2___
***

## Summary
  - Keras implementation of [Github timm/swin_transformer_v2_cr](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/swin_transformer_v2_cr.py). Paper [PDF 2111.09883 Swin Transformer V2: Scaling Up Capacity and Resolution](https://arxiv.org/pdf/2111.09883.pdf).
  - Model weights reloaded from [Github timm/swin_transformer_v2_cr](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/swin_transformer_v2_cr.py).
***

## Models
  | Model                           | Params | FLOPs   | Input | Top1 Acc | Download |
  | ------------------------------- | ------ | ------- | ----- | -------- | -------- |
  | SwinTransformerV2Tiny_ns        | 28.3M  | 4.69G   | 224   | 81.8     | [tiny_ns_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_tiny_ns_224_imagenet.h5) |
  | SwinTransformerV2Small          | 49.7M  | 9.12G   | 224   | 83.13    | [small_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_small_224_imagenet.h5) |
  | SwinTransformerV2Small_ns       | 49.7M  | 9.12G   | 224   | 83.5     | [small_ns_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/swin_transformer_v2/swin_transformer_v2_small_ns_224_imagenet.h5) |
  | SwinTransformerV2Base, 22k      | 87.9M  | 50.89G  | 384   | 87.1     |          |
  | SwinTransformerV2Large, 22k     | 196.7M | 109.40G | 384   | 87.7     |          |
  | SwinTransformerV2Giant, 22k+ext | 2.60B  | 4.26T   | 640   | 90.17    |          |
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
  **Change input resolution**. Note if `input_shape` is not divisible by `window_ratio`, which default is `32`, will pad for `shifted_window_attention`.
  ```py
  from keras_cv_attention_models import swin_transformer_v2
  mm = swin_transformer_v2.SwinTransformerV2Tiny_ns(input_shape=(510, 255, 3), pretrained="imagenet")
  # >>>> Load pretrained from: ~/.keras/models/swin_transformer_v2_tiny_ns_224_imagenet.h5

  # Run prediction
  from skimage.data import chelsea
  preds = mm(mm.preprocess_input(chelsea()))
  print(mm.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.5416627), ('n02123159', 'tiger_cat', 0.17523797), ...]
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
