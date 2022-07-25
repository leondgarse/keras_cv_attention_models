# ___Keras MobileViT___
***

## Summary
  - Keras implementation of [Github apple/ml-cvnets/mobilevit](https://github.com/apple/ml-cvnets/blob/main/cvnets/models/classification/mobilevit.py). Paper [PDF 2110.02178 MOBILEVIT: LIGHT-WEIGHT, GENERAL-PURPOSE, AND MOBILE-FRIENDLY VISION TRANSFORMER](https://arxiv.org/pdf/2110.02178.pdf).
  - `MobileViT_V2` is for [Github apple/ml-cvnets](https://github.com/apple/ml-cvnets). Paper [PDF 2206.02680 Separable Self-attention for Mobile Vision Transformers](https://arxiv.org/pdf/2206.02680.pdf).
  - `MobileViT` model weights reloaded from [Github timm/mobilevit](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/mobilevit.py).
  - `MobileViT_V2` model weights reloaded from [Github apple/ml-cvnets](https://github.com/apple/ml-cvnets).
***

## Models
  | Model         | Params | FLOPs | Input | Top1 Acc | Download |
  | ------------- | ------ | ----- | ----- | -------- | -------- |
  | MobileViT_XXS | 1.3M   | 0.42G | 256   | 69.0     | [mobilevit_xxs_imagenet](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_xxs_imagenet.h5) |
  | MobileViT_XS  | 2.3M   | 1.05G | 256   | 74.7     | [mobilevit_xs_imagenet](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_xs_imagenet.h5) |
  | MobileViT_S   | 5.6M   | 2.03G | 256   | 78.3     | [mobilevit_s_imagenet](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_s_imagenet.h5) |

  | Model              | Params | FLOPs | Input | Top1 Acc | Download |
  | ------------------ | ------ | ----- | ----- | -------- | -------- |
  | MobileViT_V2_050   | 1.37M  | 0.47G | 256   | 70.18    | [v2_050_256_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_v2_050_256_imagenet.h5) |
  | MobileViT_V2_075   | 2.87M  | 1.04G | 256   | 75.56    | [v2_075_256_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_v2_075_256_imagenet.h5) |
  | MobileViT_V2_100   | 4.90M  | 1.83G | 256   | 78.09    | [v2_100_256_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_v2_100_256_imagenet.h5) |
  | MobileViT_V2_125   | 7.48M  | 2.84G | 256   | 79.65    | [v2_125_256_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_v2_125_256_imagenet.h5) |
  | MobileViT_V2_150   | 10.6M  | 4.07G | 256   | 80.38    | [v2_150_256_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_v2_150_256_imagenet.h5) |
  | - imagenet22k      | 10.6M  | 4.07G | 256   | 81.46    | [v2_150_256_imagenet22k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_v2_150_256_imagenet22k.h5) |
  | - imagenet22k, 384 | 10.6M  | 9.15G | 384   | 82.60    | [v2_150_384_imagenet22k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_v2_150_384_imagenet22k.h5) |
  | MobileViT_V2_175   | 14.3M  | 5.52G | 256   | 80.84    | [v2_175_256_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_v2_175_256_imagenet.h5) |
  | - imagenet22k      | 14.3M  | 5.52G | 256   | 81.94    | [v2_175_256_imagenet22k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_v2_175_256_imagenet22k.h5) |
  | - imagenet22k, 384 | 14.3M  | 12.4G | 384   | 82.93    | [v2_175_384_imagenet22k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_v2_175_384_imagenet22k.h5) |
  | MobileViT_V2_200   | 18.4M  | 7.12G | 256   | 81.17    | [v2_200_256_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_v2_200_256_imagenet.h5) |
  | - imagenet22k      | 18.4M  | 7.12G | 256   | 82.36    | [v2_200_256_imagenet22k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_v2_200_256_imagenet22k.h5) |
  | - imagenet22k, 384 | 18.4M  | 16.2G | 384   | 83.41    | [v2_200_384_imagenet22k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilevit/mobilevit_v2_200_384_imagenet22k.h5) |
## Usage
  ```py
  from keras_cv_attention_models import mobilevit

  # Will download and load pretrained imagenet weights.
  mm = mobilevit.MobileViT_XXS(pretrained="imagenet")

  # Run prediction
  from skimage.data import chelsea
  import tensorflow as tf
  from tensorflow import keras
  imm = tf.expand_dims(tf.image.resize(chelsea(), mm.input_shape[1:3]), 0) / 255 # Chelsea the cat
  pred = mm(imm).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.6774389), ('n02123045', 'tabby', 0.12461892), ...]
  ```
  **Change input resolution**. For input resolution not divisible by `64`, will apply `tf.image.resize` for transformer blocks.
  ```py
  from keras_cv_attention_models import mobilevit
  mm = mobilevit.MobileViT_V2_100(input_shape=(260, 277, 3), pretrained="imagenet")
  # >>>> Load pretrained from: ~/.keras/models/mobilevit_v2_100_256_imagenet.h5

  # Run prediction
  from skimage.data import chelsea
  preds = mm(mm.preprocess_input(chelsea()))
  print(mm.decode_predictions(preds))
  # [[('n02124075', 'Egyptian_cat', 0.38652435), ('n02123159', 'tiger_cat', 0.2578847), ...]
  ```
## Verification with PyTorch version
  ```py
  """ PyTorch mobilevit_s """
  sys.path.append('../pytorch-image-models/')
  import timm
  torch_model = timm.models.mobilevit_s(pretrained=True)
  _ = torch_model.eval()

  """ Keras MobileViT_S """
  from keras_cv_attention_models import mobilevit
  mm = mobilevit.MobileViT_S(pretrained="imagenet", classifier_activation=None)

  """ Verification """
  import torch
  inputs = np.random.uniform(size=(1, *mm.input_shape[1:3], 3)).astype("float32")
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()
  keras_out = mm(inputs).numpy()
  print(f"{np.allclose(torch_out, keras_out, atol=1e-3) = }")
  # np.allclose(torch_out, keras_out, atol=1e-3) = True
  ```
***
