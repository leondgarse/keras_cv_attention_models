# ___Keras PyramidVisionTransformerV2___
***

## Summary
  - Keras implementation of [Github whai362/PVT](https://github.com/whai362/PVT/tree/v2/classification). Paper [PDF 2106.13797 PVTv2: Improved Baselines with Pyramid Vision Transformer](https://arxiv.org/pdf/2106.13797.pdf).
  - Model weights ported from official publication.
***

## Models
  | Model           | Params | FLOPs  | Input | Top1 Acc | Download |
  | --------------- | ------ | ------ | ----- | -------- | -------- |
  | PVT_V2B0        | 3.7M   | 580.3M | 224   | 70.5     | [pvt_v2_b0_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/pvt/pvt_v2_b0_imagenet.h5) |
  | PVT_V2B1        | 14.0M  | 2.14G  | 224   | 78.7     | [pvt_v2_b1_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/pvt/pvt_v2_b1_imagenet.h5) |
  | PVT_V2B2        | 25.4M  | 4.07G  | 224   | 82.0     | [pvt_v2_b2_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/pvt/pvt_v2_b2_imagenet.h5) |
  | PVT_V2B2_linear | 22.6M  | 3.94G  | 224   | 82.1     | [pvt_v2_b2_linear.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/pvt/pvt_v2_b2_linear_imagenet.h5) |
  | PVT_V2B3        | 45.2M  | 6.96G  | 224   | 83.1     | [pvt_v2_b3_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/pvt/pvt_v2_b3_imagenet.h5) |
  | PVT_V2B4        | 62.6M  | 10.19G | 224   | 83.6     | [pvt_v2_b4_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/pvt/pvt_v2_b4_imagenet.h5) |
  | PVT_V2B5        | 82.0M  | 11.81G | 224   | 83.8     | [pvt_v2_b5_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/pvt/pvt_v2_b5_imagenet.h5) |
## Usage
  ```py
  from keras_cv_attention_models import pvt

  # Will download and load pretrained imagenet weights.
  mm = pvt.PVT_V2B2(pretrained="imagenet")

  # Run prediction
  import tensorflow as tf
  from tensorflow import keras
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.6658455), ('n02123159', 'tiger_cat', 0.08825972), ...]
  ```
  **Change input resolution**. Note: for `PVT_V2B2_linear` using `addaptive_pooling_2d` with `output_size=7`, input shape should be lager than `193`.
  ```py
  from keras_cv_attention_models import pvt
  mm = pvt.PVT_V2B1(input_shape=(128, 192, 3), pretrained="imagenet")
  # >>>> Load pretrained from: ~/.keras/models/pvt_v2_b1_imagenet.h5

  # Run prediction
  from skimage.data import chelsea
  preds = mm(mm.preprocess_input(chelsea()))
  print(mm.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.8482509), ('n02123045', 'tabby', 0.07139703), ...]
  ```
## Verification with PyTorch version
  ```py
  """ PyTorch pvt_v2_b0 """
  sys.path.append('../PVT-2/')
  sys.path.append('../pytorch-image-models/')  # Needs timm
  import torch
  from classification import pvt_v2

  torch_model = pvt_v2.pvt_v2_b0()
  ss = torch.load('pvt_v2_b0.pth', map_location=torch.device('cpu'))
  torch_model.load_state_dict(ss)
  _ = torch_model.eval()

  """ Keras PVT_V2B0 """
  from keras_cv_attention_models import pvt
  mm = pvt.PVT_V2B0(pretrained="imagenet", classifier_activation=None)

  """ Verification """
  inputs = np.random.uniform(size=(1, *mm.input_shape[1:3], 3)).astype("float32")
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()
  keras_out = mm(inputs).numpy()
  print(f"{np.allclose(torch_out, keras_out, atol=1e-5) = }")
  # np.allclose(torch_out, keras_out, atol=1e-5) = True
  ```
