# ___Keras Neighborhood Attention Transformer___
***

## Summary
  - Keras implementation of [NAT](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer).
  - `NAT` Paper [PDF 2204.07143 Neighborhood Attention Transformer](https://arxiv.org/pdf/2204.07143.pdf).
  - `DiNAT` Paper [PDF 2209.15001 Dilated Neighborhood Attention Transformer](https://arxiv.org/pdf/2209.15001.pdf).
  - DiNAT is identical to NAT in architecture, with every other layer replaced with Dilated Neighborhood Attention. These variants provide similar or better classification accuracy (except for Tiny), but yield significantly better downstream performance.
  - Model weights reloaded from official publication.

  ![nat](https://user-images.githubusercontent.com/5744524/167790694-134a5f58-dac1-4b6f-ae49-5b24013e4f23.png)
***

## Models
  | Model     | Params | FLOPs  | Input | Top1 Acc | Download |
  | --------- | ------ | ------ | ----- | -------- | -------- |
  | NAT_Mini  | 20.0M  | 2.73G  | 224   | 81.8     | [nat_mini_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nat/nat_mini_imagenet.h5) |
  | NAT_Tiny  | 27.9M  | 4.34G  | 224   | 83.2     | [nat_tiny_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nat/nat_tiny_imagenet.h5) |
  | NAT_Small | 50.7M  | 7.84G  | 224   | 83.7     | [nat_small_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nat/nat_small_imagenet.h5) |
  | NAT_Base  | 89.8M  | 13.76G | 224   | 84.3     | [nat_base_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nat/nat_base_imagenet.h5) |

  | Model                     | Params | FLOPs  | Input | Top1 Acc | Download |
  | ------------------------- | ------ | ------ | ----- | -------- | -------- |
  | DiNAT_Mini                | 20.0M  | 2.73G  | 224   | 81.8     | [dinat_mini_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nat/dinat_mini_imagenet.h5) |
  | DiNAT_Tiny                | 27.9M  | 4.34G  | 224   | 82.7     | [dinat_tiny_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nat/dinat_tiny_imagenet.h5) |
  | DiNAT_Small               | 50.7M  | 7.84G  | 224   | 83.8     | [dinat_small_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nat/dinat_small_imagenet.h5) |
  | DiNAT_Base                | 89.8M  | 13.76G | 224   | 84.4     | [dinat_base_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nat/dinat_base_imagenet.h5) |
  | DiNAT_Large, 22k          | 200.9M | 30.58G | 224   | 86.6     | [dinat_large_224_imagenet21k-ft1k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nat/dinat_large_224_imagenet21k-ft1k.h5) |
  | - 21k num_classes=21841   | 200.9M | 30.58G | 224   |          | [dinat_large_imagenet21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nat/dinat_large_imagenet21k.h5) |
  | - 22k, 384                | 200.9M | 89.86G | 384   | 87.4     | [dinat_large_384_imagenet21k-ft1k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nat/dinat_large_384_imagenet21k-ft1k.h5) |
  | DiNAT_Large_K11, 22k, 384 | 201.1M | 92.57G | 384   | 87.5     | [dinat_large_k11_imagenet21k-ft1k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/nat/dinat_large_k11_imagenet21k-ft1k.h5) |
## Usage
  ```py
  from keras_cv_attention_models import nat

  # Will download and load pretrained imagenet weights.
  mm = nat.NAT_Mini(pretrained="imagenet")

  # Run prediction
  import tensorflow as tf
  from tensorflow import keras
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.7978939), ('n02123159', 'tiger_cat', 0.054351762), ...]
  ```
  **Change input resolution**
  ```py
  from keras_cv_attention_models import nat
  mm = nat.DiNAT_Mini(input_shape=(374, 269, 3), pretrained="imagenet")
  # >>>> Load pretrained from: ~/.keras/models/dinat_mini_imagenet.h5

  # Run prediction
  from skimage.data import chelsea
  preds = mm(mm.preprocess_input(chelsea()))
  print(mm.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.39229804), ('n02123159', 'tiger_cat', 0.36450642), ...]
  ```
## Verification with PyTorch version
  ```py
  """ PyTorch dinat_mini """
  sys.path.append('../Neighborhood-Attention-Transformer/classification/')
  sys.path.append('../pytorch-image-models/')  # Needs timm
  import dinat as torch_dinat
  import torch

  torch_model = torch_dinat.dinat_mini(pretrained=True)
  # torch_model.load_state_dict(torch.load('nat_mini.pth'))
  _ = torch_model.eval()

  """ Keras DiNAT_Mini """
  from keras_cv_attention_models import nat
  mm = nat.DiNAT_Mini(pretrained="imagenet", classifier_activation=None)

  """ Verification """
  inputs = np.random.uniform(size=(1, *mm.input_shape[1:3], 3)).astype("float32")
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()
  keras_out = mm(inputs).numpy()
  print(f"{np.allclose(torch_out, keras_out, atol=1e-5) = }")
  # np.allclose(torch_out, keras_out, atol=1e-5) = True
  ```
***
