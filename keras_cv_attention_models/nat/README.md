# ___Keras Neighborhood Attention Transformer___
***

## Summary
  - Keras implementation of [NAT](https://github.com/SHI-Labs/Neighborhood-Attention-Transformer). Paper [PDF 2204.07143 Neighborhood Attention Transformer](https://arxiv.org/pdf/2204.07143.pdf).
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
  mm = nat.NAT_Mini(input_shape=(374, 269, 3), pretrained="imagenet")
  # >>>> Load pretrained from: ~/.keras/models/nat_mini_imagenet.h5

  # Run prediction
  from skimage.data import chelsea
  preds = mm(mm.preprocess_input(chelsea()))
  print(mm.decode_predictions(preds))
  # [[('n02124075', 'Egyptian_cat', 0.55214083), ('n02123045', 'tabby', 0.2569016), ...]
  ```
## Verification with PyTorch version
  ```py
  """ PyTorch nat_mini """
  sys.path.append('../Neighborhood-Attention-Transformer/classification/')
  sys.path.append('../pytorch-image-models/')  # Needs timm
  import nat as torch_nat
  import torch

  torch_model = torch_nat.nat_mini()
  torch_model.load_state_dict(torch.load('nat_mini.pth'))
  _ = torch_model.eval()

  """ Keras NAT_Mini """
  from keras_cv_attention_models import nat
  mm = nat.NAT_Mini(pretrained="imagenet", classifier_activation=None)

  """ Verification """
  inputs = np.random.uniform(size=(1, *mm.input_shape[1:3], 3)).astype("float32")
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()
  keras_out = mm(inputs).numpy()
  print(f"{np.allclose(torch_out, keras_out, atol=1e-5) = }")
  # np.allclose(torch_out, keras_out, atol=1e-5) = True
  ```
***
