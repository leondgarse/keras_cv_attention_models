# ___Keras EdgeNeXt___
***

## Summary
  - Keras implementation of [Github mmaaz60/EdgeNeXt](https://github.com/mmaaz60/EdgeNeXt). Paper [PDF 2206.10589 EdgeNeXt: Efficiently Amalgamated CNN-Transformer Architecture for Mobile Vision Applications](https://arxiv.org/pdf/2206.10589.pdf).
  - Model weights reloaded from official publication.
  - Related usi distillation paper [PDF 2204.03475 Solving ImageNet: a Unified Scheme for Training any Backbone to Top Results](https://arxiv.org/pdf/2204.03475.pdf).
***

## Models
  | Model             | Params | FLOPs  | Input | Top1 Acc | Download |
  | ----------------- | ------ | ------ | ----- | -------- | -------- |
  | EdgeNeXt_XX_Small | 1.33M  | 266M   | 256   | 71.23    | [edgenext_xx_small_256_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/edgenext/edgenext_xx_small_256_imagenet.h5) |
  | EdgeNeXt_X_Small  | 2.34M  | 547M   | 256   | 74.96    | [edgenext_x_small_256_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/edgenext/edgenext_x_small_256_imagenet.h5) |
  | EdgeNeXt_Small    | 5.59M  | 1.27G  | 256   | 79.41    | [edgenext_small_256_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/edgenext/edgenext_small_256_imagenet.h5) |
  | - usi             | 5.59M  | 1.27G  | 256   | 81.07    | [edgenext_small_256_usi.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/edgenext/edgenext_small_256_usi.h5) |
## Usage
  ```py
  from keras_cv_attention_models import edgenext

  # Will download and load pretrained imagenet weights.
  mm = edgenext.EdgeNeXt_XX_Small(pretrained="imagenet")

  # Run prediction
  import tensorflow as tf
  from tensorflow import keras
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.60692847), ('n02123045', 'tabby', 0.21328166), ...]
  ```
  **Change input resolution**
  ```py
  from keras_cv_attention_models import edgenext
  mm = edgenext.EdgeNeXt_Small(input_shape=(174, 269, 3), pretrained="usi")
  # >>>> Load pretrained from: ~/.keras/models/edgenext_small_256_usi.h5

  # Run prediction
  from skimage.data import chelsea
  preds = mm(mm.preprocess_input(chelsea()))
  print(mm.decode_predictions(preds))
  # [[('n02124075', 'Egyptian_cat', 0.8444098), ('n02123159', 'tiger_cat', 0.061309356), ...]
  ```
## Verification with PyTorch version
  ```py
  """ PyTorch edgenext_small """
  sys.path.append('../EdgeNeXt/')
  sys.path.append('../pytorch-image-models/')  # Needs timm
  import torch
  from models import model
  torch_model = model.edgenext_small(classifier_dropout=0)
  _ = torch_model.eval()
  ss = torch.load('edgenext_small_usi.pth', map_location=torch.device('cpu'))
  torch_model.load_state_dict(ss.get('state_dict', ss.get('model', ss)))

  """ Keras EdgeNeXt_Small """
  from keras_cv_attention_models import edgenext
  mm = edgenext.EdgeNeXt_Small(pretrained="usi", classifier_activation=None)

  """ Verification """
  inputs = np.random.uniform(size=(1, *mm.input_shape[1:3], 3)).astype("float32")
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()
  keras_out = mm(inputs).numpy()
  print(f"{np.allclose(torch_out, keras_out, atol=1e-5) = }")
  # np.allclose(torch_out, keras_out, atol=1e-5) = True
  ```
***
