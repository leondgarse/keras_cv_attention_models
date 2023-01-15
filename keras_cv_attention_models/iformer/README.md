# ___Keras InceptionTransformer___
***

## Summary
  - Keras implementation of [Github sail-sg/iFormer](https://github.com/sail-sg/iFormer). Paper [PDF 2205.12956 Inception Transformer](https://arxiv.org/pdf/2205.12956.pdf).
  - Model weights ported from official publication.
***

## Models
  | Model        | Params | FLOPs  | Input | Top1 Acc | Download |
  | ------------ | ------ | ------ | ----- | -------- | -------- |
  | IFormerSmall | 19.9M  | 4.88G  | 224   | 83.4     | [iformer_small_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/iformer/iformer_small_224_imagenet.h5) |
  |              | 20.9M  | 16.29G | 384   | 84.6     | [iformer_small_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/iformer/iformer_small_384_imagenet.h5) |
  | IFormerBase  | 47.9M  | 9.44G  | 224   | 84.6     | [iformer_base_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/iformer/iformer_base_224_imagenet.h5) |
  |              | 48.9M  | 30.86G | 384   | 85.7     | [iformer_base_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/iformer/iformer_base_384_imagenet.h5) |
  | IFormerLarge | 86.6M  | 14.12G | 224   | 84.6     | [iformer_large_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/iformer/iformer_largel_224_imagenet.h5) |
  |              | 87.7M  | 45.74G | 384   | 85.8     | [iformer_large_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/iformer/iformer_largel_384_imagenet.h5) |
## Usage
  ```py
  from keras_cv_attention_models import iformer

  # Will download and load pretrained imagenet weights.
  mm = iformer.IFormerSmall(pretrained="imagenet")

  # Run prediction
  import tensorflow as tf
  from tensorflow import keras
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.7471715), ('n02123159', 'tiger_cat', 0.035306472), ...]
  ```
  **Change input resolution**.
  ```py
  from keras_cv_attention_models import iformer
  mm = iformer.IFormerSmall(input_shape=(512, 393, 3), pretrained="imagenet")
  # >>>> Load pretrained from: ~/.keras/models/iformer_small_384_imagenet.h5
  # >>>> Reload mismatched weights: 384 -> (512, 393)
  # >>>> Reload layer: stack1_positional_embedding
  # ...

  # Run prediction
  from skimage.data import chelsea
  preds = mm(mm.preprocess_input(chelsea()))
  print(mm.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.72780704), ('n02123159', 'tiger_cat', 0.11522171), ...]
  ```
## Verification with PyTorch version
  ```py
  """ PyTorch iformer_small """
  sys.path.append('../iFormer/')
  sys.path.append('../pytorch-image-models/')  # Needs timm
  import torch
  from models import inception_transformer

  torch_model = inception_transformer.iformer_small(pretrained=True)
  _ = torch_model.eval()

  """ Keras IFormerSmall """
  from keras_cv_attention_models import iformer
  mm = iformer.IFormerSmall(pretrained="imagenet", classifier_activation=None)

  """ Verification """
  inputs = np.random.uniform(size=(1, *mm.input_shape[1:3], 3)).astype("float32")
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()
  keras_out = mm(inputs).numpy()
  print(f"{np.allclose(torch_out, keras_out, atol=1e-5) = }")
  # np.allclose(torch_out, keras_out, atol=1e-5) = True
  ```
