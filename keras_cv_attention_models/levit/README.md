# ___Keras LeViT___
***

## Summary
  - [Github facebookresearch/LeViT](https://github.com/facebookresearch/LeViT).
  - LeViT article: [PDF 2104.01136 LeViT: a Vision Transformer in ConvNet’s Clothing for Faster Inference](https://arxiv.org/pdf/2104.01136.pdf).
## Models
  | Model                   | Params | FLOPs | Input | Top1 Acc | Download |
  | ----------------------- | ------ | ----- | ----- | -------- | -------- |
  | LeViT128S, distillation | 7.8M   | 0.31G | 224   | 76.6     | [levit128s_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/levit128s_imagenet.h5) |
  | LeViT128, distillation  | 9.2M   | 0.41G | 224   | 78.6     | [levit128_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/levit128_imagenet.h5) |
  | LeViT192, distillation  | 11M    | 0.66G | 224   | 80.0     | [levit192_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/levit192_imagenet.h5) |
  | LeViT256, distillation  | 19M    | 1.13G | 224   | 81.6     | [levit256_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/levit256_imagenet.h5) |
  | LeViT384, distillation  | 39M    | 2.36G | 224   | 82.6     | [levit384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/levit384_imagenet.h5) |
## Usage
  ```py
  from keras_cv_attention_models import levit

  # Will download and load pretrained imagenet weights.
  mm = levit.LeViT128(pretrained="imagenet", use_distillation=True, classifier_activation=None)
  print(mm.output_names, mm.output_shape)
  # ['head', 'distill_head'] [(None, 1000), (None, 1000)]

  # Run prediction
  import tensorflow as tf
  from tensorflow import keras
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0))
  pred = tf.nn.softmax((pred[0] + pred[1]) / 2).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.962495), ('n02123159', 'tiger_cat', 0.008833298), ...]
  ```
  set `use_distillation=False` for output only one head.
  ```py
  mm = levit.LeViT192(use_distillation=False, classifier_activation="softmax")
  print(mm.output_names, mm.output_shape)
  # ['head'] (None, 1000)
  ```
  **Change input resolution**
  ```py
  from keras_cv_attention_models import levit
  # Will download and load pretrained imagenet weights.
  mm = levit.LeViT256(input_shape=(292, 213, 3), pretrained="imagenet", use_distillation=True, classifier_activation=None)
  # >>>> Load pretrained from: ~/.keras/models/levit256_imagenet.h5
  # WARNING:tensorflow:Skipping loading of weights for layer stack1_block1_attn_pos due to mismatch in shape ((266, 4) vs (196, 4)).
  # ...
  # >>>> Reload mismatched PositionalEmbedding weights: 224 -> (292, 213)
  # >>>> Reload layer: stack1_block1_attn_pos
  # ...

  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0))
  pred = tf.nn.softmax((pred[0] + pred[1]) / 2).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.8671425), ('n02123159', 'tiger_cat', 0.06060877), ...]
  ```
## Verification with PyTorch version
  ```py
  """ PyTorch LeViT_128 """
  import torch
  sys.path.append('../LeViT')
  import levit as torch_levit
  torch_model = torch_levit.LeViT_128(pretrained=True)
  torch_model.eval()

  input_shape = 224
  inputs = np.random.uniform(size=(1, input_shape, input_shape, 3)).astype("float32")
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()

  """ Keras LeViT_128 """
  from keras_cv_attention_models import levit
  mm = levit.LeViT128(pretrained="imagenet", use_distillation=True, classifier_activation=None)
  pred = mm(inputs)
  keras_out = ((pred[0] + pred[1]) / 2).numpy()

  """ Verification """
  print(f"{np.allclose(torch_out, keras_out, atol=1e-5) = }")
  # np.allclose(torch_out, keras_out, atol=1e-5) = True
  ```
***
