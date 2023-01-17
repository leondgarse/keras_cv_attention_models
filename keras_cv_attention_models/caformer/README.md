# ___Keras CAFormer___
***

## Summary
  - Keras implementation of [Github sail-sg/metaformer](https://github.com/sail-sg/metaformer). Paper [PDF 2210.13452 MetaFormer Baselines for Vision](https://arxiv.org/pdf/2210.13452.pdf).
  - Model weights ported from official publication.
  - `CAFormer` is using 2 transformer stacks by `block_types=["conv", "conv", "transform", "transform"]`, while `ConvFormer` is all conv blocks by `block_types=["conv", "conv", "conv", "conv"]`.
  - Model weights are currently load and converted from huggingface pth link, as there are too many...
***

## Models
  | Model              | Params | FLOPs | Input | Top1 Acc | Download |
  | ------------------ | ------ | ----- | ----- | -------- | -------- |
  | CAFormerS18        | 26M    | 4.1G  | 224   | 83.6     | [caformer_small18_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_small18_224_imagenet.h5) |
  |                    | 26M    | 13.4G | 384   | 85.0     | [caformer_small18_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_small18_384_imagenet.h5) |
  | - imagenet21k-ft1k | 26M    | 4.1G  | 224   | 84.1     | [caformer_small18_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_small18_224_imagenet21k-ft1k.h5) |
  | - imagenet21k-ft1k | 26M    | 13.4G | 384   | 85.4     | [caformer_small18_384_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_small18_384_imagenet21k-ft1k.h5) |
  | CAFormerS36        | 39M    | 8.0G  | 224   | 84.5     | Load from huggingface pth |
  |                    | 39M    | 26.0G | 384   | 85.7     | Load from huggingface pth |
  | - imagenet21k-ft1k | 39M    | 8.0G  | 224   | 85.8     | Load from huggingface pth |
  | - imagenet21k-ft1k | 39M    | 26.0G | 384   | 86.9     | Load from huggingface pth |
  | CAFormerM36        | 56M    | 13.2G | 224   | 85.2     | Load from huggingface pth |
  |                    | 56M    | 42.0G | 384   | 86.2     | Load from huggingface pth |
  | - imagenet21k-ft1k | 56M    | 13.2G | 224   | 86.6     | Load from huggingface pth |
  | - imagenet21k-ft1k | 56M    | 42.0G | 384   | 87.5     | Load from huggingface pth |
  | CAFormerB36        | 99M    | 23.2G | 224   | 85.5     | Load from huggingface pth |
  |                    | 99M    | 72.2G | 384   | 86.4     | Load from huggingface pth |
  | - imagenet21k-ft1k | 99M    | 23.2G | 224   | 87.4     | Load from huggingface pth |
  | - imagenet21k-ft1k | 99M    | 72.2G | 384   | 88.1     | Load from huggingface pth |

  | Model              | Params | FLOPs | Input | Top1 Acc | Download |
  | ------------------ | ------ | ----- | ----- | -------- | -------- |
  | ConvFormerS18      | 27M    | 3.9G  | 224   | 83.0     | Load from huggingface pth |
  |                    | 27M    | 11.6G | 384   | 84.4     | Load from huggingface pth |
  | - imagenet21k-ft1k | 27M    | 3.9G  | 224   | 83.7     | Load from huggingface pth |
  | - imagenet21k-ft1k | 27M    | 11.6G | 384   | 85.0     | Load from huggingface pth |
  | ConvFormerS36      | 40M    | 7.6G  | 224   | 84.1     | Load from huggingface pth |
  |                    | 40M    | 22.4G | 384   | 85.4     | Load from huggingface pth |
  | - imagenet21k-ft1k | 40M    | 7.6G  | 224   | 85.4     | Load from huggingface pth |
  | - imagenet21k-ft1k | 40M    | 22.4G | 384   | 86.4     | Load from huggingface pth |
  | ConvFormerM36      | 57M    | 12.8G | 224   | 84.5     | Load from huggingface pth |
  |                    | 57M    | 37.7G | 384   | 85.6     | Load from huggingface pth |
  | - imagenet21k-ft1k | 57M    | 12.8G | 224   | 86.1     | Load from huggingface pth |
  | - imagenet21k-ft1k | 57M    | 37.7G | 384   | 86.9     | Load from huggingface pth |
  | ConvFormerB36      | 100M   | 22.6G | 224   | 84.8     | Load from huggingface pth |
  |                    | 100M   | 66.5G | 384   | 85.7     | Load from huggingface pth |
  | - imagenet21k-ft1k | 100M   | 22.6G | 224   | 87.0     | Load from huggingface pth |
  | - imagenet21k-ft1k | 100M   | 66.5G | 384   | 87.6     | Load from huggingface pth |
## Usage
  ```py
  from keras_cv_attention_models import caformer

  # Will download and load pretrained imagenet weights.
  mm = caformer.CAFormerS18(pretrained="imagenet")

  # Run prediction
  import tensorflow as tf
  from tensorflow import keras
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.77616554), ('n02123159', 'tiger_cat', 0.042136233), ...]
  ```
  **Change input resolution**.
  ```py
  from keras_cv_attention_models import caformer
  mm = caformer.CAFormerS18(input_shape=(212, 193, 3), pretrained="imagenet")
  # >>>> Load pretrained from: ~/.keras/models/caformer_small18_224_imagenet.h5

  # Run prediction
  from skimage.data import chelsea
  preds = mm(mm.preprocess_input(chelsea()))
  print(mm.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.7891602), ('n02123159', 'tiger_cat', 0.039598733), ...]
  ```
## Verification with PyTorch version
  ```py
  """ PyTorch caformer_s18 """
  sys.path.append('../metaformer')
  sys.path.append('../pytorch-image-models/')
  import torch
  import metaformer_baselines
  torch_model = metaformer_baselines.caformer_s18(pretrained=True)
  _ = torch_model.eval()

  """ Keras CAFormerS18 """
  from keras_cv_attention_models import caformer
  mm = caformer.CAFormerS18(pretrained="imagenet", classifier_activation=None)

  """ Verification """
  inputs = np.random.uniform(size=(1, *mm.input_shape[1:3], 3)).astype("float32")
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()
  keras_out = mm(inputs).numpy()
  print(f"{np.allclose(torch_out, keras_out, atol=1e-5) = }")
  # np.allclose(torch_out, keras_out, atol=1e-5) = True
  ```
