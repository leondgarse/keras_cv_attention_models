# ___Keras CAFormer___
***

## Summary
  - Keras implementation of [Github sail-sg/metaformer](https://github.com/sail-sg/metaformer). Paper [PDF 2210.13452 MetaFormer Baselines for Vision](https://arxiv.org/pdf/2210.13452.pdf).
  - Model weights ported from official publication.
  - `CAFormer` is using 2 transformer stacks by `block_types=["conv", "conv", "transform", "transform"]`, while `ConvFormer` is all conv blocks by `block_types=["conv", "conv", "conv", "conv"]`.
***

## Models
  | Model              | Params | FLOPs | Input | Top1 Acc | Download |
  | ------------------ | ------ | ----- | ----- | -------- | -------- |
  | CAFormerS18        | 26M    | 4.1G  | 224   | 83.6     | [caformer_s18_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_s18_224_imagenet.h5) |
  | - imagenet21k-ft1k | 26M    | 4.1G  | 224   | 84.1     | [caformer_s18_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_s18_224_imagenet21k-ft1k.h5) |
  |                    | 26M    | 13.4G | 384   | 85.0     | [caformer_s18_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_s18_384_imagenet.h5) |
  | - imagenet21k-ft1k | 26M    | 13.4G | 384   | 85.4     | [caformer_s18_384_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_s18_384_imagenet21k-ft1k.h5) |
  | CAFormerS36        | 39M    | 8.0G  | 224   | 84.5     | [caformer_s36_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_s36_224_imagenet.h5) |
  | - imagenet21k-ft1k | 39M    | 8.0G  | 224   | 85.8     | [caformer_s36_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_s36_224_imagenet21k-ft1k.h5) |
  |                    | 39M    | 26.0G | 384   | 85.7     | [caformer_s36_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_s36_384_imagenet.h5) |
  | - imagenet21k-ft1k | 39M    | 26.0G | 384   | 86.9     | [caformer_s36_384_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_s36_384_imagenet21k-ft1k.h5) |
  | CAFormerM36        | 56M    | 13.2G | 224   | 85.2     | [caformer_m36_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_m36_224_imagenet.h5) |
  | - imagenet21k-ft1k | 56M    | 13.2G | 224   | 86.6     | [caformer_m36_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_m36_224_imagenet21k-ft1k.h5) |
  |                    | 56M    | 42.0G | 384   | 86.2     | [caformer_m36_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_m36_384_imagenet.h5) |
  | - imagenet21k-ft1k | 56M    | 42.0G | 384   | 87.5     | [caformer_m36_384_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_m36_384_imagenet21k-ft1k.h5) |
  | CAFormerB36        | 99M    | 23.2G | 224   | 85.5     | [caformer_b36_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_b36_224_imagenet.h5) |
  | - imagenet21k-ft1k | 99M    | 23.2G | 224   | 87.4     | [caformer_b36_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_b36_224_imagenet21k-ft1k.h5) |
  |                    | 99M    | 72.2G | 384   | 86.4     | [caformer_b36_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_b36_384_imagenet.h5) |
  | - imagenet21k-ft1k | 99M    | 72.2G | 384   | 88.1     | [caformer_b36_384_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/caformer_b36_384_imagenet21k-ft1k.h5) |

  | Model              | Params | FLOPs | Input | Top1 Acc | Download |
  | ------------------ | ------ | ----- | ----- | -------- | -------- |
  | ConvFormerS18      | 27M    | 3.9G  | 224   | 83.0     | [convformer_s18_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_s18_224_imagenet.h5) |
  | - imagenet21k-ft1k | 27M    | 3.9G  | 224   | 83.7     | [convformer_s18_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_s18_224_imagenet21k-ft1k.h5) |
  |                    | 27M    | 11.6G | 384   | 84.4     | [convformer_s18_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_s18_384_imagenet.h5) |
  | - imagenet21k-ft1k | 27M    | 11.6G | 384   | 85.0     | [convformer_s36_384_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_s36_384_imagenet21k-ft1k.h5) |
  | ConvFormerS36      | 40M    | 7.6G  | 224   | 84.1     | [convformer_s36_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_s36_224_imagenet.h5) |
  | - imagenet21k-ft1k | 40M    | 7.6G  | 224   | 85.4     | [convformer_s36_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_s36_224_imagenet21k-ft1k.h5) |
  |                    | 40M    | 22.4G | 384   | 85.4     | [convformer_s36_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_s36_384_imagenet.h5) |
  | - imagenet21k-ft1k | 40M    | 22.4G | 384   | 86.4     | [convformer_s36_384_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_s36_384_imagenet21k-ft1k.h5) |
  | ConvFormerM36      | 57M    | 12.8G | 224   | 84.5     | [convformer_m36_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_m36_224_imagenet.h5) |
  | - imagenet21k-ft1k | 57M    | 12.8G | 224   | 86.1     | [convformer_m36_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_m36_224_imagenet21k-ft1k.h5) |
  |                    | 57M    | 37.7G | 384   | 85.6     | [convformer_m36_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_m36_384_imagenet.h5) |
  | - imagenet21k-ft1k | 57M    | 37.7G | 384   | 86.9     | [convformer_m36_384_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_m36_384_imagenet21k-ft1k.h5) |
  | ConvFormerB36      | 100M   | 22.6G | 224   | 84.8     | [convformer_b36_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_b36_224_imagenet.h5) |
  | - imagenet21k-ft1k | 100M   | 22.6G | 224   | 87.0     | [convformer_b36_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_b36_224_imagenet21k-ft1k.h5) |
  |                    | 100M   | 66.5G | 384   | 85.7     | [convformer_b36_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_b36_384_imagenet.h5) |
  | - imagenet21k-ft1k | 100M   | 66.5G | 384   | 87.6     | [convformer_b36_384_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/caformer/convformer_b36_384_imagenet21k-ft1k.h5) |
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
## Ablation Study from paper
  | Activation                         | ConvFormerS18 | CAFormerS18 |
  | ---------------------------------- | ------------- | ----------- |
  | star_relu (base line)              | **83.0**      | **83.6**    |
  | relu                               | 82.1 (-0.9)   | 82.9 (-0.7) |
  | squared_relu                       | 82.6 (-0.4)   | 83.4 (-0.2) |
  | gelu                               | 82.7 (-0.3)   | 83.4 (-0.2) |
  | **Branch outputscaling**           |               |             |
  | None                               | 82.8 (-0.2)   | 83.2 (-0.4) |
  | LayerScale                         | 82.8 (-0.0)   | 83.0 (-0.6) |
  | BranchScale                        | 82.9 (-0.1)   | 83.3 (-0.3) |
  | **Biases in each block**           |               |             |
  | Enable biases of Norm, FC and Conv | 83.0 (-0.0)   | 83.5 (-0.1) |
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
