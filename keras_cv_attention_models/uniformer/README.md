# ___Keras UniFormer___
***

## Summary
  - Keras implementation of [UniFormer](https://github.com/Sense-X/UniFormer/tree/main/image_classification). Paper [PDF 2201.09450 UniFormer: Unifying Convolution and Self-attention for Visual Recognition](https://arxiv.org/pdf/2201.09450.pdf).
  - Other Related papers: [PDF 2104.10858 All Tokens Matter: Token Labeling for Training Better Vision Transformers](https://arxiv.org/pdf/2104.10858.pdf), [PDF 2103.17239 Going deeper with Image Transformers](https://arxiv.org/pdf/2103.17239.pdf).
  - Model weights reloaded from official publication.

  ![uniformer](https://user-images.githubusercontent.com/5744524/157807693-f2508131-2ee9-4f60-9f55-722fde3b218c.png)
***

## Models
  - It's claimed the `token_label` model works better for down stream tasks like object detection. Reload those weights by `pretrained="token_label"`.

  | Model                 | Params | FLOPs  | Input | Top1 Acc | Download |
  | --------------------- | ------ | ------ | ----- | -------- | -------- |
  | UniformerSmall32 + TL | 22M    | 3.66G  | 224   | 83.4     | [small_32_224_token_label](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_small_32_224_token_label.h5) |
  | UniformerSmall64      | 22M    | 3.66G  | 224   | 82.9     | [small_64_imagenet](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_small_64_224_imagenet.h5) |
  | - Token Labeling      | 22M    | 3.66G  | 224   | 83.4     | [small_64_token_label](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_small_64_224_token_label.h5) |
  | UniformerSmallPlus32  | 24M    | 4.24G  | 224   | 83.4     | [small_plus_32_imagenet](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_small_plus_32_224_imagenet.h5) |
  | - Token Labeling      | 24M    | 4.24G  | 224   | 83.9     | [small_plus_32_token_label](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_small_plus_32_224_token_label.h5) |
  | UniformerSmallPlus64  | 24M    | 4.23G  | 224   | 83.4     | [small_plus_64_imagenet](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_small_plus_64_224_imagenet.h5) |
  | - Token Labeling      | 24M    | 4.23G  | 224   | 83.6     | [small_plus_64_token_label](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_small_plus_64_224_token_label.h5) |
  | UniformerBase32 + TL  | 50M    | 8.32G  | 224   | 85.1     | [base_32_224_token_label](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_base_32_224_token_label.h5) |
  | UniformerBase64       | 50M    | 8.31G  | 224   | 83.8     | [base_64_imagenet](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_base_64_224_imagenet.h5) |
  | - Token Labeling      | 50M    | 8.31G  | 224   | 84.8     | [base_64_224_token_label](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_base_64_224_token_label.h5) |
  | UniformerLarge64 + TL | 100M   | 19.79G | 224   | 85.6     | [large_64_224_token_label](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_large_64_224_token_label.h5) |
  | UniformerLarge64 + TL | 100M   | 63.11G | 384   | 86.3     | [large_64_384_token_label](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_large_64_384_token_label.h5) |
## Usage
  ```py
  from keras_cv_attention_models import uniformer

  # Will download and load pretrained imagenet weights.
  mm = uniformer.UniformerSmall64(pretrained="imagenet")

  # Run prediction
  import tensorflow as tf
  from tensorflow import keras
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.83922714), ('n02123159', 'tiger_cat', 0.014741183), ...]
  ```
  **Change input resolution**
  ```py
  from keras_cv_attention_models import uniformer
  mm = uniformer.UniformerSmallPlus32(input_shape=(512, 512, 3), pretrained="imagenet")
  # >>>> Load pretrained from: ~/.keras/models/uniformer_small_plus_32_224_imagenet.h5

  # Run prediction
  from skimage.data import chelsea
  preds = mm(mm.preprocess_input(chelsea()))
  print(mm.decode_predictions(preds))
  # [[('n02124075', 'Egyptian_cat', 0.37126896), ('n02123045', 'tabby', 0.16558096), ...]
  ```
## Verification with PyTorch version
  ```py
  """ PyTorch uniformer_small """
  import torch
  sys.path.append("../UniFormer")
  from image_classification.models import uniformer as torch_uniformer

  torch_model = torch_uniformer.uniformer_small()
  weights = torch.load('uniformer_small_in1k.pth')
  torch_model.load_state_dict(weights['model'] if "model" in weights else weights)
  torch_model.eval()

  """ Keras UniformerSmall64 """
  from keras_cv_attention_models import uniformer
  mm = uniformer.UniformerSmall64(pretrained="imagenet", classifier_activation=None)

  """ Verification """
  inputs = np.random.uniform(size=(1, *mm.input_shape[1:3], 3)).astype("float32")
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()
  keras_out = mm(inputs).numpy()
  print(f"{np.allclose(torch_out, keras_out, atol=1e-2) = }")
  # np.allclose(torch_out, keras_out, atol=1e-2) = True
  ```
  **Token labeling model**
  ```py
  """ PyTorch uniformer_small """
  import torch
  sys.path.append("../UniFormer")
  from image_classification.token_labeling.tlt.models import uniformer as torch_uniformer

  torch_model = torch_uniformer.uniformer_small()
  weights = torch.load('uniformer_small_tl_224.pth')
  torch_model.load_state_dict(weights['model'] if "model" in weights else weights)
  torch_model.eval()

  """ Keras UniformerSmall64 """
  from keras_cv_attention_models import uniformer
  mm = uniformer.UniformerSmall32(pretrained="token_label", token_label_top=True, classifier_activation=None)

  """ Verification """
  inputs = np.random.uniform(size=(1, *mm.input_shape[1:3], 3)).astype("float32")
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()

  keras_preds = mm(inputs)
  keras_out = (keras_preds[0] + 0.5 * tf.reduce_max(keras_preds[1], axis=1)).numpy()
  # keras_out = mm.decode_predictions(keras_preds, classifier_activation=None, do_decode=False).numpy()
  print(f"{np.allclose(torch_out, keras_out, atol=2e-2) = }")
  # np.allclose(torch_out, keras_out, atol=2e-2) = True
  ```
***
