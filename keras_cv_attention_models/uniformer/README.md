# ___Keras UniFormer___
***

## Summary
  - Keras implementation of [UniFormer](https://github.com/Sense-X/UniFormer/tree/main/image_classification). Paper [PDF 2201.09450 UniFormer: Unifying Convolution and Self-attention for Visual Recognition](https://arxiv.org/pdf/2201.09450.pdf).
  - Model weights reloaded from official publication.
  - Related papers: [PDF 2104.10858 All Tokens Matter: Token Labeling for Training Better Vision Transformers](https://arxiv.org/pdf/2104.10858.pdf), [PDF 2103.17239 Going deeper with Image Transformers](https://arxiv.org/pdf/2103.17239.pdf).
***

## Models
  | Model                 | Params | Image  resolution | Top1 Acc | Download |
  | --------------------- | ------ | ----------------- | -------- | -------- |
  | UniformerSmall32 + TL | 22M    | 224               | 83.4     |          |
  | UniformerSmall64      | 22M    | 224               | 82.9     | [small_64_imagenet](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_small_64_224_imagenet.h5) |
  | - Token Labeling      | 22M    | 224               | 83.4     |          |
  | UniformerSmallPlus32  | 24M    | 224               | 83.4     | [small_plus_32_imagenet](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_small_plus_32_224_imagenet.h5) |
  | - Token Labeling      | 24M    | 224               | 83.9     |          |
  | UniformerSmallPlus64  | 24M    | 224               | 83.4     | [small_plus_64_imagenet](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_small_plus_64_224_imagenet.h5) |
  | - Token Labeling      | 24M    | 224               | 83.6     |          |
  | UniformerBase32 + TL  | 50M    | 224               | 85.1     |          |
  | UniformerBase64       | 50M    | 224               | 83.8     | [base_64_imagenet](https://github.com/leondgarse/keras_cv_attention_models/releases/download/uniformer/uniformer_base_64_224_imagenet.h5) |
  | - Token Labeling      | 50M    | 224               | 84.8     |          |
  | UniformerLarge64 + TL | 100M   | 224               | 85.6     |          |
  | UniformerLarge64 + TL | 100M   | 384               | 86.3     |          |
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
  sys.path.append("../UniFormer/image_classification")
  from models import uniformer as torch_uniformer

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
***
