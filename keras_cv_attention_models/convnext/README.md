# ___Keras CovNeXt___
***

## Summary
  - CovNeXt article: [PDF 2201.03545 A ConvNet for the 2020s](https://arxiv.org/pdf/2201.03545.pdf). Model weights reloaded from [Github facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt).
  - CovNeXtV2 article: [PDF 2301.00808 ConvNeXt V2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/pdf/2301.00808.pdf). Model weights reloaded from [Github facebookresearch/ConvNeXt-V2](https://github.com/facebookresearch/ConvNeXt-V2).

  ![convnext](https://user-images.githubusercontent.com/5744524/212474537-1f88ca1e-f1fa-4679-98b9-5b6716e05079.png)
***

## Models
  | Model               | Params | FLOPs   | Input | Top1 Acc | Download |
  | ------------------- | ------ | ------- | ----- | -------- | -------- |
  | ConvNeXtTiny        | 28M    | 4.49G   | 224   | 82.1     | [tiny_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_tiny_imagenet.h5) |
  | - ImageNet21k-ft1k  | 28M    | 4.49G   | 224   | 82.9     | [tiny_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_tiny_224_imagenet21k-ft1k.h5) |
  | - ImageNet21k-ft1k  | 28M    | 13.19G  | 384   | 84.1     | [tiny_384_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_tiny_384_imagenet21k-ft1k.h5) |
  | ConvNeXtSmall       | 50M    | 8.73G   | 224   | 83.1     | [small_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_small_imagenet.h5) |
  | - ImageNet21k-ft1k  | 50M    | 8.73G   | 224   | 84.6     | [small_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_small_224_imagenet21k-ft1k.h5) |
  | - ImageNet21k-ft1k  | 50M    | 25.67G  | 384   | 85.8     | [small_384_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_small_384_imagenet21k-ft1k.h5) |
  | ConvNeXtBase        | 89M    | 15.42G  | 224   | 83.8     | [base_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_base_224_imagenet.h5) |
  | ConvNeXtBase        | 89M    | 45.32G  | 384   | 85.1     | [base_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_base_384_imagenet.h5) |
  | - ImageNet21k-ft1k  | 89M    | 15.42G  | 224   | 85.8     | [base_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_base_224_imagenet21k-ft1k.h5) |
  | - ImageNet21k-ft1k  | 89M    | 45.32G  | 384   | 86.8     | [base_384_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_base_384_imagenet21k-ft1k.h5) |
  | ConvNeXtLarge       | 198M   | 34.46G  | 224   | 84.3     | [large_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_large_224_imagenet.h5) |
  | ConvNeXtLarge       | 198M   | 101.28G | 384   | 85.5     | [large_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_large_384_imagenet.h5) |
  | - ImageNet21k-ft1k  | 198M   | 34.46G  | 224   | 86.6     | [large_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_large_224_imagenet21k-ft1k.h5) |
  | - ImageNet21k-ft1k  | 198M   | 101.28G | 384   | 87.5     | [large_384_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_large_384_imagenet21k-ft1k.h5) |
  | ConvNeXtXLarge, 21k | 350M   | 61.06G  | 224   | 87.0     | [xlarge_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_xlarge_224_imagenet21k-ft1k.h5) |
  | ConvNeXtXLarge, 21k | 350M   | 179.43G | 384   | 87.8     | [xlarge_384_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_xlarge_384_imagenet21k-ft1k.h5) |

  - **ConvNeXtV2 models** Note: `ConvNeXtV2Huge` weights are in `float16` format, as `float32` ones are too large that exceed 2GB.

  | Model              | Params | FLOPs  | Input | Top1 Acc | Download |
  | ------------------ | ------ | ------ | ----- | -------- | -------- |
  | ConvNeXtV2Atto     | 3.7M   | 0.55G  | 224   | 76.7     | [v2_atto_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_atto_imagenet.h5) |
  | ConvNeXtV2Femto    | 5.2M   | 0.78G  | 224   | 78.5     | [v2_femto_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_femto_imagenet.h5) |
  | ConvNeXtV2Pico     | 9.1M   | 1.37G  | 224   | 80.3     | [v2_pico_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_pico_imagenet.h5) |
  | ConvNeXtV2Nano     | 15.6M  | 2.45G  | 224   | 81.9     | [v2_nano_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_nano_imagenet.h5) |
  | - ImageNet21k-ft1k | 15.6M  | 2.45G  | 224   | 82.1     | [v2_nano_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_nano_224_imagenet21k-ft1k.h5) |
  | - ImageNet21k-ft1k | 15.6M  | 7.21G  | 384   | 83.4     | [v2_nano_384_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_nano_384_imagenet21k-ft1k.h5) |
  | ConvNeXtV2Tiny     | 28.6M  | 4.47G  | 224   | 83.0     | [v2_tiny_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_tiny_imagenet.h5) |
  | - ImageNet21k-ft1k | 28.6M  | 4.47G  | 224   | 83.9     | [v2_tiny_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_tiny_224_imagenet21k-ft1k.h5) |
  | - ImageNet21k-ft1k | 28.6M  | 13.1G  | 384   | 85.1     | [v2_tiny_384_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_tiny_384_imagenet21k-ft1k.h5) |
  | ConvNeXtV2Base     | 89M    | 15.4G  | 224   | 84.9     | [v2_base_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_base_imagenet.h5) |
  | - ImageNet21k-ft1k | 89M    | 15.4G  | 224   | 86.8     | [v2_base_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_base_224_imagenet21k-ft1k.h5) |
  | - ImageNet21k-ft1k | 89M    | 45.2G  | 384   | 87.7     | [v2_base_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_base_384_imagenet21k-ft1k.h5) |
  | ConvNeXtV2Large    | 198M   | 34.4G  | 224   | 85.8     | [v2_large_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_large_imagenet.h5) |
  | - ImageNet21k-ft1k | 198M   | 34.4G  | 224   | 87.3     | [v2_large_224_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_large_224_imagenet21k-ft1k.h5) |
  | - ImageNet21k-ft1k | 198M   | 101.1G | 384   | 88.2     | [v2_large_384_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_large_384_imagenet21k-ft1k.h5) |
  | ConvNeXtV2Huge     | 660M   | 115G   | 224   | 86.3     | [v2_huge_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_huge_imagenet.h5) |
  | - ImageNet21k-ft1k | 660M   | 337.9G | 384   | 88.7     | [v2_huge_384_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_huge_384_imagenet21k-ft1k.h5) |
  | - ImageNet21k-ft1k | 660M   | 600.8G | 512   | 88.9     | [v2_huge_512_21k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_v2_huge_512_imagenet21k-ft1k.h5) |
## Usage
  ```py
  from keras_cv_attention_models import convnext
  mm = convnext.ConvNeXtBase()
  # >>>> Load pretrained from: ~/.keras/models/convnext_base_224_imagenet.h5

  # Run prediction
  import tensorflow as tf
  from tensorflow import keras
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.9021103), ('n02123159', 'tiger_cat', 0.017981088), ...]
  ```
  **Change input resolution**
  ```py
  from keras_cv_attention_models import convnext
  mm = convnext.ConvNeXtV2Nano(input_shape=(480, 480, 3), pretrained='imagenet21k-ft1k')
  # >>>> Load pretrained from: ~/.keras/models/convnext_v2_nano_384_imagenet21k-ft1k.h5

  from skimage.data import chelsea
  preds = mm(mm.preprocess_input(chelsea()))
  print(mm.decode_predictions(preds)[0])
  # [('n02124075', 'Egyptian_cat', 0.7427755), ('n02123159', 'tiger_cat', 0.092012934), ...]
  ```
  **Use dynamic input resolution** by set `input_shape=(None, None, 3)`.
  ```py
  from keras_cv_attention_models import convnext
  model = convnext.ConvNeXtBase(input_shape=(None, None, 3), num_classes=0)

  print(model(np.ones([1, 223, 123, 3])).shape)
  # (1, 6, 3, 1024)
  print(model(np.ones([1, 32, 526, 3])).shape)
  # (1, 1, 16, 1024)
  ```
## Verification with PyTorch version
  ```py
  inputs = np.random.uniform(size=(1, 224, 224, 3)).astype("float32")

  """ PyTorch convnext_base """
  sys.path.append('../ConvNeXt/')
  import torch
  from models import convnext as torch_convnext
  torch_model = torch_convnext.convnext_base(pretrained=True)
  _ = torch_model.eval()
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()

  """ Keras ConvNeXtBase """
  from keras_cv_attention_models import convnext
  mm = convnext.ConvNeXtBase(pretrained="imagenet", classifier_activation=None)
  keras_out = mm(inputs).numpy()

  """ Verification """
  print(f"{np.allclose(torch_out, keras_out, atol=1e-3) = }")
  # np.allclose(torch_out, keras_out, atol=1e-3) = True
  ```
***
