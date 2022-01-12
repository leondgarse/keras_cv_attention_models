# ___Keras CovNeXt___
***

## Summary
- CoAtNet article: [PDF 2201.03545 A ConvNet for the 2020s](https://arxiv.org/pdf/2201.03545.pdf).
- Model weights reloaded from [Github facebookresearch/ConvNeXt](https://github.com/facebookresearch/ConvNeXt).
***

## Models
  | Model               | Params | Image resolution | Top1 Acc | Model link |
  | ------------------- | ------ | ---------------- | -------- | ---------- |
  | ConvNeXtTiny        | 28M    | 224              | 82.1     | [convnext_tiny_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_tiny_imagenet.h5) |
  | ConvNeXtSmall       | 50M    | 224              | 83.1     | [convnext_small_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_small_imagenet.h5) |
  | ConvNeXtBase        | 89M    | 224              | 83.8     | [convnext_base_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_base_imagenet.h5) |
  | ConvNeXtBase        | 89M    | 384              | 85.1     |            |
  | - ImageNet21k-ft1k  | 89M    | 224              | 85.8     | [convnext_base_imagenet-ft1k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_base_imagenet-ft1k.h5) |
  | - ImageNet21k-ft1k  | 89M    | 384              | 86.8     |            |
  | ConvNeXtLarge       | 198M   | 224              | 84.3     | [convnext_large_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_large_imagenet.h5) |
  | ConvNeXtLarge       | 198M   | 384              | 85.5     |            |
  | - ImageNet21k-ft1k  | 198M   | 224              | 86.6     | [convnext_large_imagenet-ft1k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_large_imagenet-ft1k.h5) |
  | - ImageNet21k-ft1k  | 198M   | 384              | 87.5     |            |
  | ConvNeXtXLarge, 21k | 350M   | 224              | 87.0     | [convnext_xlarge_imagenet-ft1k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/convnext/convnext_xlarge_imagenet-ft1k.h5) |
  | ConvNeXtXLarge, 21k | 350M   | 384              | 87.8     |            |
## Usage
  ```py
  from keras_cv_attention_models import convnext, test_images
  mm = convnext.ConvNeXtBase()
  # >>>> Load pretrained from: ~/.keras/models/convnext_base_imagenet.h5
  preds = mm(mm.preprocess_input(test_images.cat()))
  print(mm.decode_predictions(preds)[0])
  # [('n02123159', 'tiger_cat', 0.9018271), ('n02123045', 'tabby', 0.019625964), ...]
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
