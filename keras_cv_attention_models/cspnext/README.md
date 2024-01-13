# ___Keras CSPNeXt___
***

## Summary
  - CSPNeXt is the backbone from article: [PDF 2212.07784 RTMDet: An Empirical Study of Designing Real-Time Object Detectors](https://arxiv.org/abs/2212.07784).
  - Model weights ported from [Github open-mmlab/mmdetection/rtmdet](https://github.com/open-mmlab/mmdetection/tree/main/configs/rtmdet#classification).
***

## Models
  | Model         | Params | FLOPs | Input | Top1 Acc | Download |
  | ------------- | ------ | ----- | ----- | -------- | -------- |
  | CSPNeXtTiny   | 2.73M  | 0.34G | 224   | 69.44    | [cspnext_tiny_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cspnext/cspnext_tiny_imagenet.h5) |
  | CSPNeXtSmall  | 4.89M  | 0.66G | 224   | 74.41    | [cspnext_small_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cspnext/cspnext_small_imagenet.h5) |
  | CSPNeXtMedium | 13.05M | 1.92G | 224   | 79.27    | [cspnext_medium_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cspnext/cspnext_medium_imagenet.h5) |
  | CSPNeXtLarge  | 27.16M | 4.19G | 224   | 81.30    | [cspnext_large_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cspnext/cspnext_large_imagenet.h5) |
  | CSPNeXtXLarge | 48.85M | 7.75G | 224   | 82.10    | [cspnext_xlarge_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cspnext/cspnext_xlarge_imagenet.h5) |

## Usage
  ```py
  from keras_cv_attention_models import cspnext, test_images
  mm = cspnext.CSPNeXtTiny()

  # Run prediction
  preds = mm(mm.preprocess_input(test_images.cat()))
  print(mm.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.46106383), ('n02123045', 'tabby', 0.19603978), ...]
  ```
  **Use dynamic input resolution** by set `input_shape=(None, None, 3)`.
  ```py
  from keras_cv_attention_models import cspnext
  model = cspnext.CSPNeXtTiny(input_shape=(None, None, 3), num_classes=0)

  print(model(np.ones([1, 223, 123, 3])).shape)
  # (1, 7, 4, 384)
  print(model(np.ones([1, 32, 526, 3])).shape)
  # (1, 1, 17, 384)
  ```
  **Using PyTorch backend** by set `KECAM_BACKEND='torch'` environment variable.
  ```py
  os.environ['KECAM_BACKEND'] = 'torch'
  from keras_cv_attention_models import cspnext, test_images
  mm = cspnext.CSPNeXtSmall(input_shape=(219, 112, 3))
  # >>>> Using PyTorch backend
  # >>>> Load pretrained from: ~/.keras/models/cspnext_small_imagenet.h5

  # Run prediction
  preds = mm(mm.preprocess_input(test_images.cat()))
  print(mm.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.7909507), ('n02123045', 'tabby', 0.038315363), ...]
  ```
## Verification with PyTorch version
  ```py
  inputs = np.random.uniform(size=(1, 224, 224, 3)).astype("float32")

  """ PyTorch CSPNeXt """
  from mmdet import models
  torch_model = models.backbones.CSPNeXt()
  import torch
  ss = torch.load('cspnext-l_8xb256-rsb-a1-600e_in1k-6a760974.pth')
  ss = {kk.replace('backbone.', ''): vv for kk, vv in ss['state_dict'].items() if kk.startswith('backbone.')}
  torch_model.load_state_dict(ss)
  _ = torch_model.eval()
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2))[-1].permute([0, 2, 3, 1]).detach().numpy()

  """ Keras CSPNeXtLarge """
  from keras_cv_attention_models import cspnext
  mm = cspnext.CSPNeXtLarge(pretrained="imagenet", num_classes=0)  # Exclude header
  keras_out = mm(inputs).numpy()

  """ Verification """
  print(f"{np.allclose(torch_out, keras_out, atol=1e-4) = }")
  # np.allclose(torch_out, keras_out, atol=1e-4) = True
  ```
***
