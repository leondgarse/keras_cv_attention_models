# ___Keras InceptionNeXt___
***

## Summary
  - Keras implementation of [Github sail-sg/inceptionnext](https://github.com/sail-sg/inceptionnext). Paper [PDF 2303.16900 InceptionNeXt: When Inception Meets ConvNeXt](https://arxiv.org/pdf/2303.16900.pdf).
  - Model weights ported from official publication.
***

## Models
  | Model              | Params | FLOP s | Input | Top1 Acc | Download |
  | ------------------ | ------ | ------ | ----- | -------- | -------- |
  | InceptionNeXtTiny  | 28.05M | 4.21G  | 224   | 82.3     | [inceptionnext_tiny_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/inceptionnext/inceptionnext_tiny_imagenet.h5) |
  | InceptionNeXtSmall | 49.37M | 8.39G  | 224   | 83.5     | [inceptionnext_small_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/inceptionnext/inceptionnext_small_imagenet.h5) |
  | InceptionNeXtBase  | 86.67M | 14.88G | 224   | 84.0     | [inceptionnext_base_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/inceptionnext/inceptionnext_base_224_imagenet.h5) |
  |                    | 86.67M | 43.73G | 384   | 85.2     | [inceptionnext_base_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/inceptionnext/inceptionnext_base_384_imagenet.h5) |

## Usage
  ```py
  from keras_cv_attention_models import inceptionnext

  # Will download and load pretrained imagenet weights.
  model = inceptionnext.InceptionNeXtTiny(pretrained="imagenet")

  # Run prediction
  from skimage.data import chelsea # Chelsea the cat
  preds = model(model.preprocess_input(chelsea()))
  print(model.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.8221698), ('n02123159', 'tiger_cat', 0.019049658), ...]
  ```
  **Use dynamic input resolution** by set `input_shape=(None, None, 3)`.
  ```py
  from keras_cv_attention_models import inceptionnext
  model = inceptionnext.InceptionNeXtTiny(input_shape=(None, None, 3), num_classes=0)
  # >>>> Load pretrained from: ~/.keras/models/inceptionnext_tiny_imagenet.h5
  print(model.output_shape)
  # (None, None, None, 768)

  print(model(np.ones([1, 223, 123, 3])).shape)
  # (1, 6, 3, 768)
  print(model(np.ones([1, 32, 526, 3])).shape)
  # (1, 1, 16, 768)
  ```
  **Using PyTorch backend** by set `KECAM_BACKEND='torch'` environment variable.
  ```py
  os.environ['KECAM_BACKEND'] = 'torch'

  from keras_cv_attention_models import inceptionnext
  model = inceptionnext.InceptionNeXtTiny(input_shape=(None, None, 3), num_classes=0)
  # >>>> Using PyTorch backend
  # >>>> Aligned input_shape: [3, None, None]
  # >>>> Load pretrained from: ~/.keras/models/inceptionnext_tiny_imagenet.h5
  print(model.output_shape)
  # (None, 768, None, None)

  import torch
  print(model(torch.ones([1, 3, 223, 123])).shape)
  # (1, 768, 6, 3 )
  print(model(torch.ones([1, 3, 32, 526])).shape)
  # (1, 768, 1, 16)
  ```  
## Verification with PyTorch version
  ```py
  """ PyTorch inceptionnext_tiny """
  sys.path.append('../inceptionnext/')
  sys.path.append('../pytorch-image-models/')  # Needs timm
  import torch
  from models import inceptionnext as inceptionnext_torch

  torch_model = inceptionnext_torch.inceptionnext_tiny(pretrained=True)
  _ = torch_model.eval()

  """ Keras InceptionNeXtTiny """
  from keras_cv_attention_models import inceptionnext
  mm = inceptionnext.InceptionNeXtTiny(pretrained="imagenet", classifier_activation=None)

  """ Verification """
  inputs = np.random.uniform(size=(1, *mm.input_shape[1:3], 3)).astype("float32")
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()
  keras_out = mm(inputs).numpy()
  print(f"{np.allclose(torch_out, keras_out, atol=5e-5) = }")
  # np.allclose(torch_out, keras_out, atol=5e-5) = True
  ```
