# ___Keras GCViT___
***

## Summary
  - Keras implementation of [Github NVlabs/GCVit](https://github.com/NVlabs/GCVit). Paper [PDF 2206.09959 Global Context Vision Transformers](https://arxiv.org/pdf/2206.09959.pdf).
  - Model weights ported from official publication.
***

## Models
  | Model        | Params | FLOPs | Input | Top1 Acc | Download |
  | ------------ | ------ | ----- | ----- | -------- | -------- |
  | GCViT_XXTiny | 12.0M  | 2.15G | 224   | 79.8     | [gcvit_xx_tiny_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gcvit/gcvit_xx_tiny_224_imagenet.h5) |
  | GCViT_XTiny  | 20.0M  | 2.96G | 224   | 82.04    | [gcvit_x_tiny_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gcvit/gcvit_x_tiny_224_imagenet.h5) |
  | GCViT_Tiny   | 28.2M  | 4.83G | 224   | 83.4     | [gcvit_tiny_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gcvit/gcvit_tiny_224_imagenet.h5) |
  | GCViT_Small  | 51.1M  | 8.63G | 224   | 83.95    | [gcvit_small_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gcvit/gcvit_small_224_imagenet.h5) |
  | GCViT_Base   | 90.3M  | 14.9G | 224   | 84.47    | [gcvit_base_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gcvit/gcvit_base_224_imagenet.h5) |

## Usage
  ```py
  from keras_cv_attention_models import gcvit

  # Will download and load pretrained imagenet weights.
  mm = gcvit.GCViT_Tiny(pretrained="imagenet")

  # Run prediction
  import tensorflow as tf
  from tensorflow import keras
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.74019605), ('n02123045', 'tabby', 0.07751605), ...]
  ```
  **Change input resolution**. `input_shape` has to be divisible by a combination of strides and `window_ratios`, for default `window_ratios=[8, 4, 1, 1]`, it should be divisible by **32**.
  ```py
  from keras_cv_attention_models import gcvit
  mm = gcvit.GCViT_Tiny(input_shape=(128, 192, 3), pretrained="imagenet")
  # >>>> Load pretrained from: ~/.keras/models/gcvit_tiny_224_imagenet.h5

  # Run prediction
  from skimage.data import chelsea
  preds = mm(mm.preprocess_input(chelsea()))
  print(mm.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.5739756), ('n02123045', 'tabby', 0.03262589), ...]
  ```
## Verification with PyTorch version
  ```py
  """ PyTorch gc_vit_tiny """
  sys.path.append('../GCVit/')
  sys.path.append('../pytorch-image-models/')  # Needs timm
  import torch
  from models import gc_vit

  torch_model = gc_vit.gc_vit_tiny()
  ss = torch.load('gcvit_tiny_best_1k.pth.tar', map_location=torch.device('cpu'))
  torch_model.load_state_dict(ss['state_dict'])
  _ = torch_model.eval()

  """ Keras GCViT_Tiny """
  from keras_cv_attention_models import gcvit
  mm = gcvit.GCViT_Tiny(pretrained="imagenet", classifier_activation=None)

  """ Verification """
  inputs = np.random.uniform(size=(1, *mm.input_shape[1:3], 3)).astype("float32")
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()
  keras_out = mm(inputs).numpy()
  print(f"{np.allclose(torch_out, keras_out, atol=5e-2) = }")
  # np.allclose(torch_out, keras_out, atol=5e-2) = True
  ```
