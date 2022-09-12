# ___Keras HorNet___
***

## Summary
  - Keras implementation of [Github raoyongming/hornet](https://github.com/raoyongming/hornet). Paper [PDF 2207.14284 HorNet: Efficient High-Order Spatial Interactions with Recursive Gated Convolutions](https://arxiv.org/pdf/2207.14284.pdf).
  - Model weights ported from official publication.

  ![hornet](https://user-images.githubusercontent.com/5744524/189527601-4aae9277-1211-47cd-9042-f8fe193284d9.PNG)
***

## Models
  | Model         | Params | FLOPs  | Input | Top1 Acc | Download |
  | ------------- | ------ | ------ | ----- | -------- | -------- |
  | HorNetTiny    | 22.4M  | 4.01G  | 224   | 82.8     | [hornet_tiny_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/hornet/hornet_tiny_224_imagenet.h5) |
  | HorNetTinyGF  | 23.0M  | 3.94G  | 224   | 83.0     | [hornet_tiny_gf_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/hornet/hornet_tiny_gf_224_imagenet.h5) |
  | HorNetSmall   | 49.5M  | 8.87G  | 224   | 83.8     | [hornet_small_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/hornet/hornet_small_224_imagenet.h5) |
  | HorNetSmallGF | 50.4M  | 8.77G  | 224   | 84.0     | [hornet_small_gf_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/hornet/hornet_small_gf_224_imagenet.h5) |
  | HorNetBase    | 87.3M  | 15.65G | 224   | 84.2     | [hornet_base_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/hornet/hornet_base_224_imagenet.h5) |
  | HorNetBaseGF  | 88.4M  | 15.51G | 224   | 84.3     | [hornet_base_gf_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/hornet/hornet_base_gf_224_imagenet.h5) |
  | HorNetLarge   | 194.5M | 34.91G | 224   | 86.8     | [hornet_large_224_imagenet22k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/hornet/hornet_large_224_imagenet22k.h5) |
  | HorNetLargeGF | 196.3M | 34.72G | 224   | 87.0     | [hornet_large_gf_224_imagenet22k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/hornet/hornet_large_gf_224_imagenet22k.h5) |
  | HorNetLargeGF | 201.8M | 102.0G | 384   | 87.7     | [hornet_large_gf_384_imagenet22k.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/hornet/hornet_large_gf_384_imagenet22k.h5) |

## Usage
  ```py
  from keras_cv_attention_models import hornet

  # Will download and load pretrained imagenet weights.
  mm = hornet.HorNetTiny(pretrained="imagenet")

  # Run prediction
  import tensorflow as tf
  from tensorflow import keras
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.73596513), ('n02123045', 'tabby', 0.091754995), ...]
  ```
  **Change input resolution**.
  ```py
  from keras_cv_attention_models import hornet
  mm = hornet.HorNetTinyGF(input_shape=(117, 393, 3), pretrained="imagenet")
  # >>>> Load pretrained from: ~/.keras/models/hornet_tiny_gf_224_imagenet.h5
  # ...
  # >>>> Reload mismatched weights: 224 -> (117, 393)
  # >>>> Reload layer: stack3_block1_gnconv_gf_complex_dense
  # ...
  # >>>> Reload layer: stack4_block2_gnconv_gf_complex_dense

  # Run prediction
  from skimage.data import chelsea
  preds = mm(mm.preprocess_input(chelsea()))
  print(mm.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.6855306), ('n02123045', 'tabby', 0.18513484), ...]
  ```
  **Dynamic input_shape for non-GF models**
  ```py
  from keras_cv_attention_models import hornet
  mm = hornet.HorNetTiny(input_shape=(None, None, 3), pretrained="imagenet", num_classes=0)

  print(mm(tf.ones([1, 122, 237, 3])).shape)
  # (1, 3, 7, 512)
  print(mm(tf.ones([1, 222, 137, 3])).shape)
  # (1, 6, 4, 512)
  ```
## Verification with PyTorch version
  ```py
  """ PyTorch hornet_tiny_gf """
  sys.path.append('../HorNet/')
  sys.path.append('../pytorch-image-models/')  # Needs timm
  import torch
  import hornet as torch_hornet

  torch_model = torch_hornet.hornet_tiny_gf()
  ss = torch.load('hornet_tiny_gf.pth', map_location=torch.device('cpu'))
  torch_model.load_state_dict(ss['model'])
  _ = torch_model.eval()

  """ Keras HorNetTinyGF """
  from keras_cv_attention_models import hornet
  mm = hornet.HorNetTinyGF(pretrained="imagenet", classifier_activation=None)

  """ Verification """
  inputs = np.random.uniform(size=(1, *mm.input_shape[1:3], 3)).astype("float32")
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()
  keras_out = mm(inputs).numpy()
  print(f"{np.allclose(torch_out, keras_out, atol=5e-3) = }")
  # np.allclose(torch_out, keras_out, atol=5e-3) = True
  ```
