# ___Keras PyTorch Backend___
***

## Summary
  - Experimental implementation for using PyTorch as Keras backend.
  - Currently supports most recognition and detection models except coat / halonet / hornet / nat / nfnets / volo.
***

## Usage
- **Set os environment `export KECAM_BACKEND='torch'` to enable this PyTorch backend**.
- **Create model and run predict**.
  - Will load same `h5` weights as TF one if available.
  - Note: `input_shape` will auto fit image data format. Given `input_shape=(224, 224, 3)`, will set to `(3, 224, 224)` if `channels_first`.
  - Note: model is defaultly set to `eval` mode.
  ```py
  from keras_cv_attention_models import res_mlp
  mm = res_mlp.ResMLP12()
  # >>>> Load pretrained from: ~/.keras/models/resmlp12_imagenet.h5
  print(f"{mm.input_shape = }")
  # mm.input_shape = [None, 3, 224, 224]

  import torch
  print(f"{isinstance(mm, torch.nn.Module) = }")
  # isinstance(mm, torch.nn.Module) = True

  # Run prediction
  from skimage.data import chelsea # Chelsea the cat
  print(mm.decode_predictions(mm(mm.preprocess_input(chelsea())))[0])
  # [('n02124075', 'Egyptian_cat', 0.86188155), ('n02123159', 'tiger_cat', 0.05125639), ...]
  ```
- Model is a typical `torch.nn.Module`, so PyTorch training process can be applied. Also exporting `onnx` / `pth` supported.
  ```py
  from keras_cv_attention_models import wave_mlp
  mm = wave_mlp.WaveMLP_T(input_shape=(3, 224, 224))
  # >>>> Load pretrained from: ~/.keras/models/wavemlp_t_imagenet.h5

  mm.export_onnx()
  # Exported onnx: wavemlp_t.onnx

  mm.export_pth()
  # Exported pth: wavemlp_t.pth
  ```
- `load_weights` / `save_weights` will load / save `h5` weights, which can be used for converting between TF and PyTorch. Currently it's only weights without model structure supported.
  ```py
  from keras_cv_attention_models import gated_mlp
  mm = gated_mlp.GMLPS16(input_shape=(3, 224, 224))
  # >>>> Load pretrained from: ~/.keras/models/gmlp_s16_imagenet.h5

  mm.save_weights("foo.h5")
  ```
  Then unset `KECAM_BACKEND` or `export KECAM_BACKEND=tensorflow` for using typical TF backend. Note input_shape is channels last.
  ```py
  from keras_cv_attention_models import gated_mlp
  mm = gated_mlp.GMLPS16(input_shape=(224, 224, 3), pretrained=None)  # channels_last input_shape
  mm.load_weights('foo.h5', by_name=True)  # Reload weights from PyTorch backend

  # Run prediction
  from skimage.data import chelsea # Chelsea the cat
  mm.decode_predictions(mm(mm.preprocess_input(chelsea())))
  # [('n02124075', 'Egyptian_cat', 0.8495876), ('n02123159', 'tiger_cat', 0.029945023), ...]
  ```
- **Create custom PyTorch model using keras API**
  ```py
  from keras_cv_attention_models.pytorch_backend import layers, models
  inputs = layers.Input([3, 224, 224])
  pre = layers.Conv2D(32, kernel_size=3, padding="SAME", name="deep_pre_conv")(inputs)
  deep_1 = layers.Conv2D(32, kernel_size=3, padding="SAME", name="deep_1_1_conv")(pre)
  deep_1 = layers.Conv2D(32, kernel_size=3, padding="SAME", name="deep_1_2_conv")(deep_1)
  deep_2 = layers.Conv2D(32, kernel_size=3, padding="SAME", name="deep_2_conv")(pre)
  deep = layers.Add(name="deep_add")([deep_1, deep_2])
  short = layers.Conv2D(32, kernel_size=3, padding="SAME", name="short_conv")(inputs)
  outputs = layers.Add(name="outputs")([short, deep])
  mm = models.Model(inputs, outputs)
  mm.summary()

  import torch
  print(mm(torch.ones([1, 3, 224, 224])).shape)
  # torch.Size([1, 32, 224, 224])

  # Save load test
  mm.save_weights("aa.h5")
  mm.load_weights('aa.h5')
  ```
***
