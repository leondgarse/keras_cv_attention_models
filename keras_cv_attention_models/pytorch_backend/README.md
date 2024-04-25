# ___Keras PyTorch Backend___
***

## Summary
  - Experimental implementation for using PyTorch as Keras backend.
  - Currently supports most recognition and detection models except hornet*gf / nfnets / volo. For detection models, using `torchvision.ops.nms` while running prediction.
***

## Basic Usage
- **Set os environment `export KECAM_BACKEND='torch'` to enable this PyTorch backend**.
  ```py
  import os
  os.environ['KECAM_BACKEND'] = 'torch'

  import numpy as np
  from keras_cv_attention_models.backend import models, layers, functional, losses
  xx, yy = np.random.uniform(size=[64, 3, 32, 32]).astype('float32'), np.random.uniform(0, 10, size=[64]).astype('int32')
  mm = models.Sequential([layers.Input([3, 32, 32]), layers.Conv2D(128), layers.GlobalAveragePooling2D(), layers.Dense(10)])
  mm.compile(optimizer='SGD', loss=losses.sparse_categorical_crossentropy, metrics='acc')
  mm.fit(xx, yy, batch_size=4, epochs=3)
  ```
- **Create model and run predict**.
  - Will load same `h5` weights as TF one if available.
  - Note: `input_shape` will auto fit image data format. Given `input_shape=(224, 224, 3)` or `input_shape=(3, 224, 224)`, will set to `(3, 224, 224)` if `channels_first`.
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

  # Export typical PyTorch onnx / pth
  import torch
  torch.onnx.export(mm, torch.randn(1, 3, *mm.input_shape[2:]), mm.name + ".onnx")

  # Or by export_onnx
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
## Create custom PyTorch model using keras API
  ```py
  from keras_cv_attention_models.pytorch_backend import layers, models
  inputs = layers.Input([3, 224, 224])
  pre = layers.Conv2D(32, kernel_size=3, padding="same", name="deep_pre_conv")(inputs)
  deep_1 = layers.Conv2D(32, kernel_size=3, padding="same", name="deep_1_1_conv")(pre)
  deep_1 = layers.Conv2D(32, kernel_size=3, padding="same", name="deep_1_2_conv")(deep_1)
  deep_2 = layers.Conv2D(32, kernel_size=3, padding="same", name="deep_2_conv")(pre)
  deep = layers.Add(name="deep_add")([deep_1, deep_2])
  short = layers.Conv2D(32, kernel_size=3, padding="same", name="short_conv")(inputs)
  outputs = layers.Add(name="outputs")([short, deep])
  outputs = layers.GlobalAveragePooling2D()(outputs)
  outputs = layers.Dense(10)(outputs)
  mm = models.Model(inputs, outputs)
  mm.summary()

  import torch
  print(mm(torch.ones([1, 3, 224, 224])).shape)
  # torch.Size([1, 10])

  # Save load test
  mm.save_weights("aa.h5")
  mm.load_weights('aa.h5')
  ```
  **Sequential model**
  ```py
  from keras_cv_attention_models.pytorch_backend import layers, models, functional
  mm = models.Sequential([
      layers.Input([3, 32, 32]),
      layers.Conv2D(32, 3, 2, padding='same'),
      layers.GlobalAveragePooling2D(),
      layers.Dense(10),
      functional.softmax,  # Can also be an functional callable
  ])
  mm.summary()
  ```
## Simple compile fit training
  - It can be either typical PyTorch training process or a simple version of `compile` + `fit`. **Note: will check if any kwargs in `["loss", "optmizer", "metrics"]` provided, deciding whether calling `self.train_compile` for training, or the default `torch.nn.Module.compile`**.
  ```py
  import os
  os.environ['KECAM_BACKEND'] = 'torch'

  from keras_cv_attention_models.imagenet import data, callbacks
  input_shape = (32, 32, 3)
  batch_size = 16
  train_dataset, test_dataset, total_images, num_classes, steps_per_epoch = data.init_dataset(
      'cifar10', input_shape=input_shape, batch_size=batch_size,
  )

  import torch
  from kecam import mobilenetv3
  mm = mobilenetv3.MobileNetV3Large100(input_shape=input_shape, num_classes=num_classes, classifier_activation=None, pretrained=None)
  if torch.cuda.is_available():
      _ = mm.to("cuda")
  if hasattr(torch, "compile") and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 6:
      mm = torch.compile(mm)

  """ Simple compile + fit """
  mm.compile(optimizer="AdamW", metrics='acc')
  cur_callbacks = [
      callbacks.MyHistory(initial_file='checkpoints/test_hist.json'),
      callbacks.MyCheckpoint('test', save_path="checkpoints"),
      callbacks.CosineLrScheduler(lr_base=0.001, first_restart_step=10, steps_per_epoch=len(train_dataset)),
  ]
  mm.fit(train_dataset, epochs=10, validation_data=test_dataset, callbacks=cur_callbacks)
  ```
  Or use typical PyTorch training process
  ```py
  optimizer = torch.optim.AdamW(mm.parameters())
  device = next(mm.parameters()).device
  device_type = device.type

  if device_type == "cuda":
      scaler = torch.cuda.amp.GradScaler(enabled=True)
      global_context = torch.amp.autocast(device_type=device_type, dtype=torch.float16)
  else:
      from contextlib import nullcontext

      scaler = torch.cuda.amp.GradScaler(enabled=False)
      global_context = nullcontext()

  for epoch in range(10):
      data_gen = train_dataset.as_numpy_iterator()
      mm.train()
      for batch, (xx, yy) in enumerate(data_gen):
          xx = torch.from_numpy(xx).permute(0, 3, 1, 2)
          yy = torch.from_numpy(yy)
          if device_type == "cuda":
              xx = xx.pin_memory().to(device, non_blocking=True)
              yy = yy.pin_memory().to(device, non_blocking=True)

          optimizer.zero_grad()
          with global_context:
              out = mm(xx)
              loss = torch.functional.F.cross_entropy(out, yy)

          scaler.scale(loss).backward()
          scaler.unscale_(optimizer)
          scaler.step(optimizer)
          scaler.update()
          print(">>>> Epoch {}, batch: {}, loss: {:.4f}".format(epoch, batch, loss.item()))
  ```
***
