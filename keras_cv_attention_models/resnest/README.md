# ___Keras ResNeSt___
***

## Summary
  - Keras implementation of [ResNeSt](https://github.com/zhanghang1989/ResNeSt). Paper [PDF 2004.08955 ResNeSt: Split-Attention Networks](https://arxiv.org/pdf/2004.08955.pdf).
  - Model weights reloaded from official publication.
***

## Models
  | Model          | Params | FLOPs  | Input | Top1 Acc | Download |
  | -------------- | ------ | ------ | ----- | -------- | -------- |
  | resnest50      | 28M    | 5.38G  | 224   | 81.03    | [resnest50.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnest/resnest50_imagenet.h5) |
  | resnest101     | 49M    | 13.33G | 256   | 82.83    | [resnest101.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnest/resnest101_imagenet.h5) |
  | resnest200     | 71M    | 35.55G | 320   | 83.84    | [resnest200.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnest/resnest200_imagenet.h5) |
  | resnest269     | 111M   | 77.42G | 416   | 84.54    | [resnest269.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnest/resnest269_imagenet.h5) |
## Usage
  ```py
  from keras_cv_attention_models import resnest

  # Will download and load pretrained imagenet weights.
  mm = resnest.ResNest50(pretrained="imagenet")

  # Run prediction
  import tensorflow as tf
  from tensorflow import keras
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.7793046),
  #  ('n02123159', 'tiger_cat', 0.028313603),
  #  ('n04209239', 'tabby', 0.02322878),
  #  ('n02127052', 'lynx', 0.0036637571),
  #  ('n03085013', 'computer_keyboard', 0.0008628946)]
  ```
  **Use dynamic input resolution**
  ```py
  from keras_cv_attention_models import resnest
  mm = resnest.ResNest50(input_shape=(None, None, 3), num_classes=0)

  print(mm(np.ones([1, 224, 224, 3])).shape)
  # (1, 7, 7, 2048)
  print(mm(np.ones([1, 512, 512, 3])).shape)
  # (1, 16, 16, 2048)

  mm.save("../models/resnest50_dynamic_notop.h5")
  ```
## Verification with PyTorch version
  ```py
  """ PyTorch resnest50 """
  import torch
  sys.path.append("../")
  from ResNeSt.resnest.torch import resnest as torch_resnest

  torch_model = torch_resnest.resnest50(pretrained=True)
  torch_model.eval()

  """ Keras ResNest50 """
  from keras_cv_attention_models import resnest
  mm = resnest.ResNest50(pretrained="imagenet", classifier_activation=None)

  """ Verification """
  inputs = np.random.uniform(size=(1, *mm.input_shape[1:3], 3)).astype("float32")
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()
  keras_out = mm(inputs).numpy()
  print(f"{np.allclose(torch_out, keras_out, atol=1e-4) = }")
  # np.allclose(torch_out, keras_out, atol=1e-4) = True
  ```
***
