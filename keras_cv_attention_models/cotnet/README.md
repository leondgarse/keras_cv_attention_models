# Keras_CoTNet
***

## Summary
  - [PDF 2107.12292 Contextual Transformer Networks for Visual Recognition](https://arxiv.org/pdf/2107.12292.pdf)
  - [Github JDAI-CV/CoTNet](https://github.com/JDAI-CV/CoTNet)
***

## Models
  | Model          | Params | Image resolution | FLOPs | Top1 Acc | Download            |
  | -------------- |:------:| ---------------- | ----- |:--------:| ------------------- |
  | CoTNet-50      | 22.2M  | 224              | 3.3   |   81.3   | [cotnet50_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cotnet/cotnet50_224.h5) |
  | CoTNeXt-50     | 30.1M  | 224              | 4.3   |   82.1   |  |
  | SE-CoTNetD-50  | 23.1M  | 224              | 4.1   |   81.6   | [se_cotnetd50_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cotnet/se_cotnetd50_224.h5) |
  | CoTNet-101     | 38.3M  | 224              | 6.1   |   82.8   | [cotnet101_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cotnet/cotnet101_224.h5) |
  | CoTNeXt-101    | 53.4M  | 224              | 8.2   |   83.2   |  |
  | SE-CoTNetD-101 | 40.9M  | 224              | 8.5   |   83.2   | [se_cotnetd101_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cotnet/se_cotnetd101_224.h5) |
  | SE-CoTNetD-152 | 55.8M  | 224              | 17.0  |   84.0   | [se_cotnetd152_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cotnet/se_cotnetd152_224.h5) |
  | SE-CoTNetD-152 | 55.8M  | 320              | 26.5  |   84.6   | [se_cotnetd152_320.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cotnet/se_cotnetd152_320.h5) |
## Usage
  ```py
  from keras_cv_attention_models import cotnet

  # Will download and load pretrained imagenet weights.
  mm = cotnet.CotNet50(pretrained="imagenet")

  # Run prediction
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.74894345),
  #  ('n02123159', 'tiger_cat', 0.06659871),
  #  ('n02123045', 'tabby', 0.04352202),
  #  ('n02127052', 'lynx', 0.004080989),
  #  ('n03720891', 'maraca', 0.002005524)]
  ```
  **Change input resolution**
  ```py
  from keras_cv_attention_models import cotnet
  mm = cotnet.SECotNetD101(input_shape=(480, 480, 3), pretrained="imagenet")

  # Run prediction on Chelsea with (480, 480) resolution
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.75343966), ('n02123159', 'tiger_cat', 0.09504254), ...]

  print(f"{cotnet.SECotNetD101(input_shape=(224, 224, 3), num_classes=0).output_shape = }")
  # cotnet.SECotNetD101(input_shape=(224, 224, 3), num_classes=0).output_shape = (None, 7, 7, 2048)
  print(f"{cotnet.SECotNetD101(input_shape=(480, 480, 3), num_classes=0).output_shape = }")
  # cotnet.SECotNetD101(input_shape=(480, 480, 3), num_classes=0).output_shape = (None, 15, 15, 2048)
  ```
## Verification with PyTorch version
  ```py
  """ PyTorch SECotNetD101 """
  import torch
  import argparse

  sys.path.append("../CoTNet")
  from train import setup_env, setup_model

  parser = argparse.ArgumentParser()
  parser.add_argument('--folder', dest='folder', type=str, default=None)
  args = parser.parse_args("--folder ../CoTNet/cot_experiments/SE-CoTNetD-101_350epoch/".split(' '))
  setup_env(args)
  torch_model, data_config = setup_model()
  torch_model.eval()
  weight = torch.load('../models/se_cotnetd_101.pth.tar', map_location=torch.device('cpu'))
  torch_model.load_state_dict(weight)

  """ Keras SECotNetD101 """
  from keras_cv_attention_models import cotnet
  mm = cotnet.SECotNetD101(classifier_activation=None)

  """ Verification """
  inputs = np.random.uniform(size=(1, *mm.input_shape[1:3], 3)).astype("float32")
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()
  keras_out = mm(inputs).numpy()
  print(f"{np.allclose(torch_out, keras_out, atol=1e-4) = }")
  # np.allclose(torch_out, keras_out, atol=1e-4) = True
  ```
***
