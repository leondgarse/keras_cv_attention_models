# ___Keras CoTNet___
***

## Summary
  - [PDF 2107.12292 Contextual Transformer Networks for Visual Recognition](https://arxiv.org/pdf/2107.12292.pdf)
  - [Github JDAI-CV/CoTNet](https://github.com/JDAI-CV/CoTNet)
***

## Models
  | Model        | Params | FLOPs  | Input | Top1 Acc | Download            |
  | ------------ |:------:| ------ | ----- |:--------:| ------------------- |
  | CotNet50     | 22.2M  | 3.25G  | 224   |   81.3   | [cotnet50_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cotnet/cotnet50_224_imagenet.h5) |
  | CotNeXt50    | 30.1M  | 4.3G   | 224   |   82.1   |  |
  | CotNetSE50D  | 23.1M  | 4.05G  | 224   |   81.6   | [cotnet_se50d_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cotnet/cotnet_se50d_224_imagenet.h5) |
  | CotNet101    | 38.3M  | 6.07G  | 224   |   82.8   | [cotnet101_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cotnet/cotnet101_224_imagenet.h5) |
  | CotNeXt101   | 53.4M  | 8.2G   | 224   |   83.2   |  |
  | CotNetSE101D | 40.9M  | 8.44G  | 224   |   83.2   | [cotnet_se101d_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cotnet/cotnet_se101d_224_imagenet.h5) |
  | CotNetSE152D | 55.8M  | 12.22G | 224   |   84.0   | [cotnet_se152d_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cotnet/cotnet_se152d_224_imagenet.h5) |
  | CotNetSE152D | 55.8M  | 24.92G | 320   |   84.6   | [cotnet_se152d_320_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cotnet/cotnet_se152d_320_imagenet.h5) |
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
  mm = cotnet.CotNetSE101D(input_shape=(480, 480, 3), pretrained="imagenet")

  # Run prediction on Chelsea with (480, 480) resolution
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.75343966), ('n02123159', 'tiger_cat', 0.09504254), ...]

  print(f"{cotnet.CotNetSE101D(input_shape=(224, 224, 3), num_classes=0).output_shape = }")
  # cotnet.CotNetSE101D(input_shape=(224, 224, 3), num_classes=0).output_shape = (None, 7, 7, 2048)
  print(f"{cotnet.CotNetSE101D(input_shape=(480, 480, 3), num_classes=0).output_shape = }")
  # cotnet.CotNetSE101D(input_shape=(480, 480, 3), num_classes=0).output_shape = (None, 15, 15, 2048)
  ```
## Verification with PyTorch version
  ```py
  """ PyTorch CotNetSE101D """
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

  """ Keras CotNetSE101D """
  from keras_cv_attention_models import cotnet
  mm = cotnet.CotNetSE101D(classifier_activation=None)

  """ Verification """
  inputs = np.random.uniform(size=(1, *mm.input_shape[1:3], 3)).astype("float32")
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()
  keras_out = mm(inputs).numpy()
  print(f"{np.allclose(torch_out, keras_out, atol=1e-4) = }")
  # np.allclose(torch_out, keras_out, atol=1e-4) = True
  ```
***
