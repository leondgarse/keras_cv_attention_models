# ___Keras CoaT___
***

## Summary
  - Coat article: [PDF 2104.06399 CoaT: Co-Scale Conv-Attentional Image Transformers](http://arxiv.org/abs/2104.06399).
  - [Github mlpc-ucsd/CoaT](https://github.com/mlpc-ucsd/CoaT).
***

## Models
  | Model         | Params | FLOPs | Input | Top1 Acc | Download |
  | ------------- | ------ | ----- | ----- | -------- | -------- |
  | CoaTLiteTiny  | 5.7M   | 1.60G | 224   | 77.5     | [coat_lite_tiny_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/coat/coat_lite_tiny_imagenet.h5) |
  | CoaTLiteMini  | 11M    | 2.00G | 224   | 79.1     | [coat_lite_mini_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/coat/coat_lite_mini_imagenet.h5) |
  | CoaTLiteSmall | 20M    | 3.97G | 224   | 81.9     | [coat_lite_small_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/coat/coat_lite_small_imagenet.h5) |
  | CoaTTiny      | 5.5M   | 4.33G | 224   | 78.3     | [coat_tiny_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/coat/coat_tiny_imagenet.h5) |
  | CoaTMini      | 10M    | 6.78G | 224   | 81.0     | [coat_mini_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/coat/coat_mini_imagenet.h5) |
## Usage
  ```py
  from keras_cv_attention_models import coat

  # Will download and load pretrained imagenet weights.
  mm = coat.CoaTLiteSmall(pretrained="imagenet")

  # Run prediction
  import tensorflow as tf
  from tensorflow import keras
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.7886666), ('n02123159', 'tiger_cat', 0.031680673), ...]
  ```
  **Change input resolution**
  ```py
  # Will download and load pretrained imagenet weights.
  mm = coat.CoaTLiteSmall(input_shape=(320, 320, 3), pretrained="imagenet")

  # Run predict again using (320, 320)
  import tensorflow as tf
  from tensorflow import keras
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.7967514), ('n02123159', 'tiger_cat', 0.04747868), ...]
  ```
  Specific `out_features` a list of number in `[0, 1, 2, 3]` for output of relative `serial block` output.
  ```py
  mm = coat.CoaTMini(pretrained="imagenet", classifier_activation=None, input_shape=(224, 224, 3), out_features=[1, 2, 3])
  print(mm.output_shape)
  # [(None, 784, 216), (None, 196, 216), (None, 49, 216)]
  ```
  Set `use_shared_cpe=False, use_shared_crpe=False` to disable using shared `ConvPositionalEncoding` and `ConvRelativePositionalEncoding` blocks. will have a better structure view using `netron` or other visualization tools. Note it's for checking model architecture only, keep input_shape `height == width` if set False.
  ```py
  mm = coat.CoaTMini(pretrained="imagenet", classifier_activation=None, input_shape=(224, 224, 3))
  mm.summary()
  # Total params: 14,828,940
  mm = coat.CoaTMini(pretrained="imagenet", classifier_activation=None, input_shape=(224, 224, 3), use_shared_cpe=False, use_shared_crpe=False)
  mm.summary()
  # Total params: 15,021,148
  ```
## Verification with PyTorch version
  ```py
  """ PyTorch coat_mini """
  import torch
  sys.path.append("../CoaT/src/models")
  import coat
  torch_model = coat.coat_mini()
  torch_model.eval()
  weight = torch.load('../models/coat_mini_40667eec.pth', map_location=torch.device('cpu'))
  torch_model.load_state_dict(weight['model'])

  input_shape = 224
  inputs = np.random.uniform(size=(1, input_shape, input_shape, 3)).astype("float32")
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()

  """ Keras coat_mini """
  from keras_cv_attention_models import coat
  mm = coat.CoaTMini(pretrained="imagenet", classifier_activation=None)
  keras_out = mm(inputs).numpy()

  """ Verification """
  print(f"{np.allclose(torch_out, keras_out, atol=1e-3) = }")
  # np.allclose(torch_out, keras_out, atol=1e-3) = True
  ```
***
