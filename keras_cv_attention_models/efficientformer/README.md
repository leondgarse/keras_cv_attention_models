# ___Keras EfficientFormer___
***

## Summary
  - Keras implementation of [Github snap-research/efficientformer](https://github.com/snap-research/efficientformer). Paper [PDF 2206.01191 EfficientFormer: Vision Transformers at MobileNet Speed](https://arxiv.org/pdf/2206.01191.pdf).
  - Model weights reloaded from official publication.
## Models
  | Model                      | Params | FLOPs | Input | Top1 Acc | Download |
  | -------------------------- | ------ | ----- | ----- | -------- | -------- |
  | EfficientFormerL1, distill | 12.3M  | 1.31G | 224   | 79.2     | [l1_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/efficientformer_l1_224_imagenet.h5) |
  | EfficientFormerL3, distill | 31.4M  | 3.95G | 224   | 82.4     | [l3_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/efficientformer_l3_224_imagenet.h5) |
  | EfficientFormerL7, distill | 74.4M  | 9.79G | 224   | 83.3     | [l7_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/efficientformer_l7_224_imagenet.h5) |
## Usage
  ```py
  from keras_cv_attention_models import efficientformer

  # Will download and load pretrained imagenet weights.
  mm = efficientformer.EfficientFormerL1(pretrained="imagenet", use_distillation=True)
  print(mm.output_names, mm.output_shape)
  # ['head', 'distill_head'] [(None, 1000), (None, 1000)]

  # Run prediction
  import tensorflow as tf
  from tensorflow import keras
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0))
  pred = tf.nn.softmax((pred[0] + pred[1]) / 2).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.98188466), ('n02123159', 'tiger_cat', 0.011581295), ...]
  ```
  set `use_distillation=False` for output only one head.
  ```py
  mm = efficientformer.EfficientFormerL1(use_distillation=False, classifier_activation="softmax")
  print(mm.output_names, mm.output_shape)
  # ['head'] (None, 1000)
  ```
  **Change input resolution**
  ```py
  from keras_cv_attention_models import efficientformer
  # Will download and load pretrained imagenet weights.
  mm = efficientformer.EfficientFormerL1(input_shape=(292, 213, 3), pretrained="imagenet", use_distillation=True)
  # >>>> Load pretrained from: ~/.keras/models/efficientformer_l1_224_imagenet.h5
  # ...
  # >>>> Reload mismatched weights: -1 -> (292, 213)
  # >>>> Reload layer: stack4_block4_attn_pos

  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0))
  pred = tf.nn.softmax((pred[0] + pred[1]) / 2).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.9775359), ('n02123159', 'tiger_cat', 0.010384818), ...]
  ```
## Verification with PyTorch version
  ```py
  input_shape = 224
  # inputs = np.random.uniform(size=(1, input_shape, input_shape, 3)).astype("float32")
  inputs = np.ones((1, input_shape, input_shape, 3)).astype("float32")

  """ PyTorch efficientformer_l1 """
  sys.path.append('../EfficientFormer/')
  sys.path.append('../pytorch-image-models/')
  import torch
  from models import efficientformer as torch_efficientformer
  torch_model = torch_efficientformer.efficientformer_l1(pretrained=True)
  ss = torch.load('efficientformer_l1_1000d.pth', map_location=torch.device('cpu'))
  torch_model.load_state_dict(ss['model'])
  _ = torch_model.eval()
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()

  """ Keras EfficientFormerL1 """
  from keras_cv_attention_models import efficientformer
  mm = efficientformer.EfficientFormerL1(pretrained="imagenet", use_distillation=True)
  pred = mm(inputs)
  keras_out = ((pred[0] + pred[1]) / 2).numpy()

  """ Verification """
  print(f"{np.allclose(torch_out, keras_out, atol=1e-4) = }")
  # np.allclose(torch_out, keras_out, atol=1e-4) = True
  ```
***
