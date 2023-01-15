# ___Keras EfficientFormer___
***

## Summary
  - Keras implementation of [Github snap-research/efficientformer](https://github.com/snap-research/efficientformer).
  - EfficientFormer Paper [PDF 2206.01191 EfficientFormer: Vision Transformers at MobileNet Speed](https://arxiv.org/pdf/2206.01191.pdf).
  - EfficientFormerV2 Paper [PDF 2212.08059 Rethinking Vision Transformers for MobileNet Size and Speed](https://arxiv.org/pdf/2212.08059.pdf).
  - Model weights reloaded from official publication.

  ![efficientformer](https://user-images.githubusercontent.com/5744524/212475387-a4edffe6-9db5-463b-acaa-b20a57ab0c7c.png)
## Models
  | Model                      | Params | FLOPs | Input | Top1 Acc | Download |
  | -------------------------- | ------ | ----- | ----- | -------- | -------- |
  | EfficientFormerL1, distill | 12.3M  | 1.31G | 224   | 79.2     | [l1_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/efficientformer_l1_224_imagenet.h5) |
  | EfficientFormerL3, distill | 31.4M  | 3.95G | 224   | 82.4     | [l3_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/efficientformer_l3_224_imagenet.h5) |
  | EfficientFormerL7, distill | 74.4M  | 9.79G | 224   | 83.3     | [l7_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/efficientformer_l7_224_imagenet.h5) |

  | Model                        | Params | FLOPs  | Input | Top1 Acc | Download |
  | ---------------------------- | ------ | ------ | ----- | -------- | -------- |
  | EfficientFormerV2S0, distill | 3.60M  | 405.2M | 224   | 76.2     | [v2_s0_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientformer/efficientformer_v2_s0_224_imagenet.h5) |
  | EfficientFormerV2S1, distill | 6.19M  | 665.6M | 224   | 79.7     | [v2_s1_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientformer/efficientformer_v2_s1_224_imagenet.h5) |
  | EfficientFormerV2S2, distill | 12.7M  | 1.27G  | 224   | 82.0     | [v2_s2_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientformer/efficientformer_v2_s2_224_imagenet.h5) |
  | EfficientFormerV2L, distill  | 26.3M  | 2.59G  | 224   | 83.5     | [v2_l_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientformer/efficientformer_v2_l_224_imagenet.h5) |

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
  mm = efficientformer.EfficientFormerV2S0(input_shape=(292, 213, 3), pretrained="imagenet", use_distillation=True)
  # >>>> Load pretrained from: ~/.keras/models/efficientformer_v2_s0_224_imagenet.h5
  # ...
  # >>>> Reload mismatched weights: -1 -> (292, 213)
  # >>>> Reload layer: stack3_block5_attn_pos_emb

  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0))
  pred = tf.nn.softmax((pred[0] + pred[1]) / 2).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.575441), ('n02123159', 'tiger_cat', 0.25620097), ...]
  ```
## Verification with PyTorch version
  ```py
  input_shape = 224
  inputs = np.random.uniform(size=(1, input_shape, input_shape, 3)).astype("float32")

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
