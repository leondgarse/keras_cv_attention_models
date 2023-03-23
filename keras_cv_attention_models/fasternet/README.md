# ___Keras FasterNet___
***

## Summary
  - Keras implementation of [Github JierunChen/FasterNet](https://github.com/JierunChen/FasterNet). Paper [PDF 2303.03667 Run, Don’t Walk: Chasing Higher FLOPS for Faster Neural Networks ](https://arxiv.org/pdf/2303.03667.pdf).
  - Model weights ported from official publication.

  ![fasternet](https://user-images.githubusercontent.com/5744524/227238562-5ee980ba-84c7-44d0-969d-c472f6e719a4.jpg)
***

## Models
  | Model       | Params | FLOPs  | Input | Top1 Acc | Download |
  | ----------- | ------ | ------ | ----- | -------- | -------- |
  | FasterNetT0 | 3.9M   | 0.34G  | 224   | 71.9     | [fasternet_t0_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fasternet/fasternet_t0_imagenet.h5) |
  | FasterNetT1 | 7.6M   | 0.85G  | 224   | 76.2     | [fasternet_t1_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fasternet/fasternet_t1_imagenet.h5) |
  | FasterNetT2 | 15.0M  | 1.90G  | 224   | 78.9     | [fasternet_t2_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fasternet/fasternet_t2_imagenet.h5) |
  | FasterNetS  | 31.1M  | 4.55G  | 224   | 81.3     | [fasternet_s_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fasternet/fasternet_s_imagenet.h5)   |
  | FasterNetM  | 53.5M  | 8.72G  | 224   | 83.0     | [fasternet_m_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fasternet/fasternet_m_imagenet.h5)   |
  | FasterNetL  | 93.4M  | 15.49G | 224   | 83.5     | [fasternet_l_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fasternet/fasternet_l_imagenet.h5)   |

## Usage
  ```py
  from keras_cv_attention_models import fasternet

  # Will download and load pretrained imagenet weights.
  model = fasternet.FasterNetT2(pretrained="imagenet")

  # Run prediction
  from skimage.data import chelsea # Chelsea the cat
  preds = model(model.preprocess_input(chelsea()))
  print(model.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.76938057), ('n02123159', 'tiger_cat', 0.0810011), ...]
  ```
  **Use dynamic input resolution** by set `input_shape=(None, None, 3)`.
  ```py
  from keras_cv_attention_models import fasternet
  model = fasternet.FasterNetT2(input_shape=(None, None, 3), num_classes=0)
  # >>>> Load pretrained from: ~/.keras/models/fasternet_t2_imagenet.h5
  print(model.output_shape)
  # (None, None, None, 768)

  print(model(np.ones([1, 223, 123, 3])).shape)
  # (1, 6, 3, 768)
  print(model(np.ones([1, 32, 526, 3])).shape)
  # (1, 1, 16, 768)
  ```
  **Using PyTorch backend** by set `KECAM_BACKEND='torch'` environment variable.
  ```py
  os.environ['KECAM_BACKEND'] = 'torch'

  from keras_cv_attention_models import fasternet
  model = fasternet.FasterNetT2(input_shape=(None, None, 3), num_classes=0)
  # >>>> Using PyTorch backend
  # >>>> Aligned input_shape: [3, None, None]
  # >>>> Load pretrained from: ~/.keras/models/fasternet_t2_imagenet.h5
  print(model.output_shape)
  # (None, 768, None, None)

  import torch
  print(model(torch.ones([1, 3, 223, 123])).shape)
  # (1, 768, 6, 3 )
  print(model(torch.ones([1, 3, 32, 526])).shape)
  # (1, 768, 1, 16)
  ```  
## Verification with PyTorch version
  ```py
  """ PyTorch fasternet_t2 """
  sys.path.append('../FasterNet/')
  sys.path.append('../pytorch-image-models/')  # Needs timm
  import torch
  from models import fasternet as fasternet_torch

  torch_model = fasternet_torch.FasterNet()  # Default parameters is for T2
  ss = torch.load('fasternet_t2-epoch.289-val_acc1.78.8860.pth', map_location=torch.device('cpu'))
  torch_model.load_state_dict(ss)
  _ = torch_model.eval()

  """ Keras FasterNetT2 """
  from keras_cv_attention_models import fasternet
  mm = fasternet.FasterNetT2(pretrained="imagenet", classifier_activation=None)

  """ Verification """
  inputs = np.random.uniform(size=(1, *mm.input_shape[1:3], 3)).astype("float32")
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()
  keras_out = mm(inputs).numpy()
  print(f"{np.allclose(torch_out, keras_out, atol=1e-5) = }")
  # np.allclose(torch_out, keras_out, atol=1e-5) = True
  ```
