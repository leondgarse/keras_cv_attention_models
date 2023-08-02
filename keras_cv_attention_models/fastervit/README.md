# ___Keras FasterViT___
***

## Summary
  - Keras implementation of [Github NVlabs/FasterViT](https://github.com/NVlabs/FasterViT). Paper [PDF 2306.06189 FasterViT: Fast Vision Transformers with Hierarchical Attention](https://arxiv.org/pdf/2306.06189.pdf).
  - Model weights ported from official publication.
***

## Models
  | Model      | Params   | FLOPs   | Input | Top1 Acc |
  | ---------- | -------- | ------- | ----- | -------- |
  | [FasterViT0](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fastervit/fastervit_0_224_imagenet.h5) | 31.40M   | 3.51G   | 224   | 82.1     |
  | [FasterViT1](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fastervit/fastervit_1_224_imagenet.h5) | 53.37M   | 5.52G   | 224   | 83.2     |
  | [FasterViT2](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fastervit/fastervit_2_224_imagenet.h5) | 75.92M   | 9.00G   | 224   | 84.2     |
  | [FasterViT3](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fastervit/fastervit_3_224_imagenet.h5) | 159.55M  | 18.75G  | 224   | 84.9     |
  | [FasterViT4](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fastervit/fastervit_4_224_imagenet.h5) | 351.12M  | 41.57G  | 224   | 85.4     |
  | [FasterViT5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fastervit/fastervit_5_224_imagenet.h5) | 957.52M  | 114.08G | 224   | 85.6     |
  | [FasterViT6](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fastervit/fastervit_6_224_imagenet.1.h5), [+.2](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fastervit/fastervit_6_224_imagenet.2.h5) | 1360.33M | 144.13G | 224   | 85.8     |
## Usage
  ```py
  from keras_cv_attention_models import fastervit, test_images

  # Will download and load pretrained imagenet weights.
  mm = fastervit.FasterViT0(pretrained="imagenet")

  # Run prediction
  preds = mm(mm.preprocess_input(test_images.cat()))
  print(mm.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.7712158), ('n02123045', 'tabby', 0.017085848), ...]
  ```
  **Change input resolution**
  ```py
  from keras_cv_attention_models import fastervit, test_images

  # Will download and load pretrained imagenet weights.
  mm = fastervit.FasterViT1(pretrained="imagenet", input_shape=(219, 112, 3))

  # Run prediction
  preds = mm(mm.preprocess_input(test_images.cat()))
  print(mm.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.85765785), ('n02123045', 'tabby', 0.015630195), ...]
  ```
  **Switch to deploy** by calling `model.switch_to_deploy()`, will replace all positional embedding layers with a single bias one. **Note: when running inference using `ONNX`, `onnxsim` will automatically converting it to a single `Add`, no need to call this manually**.
  ```py
  from keras_cv_attention_models import fastervit, test_images, model_surgery

  mm = fastervit.FasterViT0(pretrained="imagenet")
  model_surgery.count_params(mm)
  # Total params: 31,408,168 | Trainable params: 31,404,840 | Non-trainable params:3,328
  preds = mm(mm.preprocess_input(test_images.cat()))

  bb = mm.switch_to_deploy()
  model_surgery.count_params(bb)
  # Total params: 28,382,248 | Trainable params: 28,378,920 | Non-trainable params:3,328
  preds_deploy = bb(bb.preprocess_input(test_images.cat()))

  print(f"{np.allclose(preds, preds_deploy) = }")
  # np.allclose(preds, preds_deploy) = True
  ```
  **Using PyTorch backend** by set `KECAM_BACKEND='torch'` environment variable.
  ```py
  os.environ['KECAM_BACKEND'] = 'torch'
  from keras_cv_attention_models import fastervit, test_images, model_surgery
  mm = fastervit.FasterViT0(input_shape=(219, 112, 3))
  # >>>> Using PyTorch backend
  # >>>> Load pretrained from: ~/.keras/models/fastervit_0_224_imagenet.h5

  # Run prediction
  preds = mm(mm.preprocess_input(test_images.cat()))
  print(mm.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.8119516), ('n02123045', 'tabby', 0.011075884), ...]
  ```
## Verification with PyTorch version
  ```py
  """ PyTorch faster_vit_0_224 """
  sys.path.append('../FasterViT/')
  sys.path.append('../pytorch-image-models/')  # Needs timm
  import torch
  from fastervit.models import faster_vit

  torch_model = faster_vit.faster_vit_0_224()
  ss = torch.load('fastervit_0_224_1k.pth.tar', map_location=torch.device('cpu'))
  torch_model.load_state_dict(ss.get('state_dict', ss))
  _ = torch_model.eval()

  """ Keras FasterViT0 """
  from keras_cv_attention_models import fastervit
  mm = fastervit.FasterViT0(pretrained="imagenet", classifier_activation=None)

  """ Verification """
  inputs = np.random.uniform(size=(1, *mm.input_shape[1:3], 3)).astype("float32")
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()
  keras_out = mm(inputs).numpy()
  print(f"{np.allclose(torch_out, keras_out, atol=1e-4) = }")
  # np.allclose(torch_out, keras_out, atol=1e-4) = True
  ```
