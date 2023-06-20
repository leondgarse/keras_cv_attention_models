# ___Keras FasterViT___
***

## Summary
  - Keras implementation of [Github NVlabs/FasterViT](https://github.com/NVlabs/FasterViT). Paper [PDF 2306.06189 FasterViT: Fast Vision Transformers with Hierarchical Attention](https://arxiv.org/pdf/2306.06189.pdf).
  - Model weights ported from official publication.
***

## Models
  | Model      | Params   | FLOPs   | Input | Top1 Acc | Download |
  | ---------- | -------- | ------- | ----- | -------- | -------- |
  | FasterViT0 | 31.40M   | 3.51G   | 224   | 82.1     |          |
  | FasterViT1 | 53.37M   | 5.52G   | 224   | 83.2     |          |
  | FasterViT2 | 75.92M   | 9.00G   | 224   | 84.2     |          |
  | FasterViT3 | 159.55M  | 18.75G  | 224   | 84.9     |          |
  | FasterViT4 | 351.12M  | 41.57G  | 224   | 85.4     |          |
  | FasterViT5 | 957.52M  | 114.08G | 224   | 85.6     |          |
  | FasterViT6 | 1360.33M | 144.13G | 224   | 85.8     |          |
## Usage
  ```py
  from keras_cv_attention_models import fastervit

  # Will download and load pretrained imagenet weights.
  mm = fastervit.FasterViT0(pretrained="imagenet")

  # Run prediction
  preds = mm(mm.preprocess_input(test_images.cat()))
  print(mm.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.7712158), ('n02123045', 'tabby', 0.017085848), ...]
  ```
## Verification with PyTorch version
  ```py
  """ PyTorch faster_vit_0_224 """
  sys.path.append('../FasterViT/')
  sys.path.append('../pytorch-image-models/')  # Needs timm
  import torch
  from models import faster_vit

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
