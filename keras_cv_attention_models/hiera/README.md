# ___Keras Hiera___
***

## Summary
  - Keras implementation of [Github facebookresearch/hiera](https://github.com/facebookresearch/hiera). Paper [PDF 2306.00989 Hiera: A Hierarchical Vision Transformer without the Bells-and-Whistles](https://arxiv.org/pdf/2306.00989.pdf).
  - Model weights ported from official publication.
***

## Models
  | Model                        | Params  | FLOPs   | Input | Top1 Acc |
  | ---------------------------- | ------- | ------- | ----- | -------- |
  | [HieraTiny, mae_in1k_ft1k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/hiera/hiera_tiny_224_mae_in1k_ft1k.h5)     | 27.91M  | 4.93G   | 224   | 82.8     |
  | [HieraSmall, mae_in1k_ft1k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/hiera/hiera_small_224_mae_in1k_ft1k.h5)    | 35.01M  | 6.44G   | 224   | 83.8     |
  | [HieraBase, mae_in1k_ft1k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/hiera/hiera_base_224_mae_in1k_ft1k.h5)     | 51.52M  | 9.43G   | 224   | 84.5     |
  | [HieraBasePlus, mae_in1k_ft1k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/hiera/hiera_base_plus_224_mae_in1k_ft1k.h5) | 69.90M  | 12.71G  | 224   | 85.2     |
  | [HieraLarge, mae_in1k_ft1k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/hiera/hiera_large_224_mae_in1k_ft1k.h5)    | 213.74M | 40.43G  | 224   | 86.1     |
  | [HieraHuge, mae_in1k_ft1k](https://github.com/leondgarse/keras_cv_attention_models/releases/download/hiera/hiera_huge_224_mae_in1k_ft1k.h5)     | 672.78M | 125.03G | 224   | 86.9     |
## Usage
  ```py
  from keras_cv_attention_models import hiera, test_images

  # Will download and load pretrained imagenet weights.
  mm = hiera.HieraBase()
  # >>>> Load pretrained from: ~/.keras/models/hiera_base_224_mae_in1k_ft1k.h5

  # Run prediction
  preds = mm(mm.preprocess_input(test_images.cat()))
  print(mm.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.8947084), ('n02123045', 'tabby', 0.006296753), ...]
  ```
  **Change input resolution** input_shape should be divisible by `32`, which is `stem_strides=4 * strides=[1, 2, 2, 2]`. **Note: pretrained weights not works well with new input_shape, as `window_size` is bounded with `unroll` and `strides`**.
  ```py
  from keras_cv_attention_models import hiera, test_images
  mm = hiera.HieraBase(input_shape=(448, 448, 3))
  # >>>> Load pretrained from: ~/.keras/models/hiera_base_224_mae_in1k_ft1k.h5
  # WARNING:tensorflow:Skipping loading weights for layer #3 (named positional_embedding) ...
  # >>>> Reload mismatched weights: 224 -> (448, 448)
  # >>>> Reload layer: positional_embedding
  print(mm.decode_predictions(mm(mm.preprocess_input(test_images.cat()))))
  # [('n04275548', 'spider_web', 0.4003983), ('n01773549', 'barn_spider', 0.10982952), ...]

  """ A little better with new strides """
  mm = hiera.HieraBase(input_shape=(448, 448, 3), strides=[1, 4, 2, 2])
  print(mm.decode_predictions(mm(mm.preprocess_input(test_images.cat()))))
  # [('n02124075', 'Egyptian_cat', 0.37766436), ('n03000247', 'chain_mail', 0.09813311), ...]
  ```
  **Using PyTorch backend** by set `KECAM_BACKEND='torch'` environment variable.
  ```py
  os.environ['KECAM_BACKEND'] = 'torch'
  from keras_cv_attention_models import hiera, test_images
  model = hiera.HieraBase()
  # >>>> Using PyTorch backend
  # >>>> Load pretrained from: ~/.keras/models/hiera_base_224_mae_in1k_ft1k.h5

  # Run prediction
  preds = model(model.preprocess_input(test_images.cat()))
  print(model.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.8947087), ('n02123045', 'tabby', 0.006296773), ...]
  ```
## Verification with PyTorch version
  ```py
  """ PyTorch torch_hiera """
  sys.path.append('../hiera/')
  sys.path.append('../pytorch-image-models/')  # Needs timm
  import torch
  from hiera import hiera as torch_hiera

  torch_model = torch_hiera.hiera_base_224()
  ss = torch.load('hiera_base_224.pth', map_location=torch.device('cpu'))
  torch_model.load_state_dict(ss['model_state'])
  _ = torch_model.eval()

  """ Keras HieraBase """
  from keras_cv_attention_models import hiera
  mm = hiera.HieraBase(classifier_activation="softmax")

  """ Verification """
  inputs = np.random.uniform(size=(1, *mm.input_shape[1:3], 3)).astype("float32")
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()
  keras_out = mm(inputs).numpy()
  print(f"{np.allclose(torch_out, keras_out, atol=1e-5) = }")
  # np.allclose(torch_out, keras_out, atol=1e-5) = True
  ```
  **With new input_shape**
  ```py
  new_shape = 448

  """ PyTorch torch_hiera """
  sys.path.append('../hiera/')
  sys.path.append('../pytorch-image-models/')  # Needs timm
  import torch
  from hiera import hiera as torch_hiera

  torch_model = torch_hiera.hiera_base_224(input_size=(new_shape, new_shape))
  ss = torch.load('hiera_base_224.pth', map_location=torch.device('cpu'))['model_state']
  aa = ss['pos_embed'].detach().reshape([1, 56, 56, 96]).permute([0, 3, 1, 2])
  bb = torch.functional.F.interpolate(aa, [new_shape // 4, new_shape // 4], mode='bilinear')
  ss['pos_embed'] = bb.permute([0, 2, 3, 1]).reshape([1, -1, 96])

  torch_model.load_state_dict(ss)
  _ = torch_model.eval()

  """ Keras HieraBase """
  from keras_cv_attention_models import hiera
  mm = hiera.HieraBase(classifier_activation="softmax", input_shape=(new_shape, new_shape, 3))

  """ Verification """
  inputs = np.random.uniform(size=(1, *mm.input_shape[1:3], 3)).astype("float32")
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()
  keras_out = mm(inputs).numpy()
  print(f"{np.allclose(torch_out, keras_out, atol=1e-5) = }")
  # np.allclose(torch_out, keras_out, atol=1e-5) = True
  ```
