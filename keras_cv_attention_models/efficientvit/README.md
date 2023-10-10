# ___Keras EfficientViT___
***

## Summary
  - `EfficientViT_M` is implementation of [Github microsoft/Cream/EfficientViT/classification](https://github.com/microsoft/Cream/tree/main/EfficientViT/classification). Paper [PDF 2305.07027 EfficientViT: Memory Efficient Vision Transformer with Cascaded Group Attention](https://arxiv.org/pdf/2305.07027.pdf).
  - `EfficientViT_B` is implementation of [Github mit-han-lab/efficientvit](https://github.com/mit-han-lab/efficientvit). Paper [PDF 2205.14756 EfficientViT: Lightweight Multi-Scale Attention for On-Device Semantic Segmentation](https://arxiv.org/pdf/2205.14756.pdf).
  - Model weights ported from official publication.
  - **Note: `EfficientViT_M` and `EfficientViT_B` are actually totally different, just a name conflict.**
***

## Models
  | Model           | Params | FLOPs | Input | Top1 Acc | Download |
  | --------------- | ------ | ----- | ----- | -------- | -------- |
  | EfficientViT_M0 | 2.35M  | 79.4M | 224   | 63.2     | [efficientvit_m0_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientvit/efficientvit_m0_224_imagenet.h5) |
  | EfficientViT_M1 | 2.98M  | 167M  | 224   | 68.4     | [efficientvit_m1_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientvit/efficientvit_m1_224_imagenet.h5) |
  | EfficientViT_M2 | 4.19M  | 201M  | 224   | 70.8     | [efficientvit_m2_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientvit/efficientvit_m2_224_imagenet.h5) |
  | EfficientViT_M3 | 6.90M  | 263M  | 224   | 73.4     | [efficientvit_m3_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientvit/efficientvit_m3_224_imagenet.h5) |
  | EfficientViT_M4 | 8.80M  | 299M  | 224   | 74.3     | [efficientvit_m4_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientvit/efficientvit_m4_224_imagenet.h5) |
  | EfficientViT_M5 | 12.47M | 522M  | 224   | 77.1     | [efficientvit_m5_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientvit/efficientvit_m5_224_imagenet.h5) |

  | Model           | Params | FLOPs | Input | Top1 Acc | Download |
  | --------------- | ------ | ----- | ----- | -------- | -------- |
  | EfficientViT_B0 | 3.41M  | 0.12G | 224   | 71.6 ?   | [efficientvit_b0_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientvit/efficientvit_b0_224_imagenet.h5) |
  | EfficientViT_B1 | 9.10M  | 0.58G | 224   | 79.4     | [efficientvit_b1_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientvit/efficientvit_b1_224_imagenet.h5) |
  | EfficientViT_B1 | 9.10M  | 0.78G | 256   | 79.9     | [efficientvit_b1_256_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientvit/efficientvit_b1_256_imagenet.h5) |
  | EfficientViT_B1 | 9.10M  | 1.03G | 288   | 80.4     | [efficientvit_b1_288_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientvit/efficientvit_b1_288_imagenet.h5) |
  | EfficientViT_B2 | 24.33M | 1.68G | 224   | 82.1     | [efficientvit_b2_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientvit/efficientvit_b2_224_imagenet.h5) |
  | EfficientViT_B2 | 24.33M | 2.25G | 256   | 82.7     | [efficientvit_b2_256_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientvit/efficientvit_b2_256_imagenet.h5) |
  | EfficientViT_B2 | 24.33M | 2.92G | 288   | 83.1     | [efficientvit_b2_288_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientvit/efficientvit_b2_288_imagenet.h5) |
  | EfficientViT_B3 | 48.65M | 4.14G | 224   | 83.5     | [efficientvit_b3_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientvit/efficientvit_b3_224_imagenet.h5) |
  | EfficientViT_B3 | 48.65M | 5.51G | 256   | 83.8     | [efficientvit_b3_256_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientvit/efficientvit_b3_256_imagenet.h5) |
  | EfficientViT_B3 | 48.65M | 7.14G | 288   | 84.2     | [efficientvit_b3_288_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientvit/efficientvit_b3_288_imagenet.h5) |
  | EfficientViT_L1 | 52.65M | 5.28G | 224   | 84.48    | [efficientvit_l1_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientvit/efficientvit_l1_224_imagenet.h5) |
  | EfficientViT_L2 | 63.71M | 6.98G | 224   | 85.05    | [efficientvit_l2_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientvit/efficientvit_l2_224_imagenet.h5) |
  | EfficientViT_L2 | 63.71M | 20.7G | 384   | 85.98    | [efficientvit_l2_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientvit/efficientvit_l2_384_imagenet.h5) |
  | EfficientViT_L3 | 246.0M | 27.6G | 224   | 85.814   | [efficientvit_l3_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientvit/efficientvit_l3_224_imagenet.h5) |
  | EfficientViT_L3 | 246.0M | 81.6G | 384   | 86.408   | [efficientvit_l3_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientvit/efficientvit_l3_384_imagenet.h5) |
## Usage
  ```py
  from keras_cv_attention_models import efficientvit, test_images

  # Will download and load pretrained imagenet weights.
  model = efficientvit.EfficientViT_M0(pretrained="imagenet")

  # Run prediction
  preds = model(model.preprocess_input(test_images.cat()))
  print(model.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.4258187), ('n02123159', 'tiger_cat', 0.14353083), ...]
  ```
  **EfficientViT_M Change input resolution** if input_shape is all less than `window_size * 32 = 7 * 32 = 224`, or `window_size` is not `7`, will load `MultiHeadPositionalEmbedding` weights by `load_resized_weights`
  ```py
  from keras_cv_attention_models import efficientvit, test_images
  model = efficientvit.EfficientViT_M1(input_shape=(193, 127, 3))
  # >>>> Load pretrained from: ~/.keras/models/efficientvit_m1_224_imagenet.h5
  # Warning: skip loading weights for layer: stack2_block1_attn_1_attn_pos, required weights: [[28]], provided: [(49,)]
  # ...
  # >>>> Reload mismatched weights: 224 -> (193, 127)
  # >>>> Reload layer: stack1_block1_attn_1_attn_pos
  # ...

  # Run prediction
  preds = model(model.preprocess_input(test_images.cat()))
  print(model.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.50921583), ('n02123045', 'tabby', 0.14553155), ...]
  ```
  **EfficientViT_B Change input resolution**
  ```py
  from keras_cv_attention_models import efficientvit, test_images
  model = efficientvit.EfficientViT_B1(input_shape=(193, 257, 3), pretrained="imagenet")
  # >>>> Load pretrained from: /home/leondgarse/.keras/models/efficientvit_b1_256_imagenet.h5

  # Run prediction
  preds = model(model.preprocess_input(test_images.cat()))
  print(model.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.98537457), ('n02123159', 'tiger_cat', 0.0035056944), ...]
  ```
  **Using PyTorch backend** by set `KECAM_BACKEND='torch'` environment variable.
  ```py
  os.environ['KECAM_BACKEND'] = 'torch'
  from keras_cv_attention_models import efficientvit, test_images
  model = efficientvit.EfficientViT_M0(pretrained="imagenet", input_shape=(256, 256, 3), window_size=8)
  # >>>> Using PyTorch backend
  # >>>> Aligned input_shape: [3, 256, 256]
  # >>>> Load pretrained from: ~/.keras/models/efficientvit_m0_224_imagenet.h5
  # Warning: skip loading weights for layer: stack2_block1_attn_1_attn_pos, required weights: [[64]], provided: [(49,)]
  # ...
  # >>>> Reload mismatched weights: 224 -> (256, 256)
  # >>>> Reload layer: stack1_block1_attn_1_attn_pos
  # ...

  # Run prediction
  preds = model(model.preprocess_input(test_images.cat()))
  print(model.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.42882437), ('n02123159', 'tiger_cat', 0.15752947), ...]
  ```
## Verification with PyTorch version
  ```py
  """ PyTorch EfficientViT_M0 """
  sys.path.append('../Cream/EfficientViT/classification/')
  sys.path.append('../pytorch-image-models/')  # Needs timm
  import torch
  from model.build import EfficientViT_M0

  torch_model = EfficientViT_M0(pretrained='efficientvit_m0')
  _ = torch_model.eval()

  """ Keras EfficientViT_M0 """
  from keras_cv_attention_models import efficientvit
  mm = efficientvit.EfficientViT_M0(pretrained="imagenet", classifier_activation=None)

  """ Verification """
  inputs = np.random.uniform(size=(1, *mm.input_shape[1:3], 3)).astype("float32")
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()
  keras_out = mm(inputs).numpy()
  print(f"{np.allclose(torch_out, keras_out, atol=1e-5) = }")
  # np.allclose(torch_out, keras_out, atol=1e-5) = True
  ```
  ```py
  """ PyTorch efficientvit_cls_b1 """
  sys.path.append('../efficientvit/')
  import torch
  from models.efficientvit import cls as torch_efficientvit

  torch_model = torch_efficientvit.efficientvit_cls_b1()
  ss = torch.load('b1-r224.pt', map_location='cpu')
  torch_model.load_state_dict(ss['state_dict'])
  _ = torch_model.eval()

  """ Keras EfficientViT_B1 """
  from keras_cv_attention_models import efficientvit
  mm = efficientvit.EfficientViT_B1(pretrained="imagenet", classifier_activation=None)

  """ Verification """
  inputs = np.random.uniform(size=(1, *mm.input_shape[1:3], 3)).astype("float32")
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()
  keras_out = mm(inputs).numpy()
  print(f"{np.allclose(torch_out, keras_out, atol=1e-5) = }")
  # np.allclose(torch_out, keras_out, atol=1e-5) = True
  ```
