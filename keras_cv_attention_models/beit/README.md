# ___Keras BEiT / ViT / FlexiViT / EVA___
***

## Summary
  - Beit Paper [PDF 2106.08254 BEIT: BERT Pre-Training of Image Transformers](https://arxiv.org/pdf/2106.08254.pdf). Model weights reloaded from [Github microsoft/beit](https://github.com/microsoft/unilm/tree/master/beit).
  - BeitV2 Paper [PDF 2208.06366 BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers](https://arxiv.org/pdf/2208.06366.pdf).  Model weights reloaded from [Github microsoft/beit2](https://github.com/microsoft/unilm/tree/master/beit2)
  - FlexiViT Paper [PDF 2212.08013 FlexiViT: One Model for All Patch Sizes](https://arxiv.org/pdf/2212.08013.pdf). Model weights reloaded from [Github google-research/big_vision](https://github.com/google-research/big_vision/tree/main/big_vision/configs/proj/flexivit).
  - EVA Paper [PDF 2211.07636 EVA: Exploring the Limits of Masked Visual Representation Learning at Scale](https://arxiv.org/pdf/2211.07636.pdf). Model weights reloaded from [Github baaivision/EVA](https://github.com/baaivision/EVA).
***

## Models
  | Model                 | Params  | FLOPs   | Input | Top1 Acc | Download                         |
  | --------------------- | ------- | ------- | ----- | -------- | -------------------------------- |
  | BeitBasePatch16, 21k  | 86.53M  | 17.61G  | 224   | 85.240   | [beit_base_patch16_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/beit_base_patch16_224_imagenet21k-ft1k.h5)  |
  |                       | 86.74M  | 55.70G  | 384   | 86.808   | [beit_base_patch16_384.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/beit_base_patch16_384_imagenet21k-ft1k.h5)  |
  | BeitLargePatch16, 21k | 304.43M | 61.68G  | 224   | 87.476   | [beit_large_patch16_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/beit_large_patch16_224_imagenet21k-ft1k.h5) |
  |                       | 305.00M | 191.65G | 384   | 88.382   | [beit_large_patch16_384.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/beit_large_patch16_384_imagenet21k-ft1k.h5) |
  |                       | 305.67M | 363.46G | 512   | 88.584   | [beit_large_patch16_512.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/beit_large_patch16_512_imagenet21k-ft1k.h5) |

  | Model              | Params  | FLOPs  | Input | Top1 Acc | Download |
  | ------------------ | ------- | ------ | ----- | -------- | -------- |
  | BeitV2BasePatch16  | 86.53M  | 17.61G | 224   | 85.5     |          |
  | - imagenet21k-ft1k | 86.53M  | 17.61G | 224   | 86.5     | [beit_v2_base_patch16_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/beit_v2_base_patch16_224_imagenet21k-ft1k.h5)  |
  | BeitV2BasePatch16  | 304.43M | 61.68G | 224   | 87.3     |          |
  | - imagenet21k-ft1k | 304.43M | 61.68G | 224   | 88.4     | [beit_v2_large_patch16_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/beit_v2_large_patch16_224_imagenet21k-ft1k.h5)  |

  | Model         | Params  | FLOPs  | Input | Top1 Acc | Download |
  | ------------- | ------- | ------ | ----- | -------- | -------- |
  | FlexiViTSmall | 22.06M  | 5.36G  | 240   | 82.53    | [flexivit_small_240.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/flexivit_small_240_imagenet.h5) |
  | FlexiViTBase  | 86.59M  | 20.33G | 240   | 84.66    | [flexivit_base_240.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/flexivit_base_240_imagenet.h5) |
  | FlexiViTLarge | 304.47M | 71.09G | 240   | 85.64    | [flexivit_large_240.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/flexivit_large_240_imagenet.h5) |

  - **EVA models** Note: `EvaGiantPatch14` weights are in `float16` format, as `float32` ones are too large that exceed 2GB.

  | Model                 | Params  | FLOPs    | Input | Top1 Acc | Download |
  | --------------------- | ------- | -------- | ----- | -------- | -------- |
  | EvaLargePatch14, 22k  | 304.14M | 61.65G   | 196   | 88.59    | [eva_large_patch14_196.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/eva_large_patch14_196_imagenet21k-ft1k.h5) |
  |                       | 304.53M | 191.55G  | 336   | 89.20    | [eva_large_patch14_336.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/eva_large_patch14_336_imagenet21k-ft1k.h5) |
  | EvaGiantPatch14, clip | 1012.6M | 267.40G  | 224   | 89.10    | [eva_giant_patch14_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/eva_giant_patch14_224_imagenet21k-ft1k.h5) |
  | - m30m                | 1013.0M | 621.45G  | 336   | 89.57    | [eva_giant_patch14_336.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/eva_giant_patch14_336_imagenet21k-ft1k.h5) |
  | - m30m                | 1014.4M | 1911.61G | 560   | 89.80    | [eva_giant_patch14_560.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/eva_giant_patch14_560_imagenet21k-ft1k.h5) |
## Usage
  ```py
  from keras_cv_attention_models import beit

  # Will download and load pretrained imagenet21k-ft1k weights.
  mm = beit.BeitBasePatch16(input_shape=(384, 384, 3), pretrained="imagenet21k-ft1k")

  # Run prediction
  from skimage.data import chelsea
  pred = mm([mm.preprocess_input(chelsea())])
  print(mm.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.7306834),
  #  ('n02123159', 'tiger_cat', 0.045104492),
  #  ('n02123045', 'tabby', 0.030001672),
  #  ('n02127052', 'lynx', 0.013072581),
  #  ('n02123597', 'Siamese_cat', 0.0062989206)]
  ```
  **Change input resolution** if input_shape is not within pre-trained, will load `MultiHeadRelativePositionalEmbedding` / `PositionalEmbedding` weights by `load_resized_weights`.
  ```py
  from keras_cv_attention_models import beit
  mm = beit.EvaGiantPatch14(input_shape=(256, 160, 3), num_classes=1000)
  # >>>> Load pretrained from: /home/leondgarse/.keras/models/eva_giant_patch14_224_imagenet21k-ft1k.h5
  # WARNING:tensorflow:Skipping loading weights for layer #4 (named positional_embedding) due to mismatch in shape ...
  # >>>> Reload mismatched weights: 224 -> (256, 160)
  # >>>> Reload layer: positional_embedding

  # Run prediction on Chelsea with (256, 160) resolution
  from skimage.data import chelsea
  pred = mm([mm.preprocess_input(chelsea())])
  print(mm.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.7318501), ('n02123045', 'tabby', 0.021020193), ...]
  ```
## FlexiViT / EVA models with new patch size
  - For `FlexiViT` / `EVA` models, when setting new `patch_size`, will reload `stem_conv` weights. Source implementation [Github google-research/big_vision/flexi](https://github.com/google-research/big_vision/blob/main/big_vision/models/proj/flexi/vit.py#L30), paper [PDF 2212.08013 FlexiViT: One Model for All Patch Sizes](https://arxiv.org/pdf/2212.08013.pdf).
  - Not works for `BEiT` models.
  ```py
  from skimage.data import chelsea
  from keras_cv_attention_models import flexivit, beit, eva

  mm = flexivit.FlexiViTSmall(patch_size=32)  # Original is patch_size=16
  # >>>> Reload mismatched weights: 240 -> (240, 240)
  # >>>> Reload layer: positional_embedding
  # >>>> Reload layer: stem_conv
  print(mm.decode_predictions(mm(mm.preprocess_input(chelsea())))[0])
  # [('n02124075', 'Egyptian_cat', 0.6939351), ('n02123045', 'tabby', 0.033506528), ...]
  ```
  **Also works for EVA models**
  ```py
  from skimage.data import chelsea
  from keras_cv_attention_models import flexivit, beit, eva

  mm = beit.EvaLargePatch14(patch_size=32)  # Original is patch_size=14
  # >>>> Reload mismatched weights: 196 -> (196, 196)
  # >>>> Reload layer: positional_embedding
  # >>>> Reload layer: stem_conv
  print(mm.decode_predictions(mm(mm.preprocess_input(chelsea())))[0])
  # [('n02124075', 'Egyptian_cat', 0.47458684), ('n02123045', 'tabby', 0.04412323), ...]
  ```
## Custom ViT models load and convert weights from timm torch model
  - `keras_model_load_weights_from_pytorch_model` can be used for loading most `Beit` / `ViT` / `FlexViT` / `EVA` model weights from initialized timm torch model. Outputs is converted weight file, which can be used for afterward and further usage.
  ```py
  """ Build a timm ViT model """
  import timm
  torch_model = timm.models.vit_tiny_patch16_224(pretrained=True)
  _ = torch_model.eval()

  """ Build a ViT model same architecture with torch_model """
  from keras_cv_attention_models import flexivit
  mm = flexivit.ViT(depth=12, embed_dim=192, num_heads=3, pretrained=None, classifier_activation=None)
  flexivit.keras_model_load_weights_from_pytorch_model(mm, torch_model)
  # >>>> Save model to: vit_224.h5
  # >>>> Keras model prediction: [('n02123045', 'tabby', 11.990417), ('n02123159', 'tiger_cat', 11.630723), ...]
  # >>>> Torch model prediction: [[('n02123045', 'tabby', 11.99042), ('n02123159', 'tiger_cat', 11.630725), ...]
  ```
## Verification with PyTorch version
  ```py
  inputs = np.random.uniform(size=(1, 224, 224, 3)).astype("float32")

  """ PyTorch beit_base_patch16_224 """
  import torch
  import timm
  torch_model = timm.models.beit_base_patch16_224(pretrained=True)
  _ = torch_model.eval()
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()

  """ Keras BeitBasePatch16 """
  from keras_cv_attention_models import beit
  mm = beit.BeitBasePatch16(classifier_activation=None)
  keras_out = mm(inputs).numpy()

  """ Verification """
  print(f"{np.allclose(torch_out, keras_out, atol=1e-3) = }")
  # np.allclose(torch_out, keras_out, atol=1e-3) = True
  ```
***
