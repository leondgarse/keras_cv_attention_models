# ___Keras BeiT / BeitV2 / ViT / ViT-5 / FlexiViT / EVA / EVA02 / DINOv2 / MetaTransFormer___
***

## Summary
  - Beit Paper [PDF 2106.08254 BEIT: BERT Pre-Training of Image Transformers](https://arxiv.org/pdf/2106.08254.pdf). Model weights reloaded from [Github microsoft/beit](https://github.com/microsoft/unilm/tree/master/beit).
  - BeitV2 Paper [PDF 2208.06366 BEiT v2: Masked Image Modeling with Vector-Quantized Visual Tokenizers](https://arxiv.org/pdf/2208.06366.pdf).  Model weights reloaded from [Github microsoft/beit2](https://github.com/microsoft/unilm/tree/master/beit2)
  - DINOv2 Paper [PDF 2304.07193 DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/pdf/2304.07193.pdf). Model weights reloaded from [Github facebookresearch/dinov2](https://github.com/facebookresearch/dinov2).
  - EVA Paper [PDF 2211.07636 EVA: Exploring the Limits of Masked Visual Representation Learning at Scale](https://arxiv.org/pdf/2211.07636.pdf). Model weights reloaded from [Github baaivision/EVA](https://github.com/baaivision/EVA).
  - EVA02 Paper [PDF 2303.11331 EVA: EVA-02: A Visual Representation for Neon Genesis](https://arxiv.org/pdf/2303.11331.pdf). Model weights reloaded from [Github baaivision/EVA/EVA-02](https://github.com/baaivision/EVA/tree/master/EVA-02).
  - FlexiViT Paper [PDF 2212.08013 FlexiViT: One Model for All Patch Sizes](https://arxiv.org/pdf/2212.08013.pdf). Model weights reloaded from [Github google-research/big_vision](https://github.com/google-research/big_vision/tree/main/big_vision/configs/proj/flexivit).
  - MetaTransFormer Paper [PDF 2307.10802 Meta-Transformer: A Unified Framework for Multimodal Learning](https://arxiv.org/abs/2307.10802). Model weights reloaded from [Github invictus717/MetaTransformer](https://github.com/invictus717/MetaTransformer). **Note: image model weights for transformer blocks are same with the multi tasks one**
  - ViT-5 Paper [PDF 2602.08071 ViT-5: Vision Transformers for The Mid-2020s](https://arxiv.org/abs/2602.08071). Model weights reloaded from [Github wangf3014/ViT-5](https://github.com/wangf3014/ViT-5).
***

## Beit Models
  | Model                 | Params  | FLOPs   | Input | Top1 Acc | Download                         |
  | --------------------- | ------- | ------- | ----- | -------- | -------------------------------- |
  | BeitBasePatch16, 21k  | 86.53M  | 17.61G  | 224   | 85.240   | [beit_base_patch16_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/beit_base_patch16_224_imagenet21k-ft1k.h5)  |
  |                       | 86.74M  | 55.70G  | 384   | 86.808   | [beit_base_patch16_384.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/beit_base_patch16_384_imagenet21k-ft1k.h5)  |
  | BeitLargePatch16, 21k | 304.43M | 61.68G  | 224   | 87.476   | [beit_large_patch16_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/beit_large_patch16_224_imagenet21k-ft1k.h5) |
  |                       | 305.00M | 191.65G | 384   | 88.382   | [beit_large_patch16_384.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/beit_large_patch16_384_imagenet21k-ft1k.h5) |
  |                       | 305.67M | 363.46G | 512   | 88.584   | [beit_large_patch16_512.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/beit_large_patch16_512_imagenet21k-ft1k.h5) |
## BeitV2 Models
  | Model              | Params  | FLOPs  | Input | Top1 Acc | Download |
  | ------------------ | ------- | ------ | ----- | -------- | -------- |
  | BeitV2BasePatch16  | 86.53M  | 17.61G | 224   | 85.5     |          |
  | - imagenet21k-ft1k | 86.53M  | 17.61G | 224   | 86.5     | [beit_v2_base_patch16_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/beit_v2_base_patch16_224_imagenet21k-ft1k.h5)  |
  | BeitV2LargePatch16 | 304.43M | 61.68G | 224   | 87.3     |          |
  | - imagenet21k-ft1k | 304.43M | 61.68G | 224   | 88.4     | [beit_v2_large_patch16_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/beit_v2_large_patch16_224_imagenet21k-ft1k.h5)  |
## DINOv2 models
  - Note: `DINOv2_ViT_Giant14` weights are in `float16` format and commpressed by `gzip`, as `float32` ones are too large that exceed 2GB.

  | Model              | Params  | FLOPs   | Input | Top1 Acc | Download |
  | ------------------ | ------- | ------- | ----- | -------- | -------- |
  | DINOv2_ViT_Small14 | 22.83M  | 47.23G  | 518   | 81.1     | [dinov2_vit_small14.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/dinov2_vit_small14_518_imagenet.h5) |
  | DINOv2_ViT_Base14  | 88.12M  | 152.6G  | 518   | 84.5     | [dinov2_vit_base14.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/dinov2_vit_base14_518_imagenet.h5) |
  | DINOv2_ViT_Large14 | 306.4M  | 509.6G  | 518   | 86.3     | [dinov2_vit_large14.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/dinov2_vit_large14_518_imagenet.h5) |
  | DINOv2_ViT_Giant14 | 1139.6M | 1790.3G | 518   | 86.5     | [dinov2_vit_giant14.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/dinov2_vit_giant14_518_imagenet.h5) |
## EVA models
  - Note: `EvaGiantPatch14` weights are in `float16` format, as `float32` ones are too large that exceed 2GB.

  | Model                 | Params  | FLOPs    | Input | Top1 Acc | Download |
  | --------------------- | ------- | -------- | ----- | -------- | -------- |
  | EvaLargePatch14, 22k  | 304.14M | 61.65G   | 196   | 88.59    | [eva_large_patch14_196.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/eva_large_patch14_196_imagenet21k-ft1k.h5) |
  |                       | 304.53M | 191.55G  | 336   | 89.20    | [eva_large_patch14_336.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/eva_large_patch14_336_imagenet21k-ft1k.h5) |
  | EvaGiantPatch14, clip | 1012.6M | 267.40G  | 224   | 88.82    | [eva_giant_patch14_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/eva_giant_patch14_224_imagenet21k-ft1k.h5) |
  | - m30m                | 1013.0M | 621.45G  | 336   | 89.57    | [eva_giant_patch14_336.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/eva_giant_patch14_336_imagenet21k-ft1k.h5) |
  | - m30m                | 1014.4M | 1911.61G | 560   | 89.80    | [eva_giant_patch14_560.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/eva_giant_patch14_560_imagenet21k-ft1k.h5) |
## EVA02 models
  | Model                                  | Params  | FLOPs   | Input | Top1 Acc | Download |
  | -------------------------------------- | ------- | ------- | ----- | -------- | -------- |
  | EVA02TinyPatch14, mim_in22k_ft1k       | 5.76M   | 4.72G   | 336   | 80.658   | [eva02_tiny_patch14.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/eva02_tiny_patch14_336_mim_in22k_ft1k.h5) |
  | EVA02SmallPatch14, mim_in22k_ft1k      | 22.13M  | 15.57G  | 336   | 85.74    | [eva02_small_patch14.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/eva02_small_patch14_336_mim_in22k_ft1k.h5) |
  | EVA02BasePatch14, mim_in22k_ft22k_ft1k | 87.12M  | 107.6G  | 448   | 88.692   | [eva02_base_patch14.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/eva02_base_patch14_448_mim_in22k_ft22k_ft1k.h5) |
  | EVA02LargePatch14, mim_m38m_ft22k_ft1k | 305.08M | 363.68G | 448   | 90.054   | [eva02_large_patch14.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/eva02_large_patch14_448_mim_m38m_ft22k_ft1k.h5) |
## FlexiViT models
  | Model         | Params  | FLOPs  | Input | Top1 Acc | Download |
  | ------------- | ------- | ------ | ----- | -------- | -------- |
  | FlexiViTSmall | 22.06M  | 5.36G  | 240   | 82.53    | [flexivit_small_240.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/flexivit_small_240_imagenet.h5) |
  | FlexiViTBase  | 86.59M  | 20.33G | 240   | 84.66    | [flexivit_base_240.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/flexivit_base_240_imagenet.h5) |
  | FlexiViTLarge | 304.47M | 71.09G | 240   | 85.64    | [flexivit_large_240.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/flexivit_large_240_imagenet.h5) |
## MetaTransFormer models
  | Model                                 | Params  | FLOPs  | Input | Top1 Acc | Download |
  | ------------------------------------- | ------- | ------ | ----- | -------- | -------- |
  | MetaTransformerBasePatch16, laion_2b  | 86.86M  | 55.73G | 384   | 85.4     | [meta_transformer_base_patch16.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/meta_transformer_base_patch16_384_laion_2b.h5) |
  | MetaTransformerLargePatch14, laion_2b | 304.53M | 191.6G | 336   | 88.1     | [meta_transformer_large_patch14.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/meta_transformer_large_patch14_336_laion_2b.h5) |
## ViT-5 models
  | Model              | Params | FLOPs  | Input | Top1 Acc | Download |
  | ------------------ | ------ | ------ | ----- | -------- | -------- |
  | ViT5_Small_Patch16 | 22.04M | 4.73G  | 224   | 82.2     | [vit5_small_patch16_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/vit5_small_patch16_224_imagenet.h5) |
  | ViT5_Base_Patch16  | 86.54M | 18.00G | 224   | 84.2     | [vit5_base_patch16_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/vit5_base_patch16_224_imagenet.h5) |
  | ViT5_Base_Patch16  | 86.83M | 56.19G | 384   | 85.4     | [vit5_base_patch16_384.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/vit5_base_patch16_384_imagenet.h5) |
  | ViT5_Large_Patch16 | 304.3M | 63.01G | 224   | 84.9     | [vit5_large_patch16_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/vit5_large_patch16_224_imagenet.h5) |
  | ViT5_Large_Patch16 | 304.6M | 193.2G | 384   | 86.0     |          |
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
## FlexiViT / EVA / EVA02 / DINOv2 models with new patch size
  - When setting new `patch_size`, will reload `stem_conv` weights. Source implementation [Github google-research/big_vision/flexi](https://github.com/google-research/big_vision/blob/main/big_vision/models/proj/flexi/vit.py#L30), paper [PDF 2212.08013 FlexiViT: One Model for All Patch Sizes](https://arxiv.org/pdf/2212.08013.pdf).
  - Not works for `BEiT` models.
  ```py
  from keras_cv_attention_models import flexivit, beit, eva, eva02, dinov2
  mm = flexivit.FlexiViTSmall(patch_size=32)  # Original is patch_size=16
  # >>>> Reload mismatched weights: 240 -> (240, 240)
  # >>>> Reload layer: positional_embedding
  # >>>> Reload layer: stem_conv

  from skimage.data import chelsea
  print(mm.decode_predictions(mm(mm.preprocess_input(chelsea())))[0])
  # [('n02124075', 'Egyptian_cat', 0.6939351), ('n02123045', 'tabby', 0.033506528), ...]
  ```
  **Also works for EVA / EVA02 / DINOv2 models**
  ```py
  from keras_cv_attention_models import flexivit, beit, eva, eva02, dinov2
  mm = eva02.EVA02TinyPatch14(patch_size=32)  # Original is patch_size=14
  # >>>> Reload mismatched weights: 336 -> (336, 336)
  # >>>> Reload layer: stem_conv
  # >>>> Reload layer: positional_embedding

  from skimage.data import chelsea
  print(mm.decode_predictions(mm(mm.preprocess_input(chelsea())))[0])
  # [('n02124075', 'Egyptian_cat', 0.76599574), ('n02123045', 'tabby', 0.096458346), ...]
  ```
## Using PyTorch backend
  - Using PyTorch backend by set `KECAM_BACKEND='torch'` environment variable.
  ```py
  os.environ['KECAM_BACKEND'] = 'torch'

  from keras_cv_attention_models import dinov2
  # >>>> Using PyTorch backend

  mm = dinov2.DINOv2_ViT_Small14(patch_size=32, input_shape=(224, 224, 3))  # Original is patch_size=14
  # >>>> Aligned input_shape: [3, 224, 224]
  # >>>> Load pretrained from: ~/.keras/models/dinov2_vit_small14_518_imagenet.h5
  # >>>> Reload mismatched weights: 518 -> (224, 224)
  # >>>> Reload layer: stem_conv
  # >>>> Reload layer: positional_embedding

  from skimage.data import chelsea
  print(mm.decode_predictions(mm(mm.preprocess_input(chelsea())))[0])
  # [('n02124075', 'Egyptian_cat', 0.7959234), ('n02123045', 'tabby', 0.12575167), ...]
  ```
## Custom ViT models load and convert weights from timm torch model
  - `keras_model_load_weights_from_pytorch_model` can be used for loading most `Beit` / `ViT` / `FlexViT` / `EVA` model weights from initialized timm torch model. Outputs is converted weight file, which can be used for afterward and further usage.
  ```py
  """ Build a timm ViT model """
  import timm
  torch_model = timm.models.vit_tiny_patch16_224(pretrained=True)
  _ = torch_model.eval()

  """ Build a ViT model same architecture with torch_model """
  from keras_cv_attention_models import beit
  mm = beit.ViT(depth=12, embed_dim=192, num_heads=3, pretrained=None, classifier_activation=None)
  beit.keras_model_load_weights_from_pytorch_model(mm, torch_model)
  # >>>> Save model to: vit_224.h5
  # >>>> Keras model prediction: [(('n02123045', 'tabby', 12.275056), ('n02124075', 'Egyptian_cat', 12.053913), ...]
  # >>>> Torch model prediction: [[('n02123045', 'tabby', 12.275055), ('n02124075', 'Egyptian_cat', 12.053913), ...]
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
  **EVA02LargePatch14, m38m_medft_in21k_ft_in1k_p14**
  ```py
  inputs = np.random.uniform(size=(1, 448, 448, 3)).astype("float32")

  """ PyTorch eva02_large_patch14_448 """
  sys.path.append('../pytorch-image-models/')
  import timm
  import torch
  cfg = timm.get_pretrained_cfg('eva02_large_patch14_448')
  cfg.hf_hub_id = 'Yuxin-CV/EVA-02'
  cfg.hf_hub_filename = 'eva02/cls/in21k_to_in1k/eva02_L_pt_m38m_medft_in21k_ft_in1k_p14.pt'
  torch_model = timm.models.eva02_large_patch14_448(pretrained=True, pretrained_cfg=cfg)
  # torch_model = timm.models.eva02_base_patch14_448(pretrained=True)
  _ = torch_model.eval()
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()

  """ Keras EVA02BasePatch14 """
  from keras_cv_attention_models import eva02
  mm = eva02.EVA02LargePatch14(classifier_activation=None)
  keras_out = mm(inputs).numpy()

  """ Verification """
  print(f"{np.allclose(torch_out, keras_out, atol=1e-5) = }")
  # np.allclose(torch_out, keras_out, atol=1e-5) = True
  ```
  **ViT5SmallPatch16**

  ```py
  """ PyTorch ViT5 Small Patch 16 """
  sys.path.append('../ViT-5/')
  import torch
  # Monkey-patch cuda() to return the same tensor on CPU if no GPU
  if not torch.cuda.is_available():
      torch.Tensor.cuda = lambda self, *args, **kwargs: self
  import models_vit5 as torch_vit5
  torch_model = torch_vit5.vit5_small()
  ss = torch.load("vit5_small_patch16_224.pth", map_location=torch.device("cpu"))
  torch_model.load_state_dict(ss["model"])
  torch_model.eval()

  """ Keras ViT5 Small Patch 16 """
  from keras_cv_attention_models.beit import vit5
  mm = vit5.ViT5_Small_Patch16(pretrained="vit5_small_patch16_224_imagenet.h5", classifier_activation=None)

  """ Verification """
  inputs = np.random.uniform(size=(1, *mm.input_shape[1:])).astype("float32")
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()
  keras_out = mm(inputs).numpy()
  print(f"{np.allclose(torch_out, keras_out, atol=1e-5) = }")
  # np.allclose(torch_out, keras_out, atol=1e-5) = True
  ```
***
