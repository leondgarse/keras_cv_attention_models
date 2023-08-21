# ___Keras FastViT___
***

## Summary
  - Keras implementation of [Github apple/ml-fastvit](https://github.com/apple/ml-fastvit). Paper [PDF 2303.14189 FastViT: A Fast Hybrid Vision Transformer using Structural Reparameterization](https://arxiv.org/pdf/2303.14189.pdf).
  - Model weights ported from official publication.
***

## Models
  | Model        | Params | FLOPs | Input | Top1 Acc | Download |
  | ------------ | ------ | ----- | ----- | -------- | -------- |
  | FastViT_T8   | 4.03M  | 0.65G | 256   | 76.2     | [fastvit_t8_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fastvit/fastvit_t8_imagenet.h5) |
  | - distill    | 4.03M  | 0.65G | 256   | 77.2     | [fastvit_t8_distill.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fastvit/fastvit_t8_distill.h5) |
  | FastViT_T12  | 7.55M  | 1.34G | 256   | 79.3     | [fastvit_t12_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fastvit/fastvit_t12_imagenet.h5) |
  | - distill    | 7.55M  | 1.34G | 256   | 80.3     | [fastvit_t12_distill.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fastvit/fastvit_t12_distill.h5) |
  | FastViT_S12  | 9.47M  | 1.74G | 256   | 79.9     | [fastvit_s12_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fastvit/fastvit_s12_imagenet.h5) |
  | - distill    | 9.47M  | 1.74G | 256   | 81.1     | [fastvit_s12_distill.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fastvit/fastvit_s12_distill.h5) |
  | FastViT_SA12 | 11.58M | 1.88G | 256   | 80.9     | [fastvit_sa12_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fastvit/fastvit_sa12_imagenet.h5) |
  | - distill    | 11.58M | 1.88G | 256   | 81.9     | [fastvit_sa12_distill.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fastvit/fastvit_sa12_distill.h5) |
  | FastViT_SA24 | 21.55M | 3.66G | 256   | 82.7     | [fastvit_sa24_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fastvit/fastvit_sa24_imagenet.h5) |
  | - distill    | 21.55M | 3.66G | 256   | 83.4     | [fastvit_sa24_distill.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fastvit/fastvit_sa24_distill.h5) |
  | FastViT_SA36 | 31.53M | 5.44G | 256   | 83.6     | [fastvit_sa36_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fastvit/fastvit_sa36_imagenet.h5) |
  | - distill    | 31.53M | 5.44G | 256   | 84.2     | [fastvit_sa36_distill.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fastvit/fastvit_sa36_distill.h5) |
  | FastViT_MA36 | 44.07M | 7.64G | 256   | 83.9     | [fastvit_ma36_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fastvit/fastvit_ma36_imagenet.h5) |
  | - distill    | 44.07M | 7.64G | 256   | 84.6     | [fastvit_ma36_distill.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/fastvit/fastvit_ma36_distill.h5) |
## Usage
  ```py
  from keras_cv_attention_models import fastvit, test_images

  # Will download and load pretrained imagenet weights.
  mm = fastvit.FastViT_T8(pretrained="distill")

  # Run prediction
  preds = mm(mm.preprocess_input(test_images.cat()))
  print(mm.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.8990752), ('n02123045', 'tabby', 0.013779595), ...]
  ```
  **Change input resolution** by set new `input_shape`, or use dynamic input resolution by set `input_shape=(None, None, 3)`.
  ```py
  from keras_cv_attention_models import fastvit, test_images
  mm = fastvit.FastViT_SA12(pretrained="imagenet", input_shape=(219, 112, 3))
  # Run prediction
  preds = mm(mm.preprocess_input(test_images.cat()))
  print(mm.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.94039464), ('n02123159', 'tiger_cat', 0.0059115295), ...]
  ```
  ```py
  from keras_cv_attention_models import fastvit, test_images
  mm = fastvit.FastViT_T8(pretrained="distill", input_shape=(None, None, 3))
  # Run prediction
  preds = mm(mm.preprocess_input(test_images.cat(), input_shape=(219, 112, 3)))
  print(mm.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.9374073), ('n03942813', 'ping-pong_ball', 0.019263275), ...]
  ```
  **Switch to deploy** by calling `model.switch_to_deploy()`, will fuse reparameter block into a single `Conv2D` layer, **by calling `fuse_reparam_blocks` 3 times**, and apply `convert_to_fused_conv_bn_model` that fusing `Conv2D->BatchNorm`.
  ```py
  from keras_cv_attention_models import fastvit, test_images, model_surgery

  mm = fastvit.FastViT_SA12(pretrained="imagenet")
  model_surgery.count_params(mm)
  # Total params: 11,616,296 | Trainable params: 11,580,968 | Non-trainable params:35,328
  preds = mm(mm.preprocess_input(test_images.cat()))

  """ switch_to_deploy """
  bb = mm.switch_to_deploy()
  model_surgery.count_params(bb)
  # Total params: 11,540,456 | Trainable params: 11,538,408 | Non-trainable params:2,048
  preds_deploy = bb(bb.preprocess_input(test_images.cat()))

  print(f"{np.allclose(preds, preds_deploy, atol=1e-5) = }")
  # np.allclose(preds, preds_deploy, atol=1e-5) = True

  """ save and load weights using deploy=True """
  bb.save("aa.h5")
  cc = fastvit.FastViT_SA12(pretrained="aa.h5", deploy=True)
  print(f"{np.allclose(preds_deploy, cc(cc.preprocess_input(test_images.cat())), atol=1e-7) = }")
  # np.allclose(preds_deploy, cc(cc.preprocess_input(test_images.cat())), atol=1e-7) = True
  ```
  **Using PyTorch backend** by set `KECAM_BACKEND='torch'` environment variable.
  ```py
  os.environ['KECAM_BACKEND'] = 'torch'
  from keras_cv_attention_models import fastvit, test_images
  mm = fastvit.FastViT_T8(pretrained="distill", input_shape=(219, 112, 3))
  # >>>> Using PyTorch backend
  # >>>> Load pretrained from: ~/.keras/models/fastvit_t8_distill.h5

  # Run prediction
  preds = mm(mm.preprocess_input(test_images.cat()))
  print(mm.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.937407), ('n03942813', 'ping-pong_ball', 0.019263512), ...]
  ```
## Verification with PyTorch version
  ```py
  """ PyTorch fastvit_sa12 """
  sys.path.append('../ml-fastvit/')
  sys.path.append('../pytorch-image-models/')  # Needs timm
  import torch
  import models as torch_fastvit

  torch_model = torch_fastvit.fastvit_sa12()
  ss = torch.load('fastvit_sa12.pth.tar', map_location=torch.device('cpu'))
  torch_model.load_state_dict(ss.get('state_dict', ss))
  _ = torch_model.eval()

  """ Keras FastViT_SA12 """
  from keras_cv_attention_models import fastvit
  mm = fastvit.FastViT_SA12(pretrained="imagenet", classifier_activation=None)

  """ Verification """
  inputs = np.random.uniform(size=(1, *mm.input_shape[1:3], 3)).astype("float32")
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()
  keras_out = mm(inputs).numpy()
  print(f"{np.allclose(torch_out, keras_out, atol=1e-4) = }")
  # np.allclose(torch_out, keras_out, atol=1e-4) = True
  ```
