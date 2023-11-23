
# ___Keras RepViT___
***

## Summary
  - Keras implementation of [Github THU-MIG/RepViT](https://github.com/THU-MIG/RepViT). Paper [PDF 2307.09283 RepViT: Revisiting Mobile CNN From ViT Perspective](https://arxiv.org/pdf/2307.09283.pdf).
  - Model weights ported from official publication.
## Models
  | Model                    | Params | FLOPs | Input | Top1 Acc | Download |
  | ------------------------ | ------ | ----- | ----- | -------- | -------- |
  | RepViT_M09, distillation | 5.10M  | 0.82G | 224   | 79.1     | [repvit_m09_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/repvit_m_09_imagenet.h5) |
  | - deploy=True            | 5.07M  | 0.82G | 224   | 79.1     |          |
  | RepViT_M10, distillation | 6.85M  | 1.12G | 224   | 80.3     | [repvit_m_10_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/repvit_m_10_imagenet.h5) |
  | - deploy=True            | 6.81M  | 1.12G | 224   | 80.3     |          |
  | RepViT_M11, distillation | 8.29M  | 1.35G | 224   | 81.2     | [repvit_m_11_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/repvit_m_11_imagenet.h5) |
  | - deploy=True            | 8.24M  | 1.35G | 224   | 81.2     |          |
  | RepViT_M15, distillation | 14.13M | 2.30G | 224   | 82.5     | [repvit_m_15_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/repvit_m_15_imagenet.h5) |
  | - deploy=True            | 14.05M | 2.30G | 224   | 82.5     |          |
  | RepViT_M23, distillation | 23.01M | 4.55G | 224   | 83.7     | [repvit_m_23_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/repvit_m_23_imagenet.h5) |
  | - deploy=True            | 22.93M | 4.55G | 224   | 83.7     |          |
## Usage
  ```py
  from keras_cv_attention_models import repvit, test_images

  # Will download and load pretrained imagenet weights.
  mm = repvit.RepViT_M09(pretrained="imagenet", use_distillation=False)

  # Run prediction
  preds = mm(mm.preprocess_input(test_images.cat()))
  print(mm.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.82435805), ('n02123045', 'tabby', 0.033731826), ...]
  ```
  set `use_distillation=True` for adding an additional `BatchNorm->Dense` distill_head block, will also load distill head weights.
  ```py
  from keras_cv_attention_models import repvit, test_images
  from keras_cv_attention_models.backend import functional

  mm = repvit.RepViT_M09(use_distillation=True, classifier_activation=None)
  print(mm.output_names, mm.output_shape)
  # ['head', 'distill_head'] [(None, 1000), (None, 1000)]

  # Run prediction
  preds = mm(mm.preprocess_input(test_images.cat()))
  preds = functional.softmax((preds[0] + preds[1]) / 2)
  print(mm.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.979251), ('n02123045', 'tabby', 0.008092029), ...]
  ```
  **Use dynamic input resolution** by set `input_shape=(None, None, 3)`.
  ```py
  from keras_cv_attention_models import repvit
  # Will download and load pretrained imagenet weights.
  mm = repvit.RepViT_M09(input_shape=(None, None, 3), use_distillation=False, num_classes=0)
  print(mm.output_shape)
  # (None, None, None, 384)

  print(mm(np.ones([1, 223, 123, 3])).shape)
  # (1, 7, 4, 384)
  print(mm(np.ones([1, 32, 526, 3])).shape)
  # (1, 1, 17, 384)
  ```
  **Switch to deploy** by calling `model.switch_to_deploy()`, will fuse reparameter block into a single `Conv2D` layer, and fuse distillation header `BatchNorm-> Dense` and preciction header `BatchNorm-> Dense` into a single `Dense` layer. Also applying `convert_to_fused_conv_bn_model` that fusing `Conv2D->BatchNorm`.
  ```py
  from keras_cv_attention_models import repvit, test_images, model_surgery

  mm = repvit.RepViT_M09(pretrained="imagenet", use_distillation=True, classifier_activation=None)
  model_surgery.count_params(mm)
  # Total params: 5,537,856 | Trainable params: 5,489,328 | Non-trainable params:48,528
  preds = mm(mm.preprocess_input(test_images.cat()))

  """ switch_to_deploy """
  bb = mm.switch_to_deploy()
  model_surgery.count_params(bb)
  # Total params: 5,067,056 | Trainable params: 5,067,056 | Non-trainable params:0
  preds_deploy = bb(bb.preprocess_input(test_images.cat()))

  print(f"{np.allclose((preds[0] + preds[1]) / 2, preds_deploy, atol=1e-5) = }")
  # np.allclose((preds[0] + preds[1]) / 2, preds_deploy, atol=1e-5) = True

  """ save and load weights using deploy=True """
  bb.save("aa.h5")
  cc = repvit.RepViT_M09(pretrained=None, deploy=True, classifier_activation=None)
  cc.load_weights("aa.h5")
  print(f"{np.allclose(preds_deploy, cc(cc.preprocess_input(test_images.cat())), atol=1e-7) = }")
  # np.allclose(preds_deploy, cc(cc.preprocess_input(test_images.cat())), atol=1e-7) = True
  ```
  **Using PyTorch backend** by set `KECAM_BACKEND='torch'` environment variable.
  ```py
  os.environ['KECAM_BACKEND'] = 'torch'

  from keras_cv_attention_models import repvit
  model = repvit.RepViT_M09(input_shape=(None, None, 3), num_classes=0)
  # >>>> Load pretrained from: ~/.keras/models/repvit_m_09_imagenet.h5
  print(model.output_shape)
  # (None, 384, None, None)

  import torch
  print(model(torch.ones([1, 3, 223, 123])).shape)
  # torch.Size([1, 384, 7, 4])
  print(model(torch.ones([1, 3, 32, 526])).shape)
  # torch.Size([1, 384, 1, 17])
  ```  
## Verification with PyTorch version
  ```py
  """ PyTorch repvit_m1 """
  sys.path.append('../pytorch-image-models/')  # Needs timm
  sys.path.append('../RepViT/')
  import torch
  from model import repvit as torch_repvit
  torch_model = torch_repvit.repvit_m0_9(pretrained=True, distillation=True)
  ss = torch.load('repvit_m0_9_distill_450.pth', map_location=torch.device('cpu'))
  torch_model.load_state_dict(ss['model'])
  torch_model.eval()

  """ Keras RepViT_M09 """
  from keras_cv_attention_models import repvit
  mm = repvit.RepViT_M09(pretrained="imagenet", use_distillation=True, classifier_activation=None)

  """ Verification """
  inputs = np.random.uniform(size=(1, *mm.input_shape[1:])).astype("float32")
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()
  pred = mm(inputs)
  keras_out = ((pred[0] + pred[1]) / 2).numpy()
  print(f"{np.allclose(torch_out, keras_out, atol=1e-5) = }")
  # np.allclose(torch_out, keras_out, atol=1e-5) = True
  ```
***
