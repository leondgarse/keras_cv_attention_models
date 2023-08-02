
# ___Keras RepViT___
***

## Summary
  - Keras implementation of [Github THU-MIG/RepViT](https://github.com/THU-MIG/RepViT). Paper [PDF 2307.09283 RepViT: Revisiting Mobile CNN From ViT Perspective](https://arxiv.org/pdf/2307.09283.pdf).
  - Model weights ported from official publication.
## Models
  | Model                   | Params | FLOPs | Input | Top1 Acc | Download |
  | ----------------------- | ------ | ----- | ----- | -------- | -------- |
  | RepViT_M1, distillation | 5.10M  | 0.82G | 224   | 78.5     | [repvit_m1_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/repvit_m1_imagenet.h5) |
  | - switch_to_deploy      | 5.07M  | 0.82G | 224   | 78.5     |          |
  | RepViT_M2, distillation | 8.28M  | 1.35G | 224   | 80.6     | [repvit_m2_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/repvit_m2_imagenet.h5) |
  | - switch_to_deploy      | 8.25M  | 1.35G | 224   | 80.6     |          |
  | RepViT_M3, distillation | 10.2M  | 1.87G | 224   | 81.4     | [repvit_m3_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/levit/repvit_m3_imagenet.h5) |
  | - switch_to_deploy      | 10.12M | 1.87G | 224   | 81.4     |          |
## Usage
  ```py
  from keras_cv_attention_models import repvit, test_images

  # Will download and load pretrained imagenet weights.
  mm = repvit.RepViT_M1(pretrained="imagenet", use_distillation=False)

  # Run prediction
  preds = mm(mm.preprocess_input(test_images.cat()))
  print(mm.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.9019543), ('n02123045', 'tabby', 0.012886077), ...]
  ```
  set `use_distillation=True` for adding an additional `BatchNorm->Dense` distill_head block, will also load distill head weights.
  ```py
  from keras_cv_attention_models import repvit, test_images
  from keras_cv_attention_models.backend import functional

  mm = repvit.RepViT_M1(use_distillation=True, classifier_activation=None)
  print(mm.output_names, mm.output_shape)
  # ['head', 'distill_head'] [(None, 1000), (None, 1000)]

  # Run prediction
  preds = mm(mm.preprocess_input(test_images.cat()))
  preds = functional.softmax((preds[0] + preds[1]) / 2)
  print(mm.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.9925538), ('n02123045', 'tabby', 0.002153618), ...]
  ```
  **Use dynamic input resolution** by set `input_shape=(None, None, 3)`.
  ```py
  from keras_cv_attention_models import repvit
  # Will download and load pretrained imagenet weights.
  mm = repvit.RepViT_M1(input_shape=(None, None, 3), use_distillation=False, num_classes=0)
  print(mm.output_shape)
  # (None, None, None, 384)

  print(mm(np.ones([1, 223, 123, 3])).shape)
  # (1, 7, 4, 384)
  print(mm(np.ones([1, 32, 526, 3])).shape)
  # (1, 1, 17, 384)
  ```
  **Switch to deploy** by calling `model.switch_to_deploy()`, will fuse distillation header `BatchNorm-> Dense` and preciction header `BatchNorm-> Dense` into a single `Dense` layer. Also applying `convert_to_fused_conv_bn_model` that fusing `Conv2D->BatchNorm`.
  ```py
  from keras_cv_attention_models import repvit, test_images, model_surgery

  mm = repvit.RepViT_M1(pretrained="imagenet", use_distillation=True, classifier_activation=None)
  model_surgery.count_params(mm)
  # Total params: 5,533,776 | Trainable params: 5,485,248 | Non-trainable params:48,528
  preds = mm(mm.preprocess_input(test_images.cat()))

  bb = mm.switch_to_deploy()
  model_surgery.count_params(bb)
  # Total params: 5,067,056 | Trainable params: 5,067,056 | Non-trainable params:0
  preds_deploy = bb(bb.preprocess_input(test_images.cat()))

  print(f"{np.allclose(tf.reduce_mean(preds, axis=0), preds_deploy, atol=1e-5) = }")
  # np.allclose(tf.reduce_mean(preds, axis=0), preds_deploy, atol=1e-5) = True
  ```
  **Using PyTorch backend** by set `KECAM_BACKEND='torch'` environment variable.
  ```py
  os.environ['KECAM_BACKEND'] = 'torch'

  from keras_cv_attention_models import repvit
  model = repvit.RepViT_M1(input_shape=(None, None, 3), num_classes=0)
  # >>>> Load pretrained from: ~/.keras/models/repvit_m1_imagenet.h5
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
  torch_model = torch_repvit.repvit_m1(pretrained=True, distillation=True)
  ss = torch.load('repvit_m1_distill_300.pth', map_location=torch.device('cpu'))
  torch_model.load_state_dict(ss['model'])
  torch_model.eval()

  """ Keras RepViT_M1 """
  from keras_cv_attention_models import repvit
  mm = repvit.RepViT_M1(pretrained="imagenet", use_distillation=True, classifier_activation=None)

  """ Verification """
  inputs = np.random.uniform(size=(1, *mm.input_shape[1:])).astype("float32")
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()
  pred = mm(inputs)
  keras_out = ((pred[0] + pred[1]) / 2).numpy()
  print(f"{np.allclose(torch_out, keras_out, atol=1e-5) = }")
  # np.allclose(torch_out, keras_out, atol=1e-5) = True
  ```
***
