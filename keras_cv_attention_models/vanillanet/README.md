# ___Keras VanillaNet___
***

## Summary
  - Keras implementation of [Github huawei-noah/VanillaNet](https://github.com/huawei-noah/VanillaNet). Paper [PDF 2305.12972 VanillaNet: the Power of Minimalism in Deep Learning](https://arxiv.org/pdf/2305.12972.pdf).
  - Model weights ported from official publication.
***

## Models
  | Model         | Params | FLOPs  | Input | Top1 Acc | Download |
  | ------------- | ------ | ------ | ----- | -------- | -------- |
  | VanillaNet5   | 22.33M | 8.46G  | 224   | 72.49    | [vanillanet_5_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/vanillanet/vanillanet_5_imagenet.h5) |
  | - deploy=True | 15.52M | 5.17G  | 224   | 72.49    | [vanillanet_5_deploy_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/vanillanet/vanillanet_5_deploy_imagenet.h5) |
  | VanillaNet6   | 56.12M | 10.11G | 224   | 76.36    | [vanillanet_6_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/vanillanet/vanillanet_6_imagenet.h5) |
  | - deploy=True | 32.51M | 6.00G  | 224   | 76.36    | [vanillanet_6_deploy_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/vanillanet/vanillanet_6_deploy_imagenet.h5) |
  | VanillaNet7   | 56.67M | 11.84G | 224   | 77.98    | [vanillanet_7_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/vanillanet/vanillanet_7_imagenet.h5) |
  | - deploy=True | 32.80M | 6.90G  | 224   | 77.98    | [vanillanet_7_deploy_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/vanillanet/vanillanet_7_deploy_imagenet.h5) |
  | VanillaNet8   | 65.18M | 13.50G | 224   | 79.13    | [vanillanet_8_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/vanillanet/vanillanet_8_imagenet.h5) |
  | - deploy=True | 37.10M | 7.75G  | 224   | 79.13    | [vanillanet_8_deploy_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/vanillanet/vanillanet_8_deploy_imagenet.h5) |
  | VanillaNet9   | 73.68M | 15.17G | 224   | 79.87    | [vanillanet_9_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/vanillanet/vanillanet_9_imagenet.h5) |
  | - deploy=True | 41.40M | 8.59G  | 224   | 79.87    | [vanillanet_9_deploy_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/vanillanet/vanillanet_9_deploy_imagenet.h5) |
  | VanillaNet10  | 82.19M | 16.83G | 224   | 80.57    | [vanillanet_10_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/vanillanet/vanillanet_10_imagenet.h5) |
  | - deploy=True | 45.69M | 9.43G  | 224   | 80.57    | [vanillanet_10_deploy_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/vanillanet/vanillanet_10_deploy_imagenet.h5) |
  | VanillaNet11  | 90.69M | 18.49G | 224   | 81.08    |          |
  | - deploy=True | 50.00M | 10.27G | 224   | 81.08    |          |
  | VanillaNet12  | 99.20M | 20.16G | 224   | 81.55    |          |
  | - deploy=True | 54.29M | 11.11G | 224   | 81.55    |          |
  | VanillaNet13  | 107.7M | 21.82G | 224   | 82.05    |          |
  | - deploy=True | 58.59M | 11.96G | 224   | 82.05    |          |
## Usage
  ```py
  from keras_cv_attention_models import vanillanet, test_images

  # Will download and load pretrained imagenet weights.
  model = vanillanet.VanillaNet5(pretrained="imagenet")
  # >>>> Load pretrained from: ~/.keras/models/vanillanet_5_imagenet.h5

  # Run prediction
  preds = model(model.preprocess_input(test_images.cat()))
  print(model.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.9908214), ('n02123045', 'tabby', 0.008346258), ...]
  ```
  **Set `deploy=True`** for a fused model, and keep output same. **Evaluation only, not good for training**.
  ```py
  from keras_cv_attention_models import vanillanet, test_images
  model = vanillanet.VanillaNet5(deploy=True, pretrained="imagenet")
  # >>>> Load pretrained from: ~/.keras/models/vanillanet_5_deploy_imagenet.h5

  preds = model(model.preprocess_input(test_images.cat()))
  print(model.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.9908214), ('n02123045', 'tabby', 0.008346282), ...]
  ```
  **Use dynamic input resolution** by set `input_shape=(None, None, 3)`.
  ```py
  from keras_cv_attention_models import vanillanet
  model = vanillanet.VanillaNet6(input_shape=(None, None, 3), num_classes=0)
  # >>>> Load pretrained from: ~/.keras/models/vanillanet_6_imagenet.h5
  print(model.output_shape)
  # (None, None, None, 4096)

  print(model(np.ones([1, 223, 123, 3])).shape)
  # (1, 6, 3, 4096)
  print(model(np.ones([1, 32, 526, 3])).shape)
  # (1, 1, 16, 4096)
  ```
  **Using PyTorch backend** by set `KECAM_BACKEND='torch'` environment variable.
  ```py
  os.environ['KECAM_BACKEND'] = 'torch'

  from keras_cv_attention_models import vanillanet, test_images
  model = vanillanet.VanillaNet6(deploy=True)
  # >>>> Using PyTorch backend
  # >>>> Aligned input_shape: [3, 224, 224]
  # >>>> Load pretrained from: ~/.keras/models/vanillanet_6_deploy_imagenet.h5

  preds = model(model.preprocess_input(test_images.cat()))
  print(model.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.9729379), ('n02123045', 'tabby', 0.008538764), ...]
  ```
## Train and Deploy
  - **Currently only works for Tensorflow**
  - **`set_leaky_relu_alpha`** is introduced in the paper, used for changing activation alpha while training. Say `epochs=300, decay_epochs=100`, then `alpha = (current_epoch / decay_epochs) if current_epoch < decay_epochs else 1`.
  - **`switch_to_deploy`** is used for converting a custom pretrained model to `deploy` version. The process is `fuse conv bn` -> `remove leaky_relu with alpha=1` -> `fuse 2 sequential conv`.
  - Training process could be:
    ```py
    from keras_cv_attention_models import vanillanet

    """ Create a `deploy=False` model for custom training """
    model = vanillanet.VanillaNet5(deploy=False, pretrained="imagenet")
    print(f"{model.count_params() = }")
    # model.count_params() = 22369712
    # ...

    """ Run some training step with changing leaky_relu alpha """
    decay_epochs = 100
    current_epoch = 10
    model.set_leaky_relu_alpha(alpha=(current_epoch / decay_epochs) if current_epoch < decay_epochs else 1)

    # Or set in actual callbacks
    # on_epoch_end = lambda epoch, logs: model.set_leaky_relu_alpha(alpha=(epoch / decay_epochs) if epoch < decay_epochs else 1)
    # act_learn = keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)
    # model.fit(..., callbacks=[..., act_learn])
    # ...

    """ Fuse to deploy model """
    model.set_leaky_relu_alpha(alpha=1)  # This should already set when training reach `decay_epochs`
    tt = model.switch_to_deploy()
    deploy_save_name = model.name + "_deploy.h5"
    tt.save(deploy_save_name)

    """ Further evaluation with deploy=True """
    bb = vanillanet.VanillaNet5(deploy=True, pretrained=deploy_save_name)
    print(f"{bb.count_params() = }")
    # bb.count_params() = 15523304
    print(f"{np.allclose(model(tf.ones([1, 224, 224, 3])), bb(tf.ones([1, 224, 224, 3])), atol=1e-5) = }")
    # np.allclose(model(tf.ones([1, 224, 224, 3])), bb(tf.ones([1, 224, 224, 3])), atol=1e-5) = True
    ```
## Verification with PyTorch version
  ```py
  """ PyTorch vanillanet_6 """
  sys.path.append('../VanillaNet/')
  sys.path.append('../pytorch-image-models/')  # Needs timm
  import torch
  from models import vanillanet as torch_vanillanet

  torch_model = torch_vanillanet.vanillanet_6()
  _ = torch_model.eval()
  ss = torch.load('vanillanet_6.pth'.format(id), map_location='cpu')
  torch_model.load_state_dict(ss['model_ema'])

  """ Keras VanillaNet6 """
  from keras_cv_attention_models import vanillanet
  mm = vanillanet.VanillaNet6(pretrained="imagenet", classifier_activation=None)

  """ Verification """
  inputs = np.random.uniform(size=(1, *mm.input_shape[1:3], 3)).astype("float32")
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()
  keras_out = mm(inputs).numpy()
  print(f"{np.allclose(torch_out, keras_out, atol=1e-5) = }")
  # np.allclose(torch_out, keras_out, atol=1e-5) = True
  ```
