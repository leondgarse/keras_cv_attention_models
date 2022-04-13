# Keras_CMT
***

## Summary
  - CMT article: [PDF 2107.06263 CMT: Convolutional Neural Networks Meet Vision Transformers](https://arxiv.org/pdf/2107.06263.pdf)
  - [Github wilile26811249/CMT_CNN-meet-Vision-Transformer](https://github.com/wilile26811249/CMT_CNN-meet-Vision-Transformer)

  ![](https://user-images.githubusercontent.com/5744524/151656779-6e6f2203-a7f7-42cf-8833-f4d472c171ae.png)
***

## Usage
  ```py
  from keras_cv_attention_models import fbnetv3
  mm = fbnetv3.FBNetV3B()

  # Run prediction
  import tensorflow as tf
  from skimage.data import chelsea
  imm = tf.keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(tf.keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.5725908), ('n02123159', 'tiger_cat', 0.15323903), ...]
  ```
  **Use dynamic input resolution** by set `input_shape=(None, None, 3)`.
  ```py
  from keras_cv_attention_models import tinynet
  model = tinynet.TinyNetB(input_shape=(None, None, 3))

  from skimage.data import chelsea
  preds = model(model.preprocess_input(chelsea(), input_shape=[234, 345, 3]))
  print(model.decode_predictions(preds))
  # [('n02124075', 'Egyptian_cat', 0.4603405), ('n02123045', 'tabby', 0.34186444), ...]
  ```
## FBNetV3
  - [PDF 2006.02049 FBNetV3: Joint Architecture-Recipe Search using Predictor Pretraining](https://arxiv.org/pdf/2006.02049.pdf).
  - Model structure and weights reloaded from [timm/mobilenetv3](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/mobilenetv3.py).

  | Model    | Params | Image resolution | Top1 Acc | Download |
  | -------- | ------ | ---------------- | -------- | -------- |
  | FBNetV3B | 5.57M  | 256              | 79.15    | [fbnetv3_b_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/fbnetv3_b_imagenet.h5) |
  | FBNetV3D | 10.31M | 256              | 79.68    | [fbnetv3_d_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/fbnetv3_d_imagenet.h5) |
  | FBNetV3G | 16.62M | 256              | 82.05    | [fbnetv3_g_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/fbnetv3_g_imagenet.h5) |
## LCNet
  - [PDF 2109.15099 PP-LCNet: A Lightweight CPU Convolutional Neural Network](https://arxiv.org/pdf/2109.15099.pdf).
  - Model original weights from [Github PaddlePaddle/PaddleClas](https://github.com/PaddlePaddle/PaddleClas).

  | Model    | Params | Image resolution | Top1 Acc | Download |
  | -------- | ------ | ---------------- | -------- | -------- |
  | LCNet050 | 1.88M  | 224              | 63.10    | [lcnet_050_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/lcnet_050_imagenet.h5) |
  | LCNet075 | 2.36M  | 224              | 68.82    | [lcnet_075_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/lcnet_075_imagenet.h5) |
  | LCNet100 | 2.95M  | 224              | 72.10    | [lcnet_100_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/lcnet_100_imagenet.h5) |
## MobileNetV3
  - [PDF 1905.02244 Searching for MobileNetV3](https://arxiv.org/pdf/1905.02244.pdf).
  - Model structure and weights reloaded from [timm/mobilenetv3](https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/mobilenetv3.py).
  - `MobilenetV3Large075` weights reloaded from `keras.applications.MobileNetV3Large(alpha=0.75, weights='imagenet')`.
  - `miil` model original weights from [Github Alibaba-MIIL/ImageNet21K](https://github.com/Alibaba-MIIL/ImageNet21K). Specify `pretrained="miil_21k"` for not imagenet fine-tuned imagenet21k weights.

  | Model               | Params | Image resolution | Top1 Acc | Download |
  | ------------------- | ------ | ---------------- | -------- | -------- |
  | MobileNetV3Small050 | 1.29M  | 224              | 57.89    | [small_050_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/mobilenetv3_small_050_imagenet.h5) |
  | MobileNetV3Small075 | 2.04M  | 224              | 65.24    | [small_075_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/mobilenetv3_small_075_imagenet.h5) |
  | MobileNetV3Small100 | 2.54M  | 224              | 67.66    | [small_100_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/mobilenetv3_small_100_imagenet.h5) |
  | MobileNetV3Large075 | 3.99M  | 224              | 73.44    | [large_075_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/mobilenetv3_large_075_imagenet.h5) |
  | MobileNetV3Large100 | 5.48M  | 224              | 75.77    | [large_100_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/mobilenetv3_large_100_imagenet.h5) |
  | - miil              | 5.48M  | 224              | 77.92    | [large_100_miil.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/mobilenetv3_large_100_mill.h5) |
## TinyNet
  - [PDF 2010.14819 Model Rubik’s Cube: Twisting Resolution, Depth and Width for TinyNets](https://arxiv.org/pdf/2010.14819.pdf).
  - Model original weights from [Github huawei-noah/CV-Backbones/tinynet_pytorch](https://github.com/huawei-noah/CV-Backbones/tree/master/tinynet_pytorch).

  | Model    | Params | Image resolution | Top1 Acc | Download |
  | -------- | ------ | ---------------- | -------- | -------- |
  | TinyNetE | 2.04M  | 106              | 59.86    | [tinynet_e_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/tinynet_e_imagenet.h5) |
  | TinyNetD | 2.34M  | 152              | 66.96    | [tinynet_d_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/tinynet_d_imagenet.h5) |
  | TinyNetC | 2.46M  | 184              | 71.23    | [tinynet_c_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/tinynet_c_imagenet.h5) |
  | TinyNetB | 3.73M  | 188              | 74.98    | [tinynet_b_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/tinynet_b_imagenet.h5) |
  | TinyNetA | 6.19M  | 192              | 77.65    | [tinynet_a_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/mobilenetv3_family/tinynet_a_imagenet.h5) |
## Verification with PyTorch version
  ```py
  inputs = np.random.uniform(size=(1, 224, 224, 3)).astype("float32")

  """ PyTorch lcnet_050 """
  sys.path.append("../pytorch-image-models")
  import timm
  import torch
  torch_model = timm.models.lcnet_050(pretrained=True)
  _ = torch_model.eval()
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()

  """ Keras LCNet050 """
  from keras_cv_attention_models import lcnet
  mm = lcnet.LCNet050(pretrained="imagenet", classifier_activation=None)
  keras_out = mm(inputs).numpy()

  """ Verification """
  print(f"{np.allclose(torch_out, keras_out, atol=1e-4) = }")
  # np.allclose(torch_out, keras_out, atol=1e-4) = True
  ```
***
