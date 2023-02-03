# ___Keras VOLO___
***

## Summary
  - Keras implementation of [Github sail-sg/volo](https://github.com/sail-sg/volo). Paper [PDF 2106.13112 VOLO: Vision Outlooker for Visual Recognition](https://arxiv.org/pdf/2106.13112.pdf).
***

## Models
  | Model   | Params | FLOPs   | Input | Top1 Acc | Download            |
  | ------- | ------ | ------- | ----- | -------- | ------------------- |
  | VOLO_d1 | 27M    | 4.82G   | 224   | 84.2     | [volo_d1_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d1_224_imagenet.h5) |
  | - 384   | 27M    | 14.22G  | 384   | 85.2     | [volo_d1_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d1_384_imagenet.h5) |
  | VOLO_d2 | 59M    | 9.78G   | 224   | 85.2     | [volo_d2_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d2_224_imagenet.h5) |
  | - 384   | 59M    | 28.84G  | 384   | 86.0     | [volo_d2_384_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d2_384_imagenet.h5) |
  | VOLO_d3 | 86M    | 13.80G  | 224   | 85.4     | [volo_d3_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d3_224_imagenet.h5) |
  | - 448   | 86M    | 55.50G  | 448   | 86.3     | [volo_d3_448_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d3_448_imagenet.h5) |
  | VOLO_d4 | 193M   | 29.39G  | 224   | 85.7     | [volo_d4_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d4_224_imagenet.h5) |
  | - 448   | 193M   | 117.81G | 448   | 86.8     | [volo_d4_448_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d4_448_imagenet.h5) |
  | VOLO_d5 | 296M   | 53.34G  | 224   | 86.1     | [volo_d5_224_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d5_224_imagenet.h5) |
  | - 448   | 296M   | 213.72G | 448   | 87.0     | [volo_d5_448_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d5_448_imagenet.h5) |
  | - 512   | 296M   | 279.36G | 512   | 87.1     | [volo_d5_512_imagenet.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/volo/volo_d5_512_imagenet.h5) |
## Usage
  ```py
  from keras_cv_attention_models import volo

  # Will download and load pretrained imagenet weights.
  mm = volo.VOLO_d2(input_shape=(384, 384, 3), classfiers=2, num_classes=1000, pretrained="imagenet")

  # Run prediction
  from skimage.data import chelsea
  pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
  print(mm.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 12.834192), ('n02123045', 'tabby', 7.9686913), ...]
  ```
  **Change input resolution** if input_shape is not within pretrained, will load `PositionalEmbedding` weights by `load_resized_weights`.
  ```py
  from keras_cv_attention_models import volo

  # Define model using a new input_shape
  mm = volo.VOLO_d1(input_shape=(512, 512, 3), classfiers=2, num_classes=1000, mix_token=False)
  # >>>> Load pretrained from: ~/.keras/models/volo/volo_d1_384_imagenet.h5
  # WARNING:tensorflow:Skipping loading of weights for layer positional_embedding due to mismatch in shape ((1, 32, 32, 384) vs (1, 24, 24, 384)).
  # >>>> Reload mismatched PositionalEmbedding weights: 384 -> 512

  # Run prediction on Chelsea with (512, 512) resolution
  from skimage.data import chelsea
  pred = mm(mm.preprocess_input(chelsea()))  # Chelsea the cat
  print(mm.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 12.914974), ('n02123045', 'tabby', 9.339718), ...]
  ```
  **Mixup token**
  ```py
  from keras_cv_attention_models import volo
  mm = volo.VOLO_d1(input_shape=(224, 224, 3), classfiers=2, num_classes=1000, mix_token=True, token_label_top=True)

  from skimage.data import chelsea
  out = mm(mm.preprocess_input(chelsea()))
  print(f"{len(out) = }, {out[0].shape = }, {out[1].shape = }")
  # len(out) = 2, out[0].shape = TensorShape([1, 1001]), out[1].shape = TensorShape([1, 196, 1000])
  pred = out[0][:, :-1] + 0.5 * tf.reduce_max(out[1], 1)
  print(mm.decode_predictions(pred.numpy())[0])
  # [('n02124075', 'Egyptian_cat', 12.737301), ('n02123045', 'tabby', 8.866584), ... ]
  ```
## Verification with Pytorch model
  ```py
  """ PyTorch volo """
  sys.path.append('../pytorch-image-models/')
  sys.path.append('../volo')
  import torch
  import models.volo as torch_volo
  from utils import load_pretrained_weights

  model_path = "../models/volo/d5_512_87.07.pth.tar"
  input_shape = 512
  print(f">>>> {model_path = }, {input_shape = }")

  torch_model = torch_volo.volo_d5(img_size=input_shape)
  torch_model.eval()
  load_pretrained_weights(torch_model, model_path, use_ema=False, strict=True, num_classes=1000)

  """ Keras volo """
  from keras_cv_attention_models import volo
  mm = volo.VOLO_d5(input_shape=(512, 512, 3), classfiers=2, num_classes=1000, pretrained="imagenet")

  """ Verification """
  inputs = np.random.uniform(size=(1, input_shape, input_shape, 3)).astype("float32")
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2)).detach().numpy()
  keras_out = mm(inputs).numpy()
  print(f"{np.allclose(torch_out, keras_out, atol=1e-5) = }")
  # np.allclose(torch_out, keras_out, atol=1e-5) = True
  ```
## Transfer learning on cifar10
  - [volo_cifar10.ipynb](https://colab.research.google.com/drive/1-uB8lbVLZi_NJARjm06QzVdPPbrCp0FM?usp=sharing)

  ![](https://user-images.githubusercontent.com/5744524/151657118-b323d4c6-5f08-4965-82cb-17640cef9352.png)
## Training
  - We evaluate our proposed VOLO on the ImageNet dataset. During training, we do not use any extra training data.
  - We use the LV-ViT-S model with Token Labeling as our baseline.
  - We use the AdamW optimizer with a linear learning rate scaling strategy `lr = LRbase × batch_size / 1024` and 5 × 10−2 weight decay rate as suggested by previous work.
  - Stochastic Depth is used.
  - We train our models on the ImageNet dataset for 300 epochs.
  - For data augmentation methods, we use CutOut, RandAug, and the Token Labeling objective with MixToken.
  - We do not use MixUp or CutMix as they conflict with MixToken.

  | Specification    | D1     | D2   | D3   | D4   | D5   |
  | ---------------- | ------ | ---- | ---- | ---- | ---- |
  | MLP Ratio        | 3      | 3    | 3    | 3    | 4    |
  | Parameters       | 27M    | 59M  | 86M  | 193M | 296M |
  | Stoch. Dep. Rate | 0.1    | 0.2  | 0.5  | 0.5  | 0.75 |
  | Crop Ratio       | 0.96   | 0.96 | 0.96 | 1.15 | 1.15 |
  | LRbase           | 1.6e-3 | 1e-3 | 1e-3 | 1e-3 | 8e-4 |
  | weight decay     | 5e-2   | 5e-2 | 5e-2 | 5e-2 | 5e-2 |

  - For finetuning on larger image resolutions, we set the batch size to 512, learning rate to 5e-6, weight decay to 1e-8 and run the models for 30 epochs.
  - Other hyper-parameters are set the same as default.
  - Finetuning requires 2-8 nodes depending on the model size.
***
