# ___Keras Stable Diffusion___
***

## Summary
  - [PDF 2006.11239 Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
  - [PDF 2112.10752 High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/pdf/2112.10752.pdf)
  - [Github CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
  - [Github runwayml/stable-diffusion](https://github.com/runwayml/stable-diffusion)
  - [Github stability-ai/stablediffusion](https://github.com/stability-ai/stablediffusion)
  - [Github labmlai/annotated_deep_learning_paper_implementations/stable_diffusion](https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master/labml_nn/diffusion/stable_diffusion)
  - [Huggingface runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
  - [The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/)
  - Model weights ported from [Github runwayml/stable-diffusion](https://github.com/runwayml/stable-diffusion) `sd-v1-5.ckpt`
## Models
  | Model               | Params | FLOPs   | Input               | Download            |
  | ------------------- | ------ | ------- | ------------------- | ------------------- |
  | ViTTextLargePatch14 | 123.1M | 6.67G   | [None, 77]          | [vit_text_large_patch14_clip.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/vit_text_large_patch14_clip.h5) |
  | Encoder             | 34.16M | 559.6G  | [None, 512, 512, 3] | [encoder_v1_5.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/stable_diffusion/encoder_v1_5.h5) |
  | UNet                | 859.5M | 404.4G  | [None, 64, 64, 4]   | [unet_v1_5.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/stable_diffusion/unet_v1_5.h5) |
  | Decoder             | 49.49M | 1259.5G | [None, 64, 64, 4]   | [decoder_v1_5.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/stable_diffusion/decoder_v1_5.h5) |

  **Note: `StableDiffusion` is not defined a model. just a gather of above models and forward functions like `text_to_image`.**
## Usage
  ```py
  import tensorflow as tf
  if len(tf.config.experimental.get_visible_devices('GPU')) > 0:
      tf.keras.mixed_precision.set_global_policy("mixed_float16")

  from keras_cv_attention_models import stable_diffusion
  mm = stable_diffusion.StableDiffusion()
  imm = mm.text_to_image('Cyberpunk cityscape with towering skyscrapers, neon signs, and flying cars.', batch_size=4).numpy()

  print(f"{imm.shape = }, {imm.min() = }, {imm.max() = }")
  # imm.shape = (4, 512, 512, 3), imm.min() = -2.4545908, imm.max() = 1.851803
  plt.imshow(np.hstack(np.clip(imm.astype("float32") / 2 + 0.5, 0, 1)))
  ```
  ![stabel_diffusion](https://github.com/leondgarse/keras_cv_attention_models/assets/5744524/e565c750-f98a-4d04-a280-0d0aa382ef5f)

  **Change to other shape** by setting `image_shape`. Should be divisible by `64`, as UNet needs to concatenate down and up samples
  ```py
  import tensorflow as tf
  if len(tf.config.experimental.get_visible_devices('GPU')) > 0:
      tf.keras.mixed_precision.set_global_policy("mixed_float16")

  from keras_cv_attention_models import stable_diffusion
  mm = stable_diffusion.StableDiffusion()
  prompt = 'mountains, stars and paisley fileed sky, artstation, digital painting, sharp focus.'
  imm = mm.text_to_image(prompt=prompt, image_shape=(512, 1024)).numpy()

  print(f"{imm.shape = }, {imm.min() = }, {imm.max() = }")
  # imm.shape = (1, 512, 1024, 3), imm.min() = -1.5322105, imm.max() = 1.419162
  plt.imsave('aa.jpg', np.hstack(np.clip(imm.astype("float32") / 2 + 0.5, 0, 1)))
  ```
  ![stable_diffusion_512_1024](https://github.com/leondgarse/keras_cv_attention_models/assets/5744524/a10e3b97-38b5-4993-92ff-98f05ac0055d)

  **Using PyTorch backend** by set `KECAM_BACKEND='torch'` environment variable.
  ```py
  os.environ['KECAM_BACKEND'] = 'torch'
  import torch
  from contextlib import nullcontext
  device = torch.device("cuda:0") if torch.cuda.is_available() and int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")) >= 0 else torch.device("cpu")
  global_context = nullcontext() if device.type == "cpu" else torch.autocast(device_type=device.type, dtype=torch.float16)

  from keras_cv_attention_models import stable_diffusion
  # >>>> Using PyTorch backend
  mm = stable_diffusion.StableDiffusion().to(device)
  with torch.no_grad(), global_context:
      imm = mm.text_to_image('anime draw of a penguin under the moon on the beach.', image_shape=(768, 384), batch_size=4).cpu().numpy()

  print(f"{imm.shape = }, {imm.min() = }, {imm.max() = }")
  # imm.shape = (4, 3, 768, 384), imm.min() = -1.24831, imm.max() = 1.2017612
  plt.imsave('bb.jpg', np.hstack(np.clip(imm.transpose([0, 2, 3, 1]).astype("float32") / 2 + 0.5, 0, 1)))
  ```
  ![stable_diffusion_384_768](https://github.com/leondgarse/keras_cv_attention_models/assets/5744524/f8f322de-06c4-459e-8411-119b59bbebd2)

  **Show inner process results** by setting `return_inner=True`
  ```py
  import tensorflow as tf
  if len(tf.config.experimental.get_visible_devices('GPU')) > 0:
      tf.keras.mixed_precision.set_global_policy("mixed_float16")

  from keras_cv_attention_models import stable_diffusion
  mm = stable_diffusion.StableDiffusion()
  imms = mm.text_to_image('anime cute cat relaxing in the grass.', return_inner=True)

  imms = np.concatenate([ii.numpy().astype("float32") for ii in imms], axis=0)
  print(f"{imms.shape = }, {imms.min() = }, {imms.max() = }")
  # imms.shape = (50, 512, 512, 3), imms.min() = -1.9704391, imms.max() = 1.8913615
  imms = np.clip(imms / 2 + 0.5, 0, 1)
  plt.imshow(np.vstack([np.hstack(imms[id * 10: (id + 1) * 10]) for id in range(5)]))
  ```
  ![stable_diffusion_inner](https://github.com/leondgarse/keras_cv_attention_models/assets/5744524/efb3c8a4-6dea-4e40-b28c-a5bc8dacefbc)

  **image to image**
  ```py
  import tensorflow as tf
  if len(tf.config.experimental.get_visible_devices('GPU')) > 0:
      tf.keras.mixed_precision.set_global_policy("mixed_float16")

  from keras_cv_attention_models import stable_diffusion, test_images
  image = test_images.cat()
  mm = stable_diffusion.StableDiffusion()
  imm = mm.image_to_image(image, 'a tiger', batch_size=2).numpy()

  print(f"{imm.shape = }, {imm.min() = }, {imm.max() = }")
  # imm.shape = (2, 3, 512, 512), imm.min() = -1.066, imm.max() = 1.191
  plt.imshow(np.hstack([image / 255, *np.clip(imm.astype("float32") / 2 + 0.5, 0, 1)]))
  ```
  ![stable_duffusion_image_2_iamge](https://github.com/leondgarse/keras_cv_attention_models/assets/5744524/ff9b5cbb-6b7c-477d-b2c6-18fad9cf84d9)
