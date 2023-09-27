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
  from keras_cv_attention_models import stable_diffusion

  mm = stable_diffusion.StableDiffusion()
  imm = mm.text_to_image('Cyberpunk cityscape with towering skyscrapers, neon signs, and flying cars.', batch_size=4).numpy()
  print(f"{imm.shape = }, {imm.min() = }, {imm.max() = }")
  # imm.shape = (4, 512, 512, 3), imm.min() = -2.4545908, imm.max() = 1.851803
  plt.imshow(np.hstack(np.clip(imm / 2 + 0.5, 0, 1)))
  ```
  ![stabel_diffusion](https://github.com/leondgarse/keras_cv_attention_models/assets/5744524/e565c750-f98a-4d04-a280-0d0aa382ef5f)

  **Change to other shape** by setting `image_shape`. Should be divisible by `64`, as UNet needs to concatenate down and up samples
  ```py
  from keras_cv_attention_models import stable_diffusion
  mm = stable_diffusion.StableDiffusion(image_shape=(512, 1024, 3))
  imm = mm.text_to_image('mountains, stars and paisley fileed sky, artstation, digital painting, sharp focus.', batch_size=1).numpy()
  print(f"{imm.shape = }, {imm.min() = }, {imm.max() = }")
  # imm.shape = (1, 512, 1024, 3), imm.min() = -1.5322105, imm.max() = 1.419162
  plt.imsave('aa.jpg', np.hstack(np.clip(imm / 2 + 0.5, 0, 1)))
  ```
  ![stable_diffusion_512_1024](https://github.com/leondgarse/keras_cv_attention_models/assets/5744524/a10e3b97-38b5-4993-92ff-98f05ac0055d)

  **Using PyTorch backend** by set `KECAM_BACKEND='torch'` environment variable.
  ```py
  os.environ['KECAM_BACKEND'] = 'torch'
  import torch
  from contextlib import nullcontext
  device = torch.device("cuda:0") if torch.cuda.is_available() and int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")) >= 0 else torch.device("cpu")
  global_context = nullcontext() if device.type == "cpu" else torch.amp.autocast(device_type=device.type, dtype=torch.float16)

  from keras_cv_attention_models import stable_diffusion
  # >>>> Using PyTorch backend
  mm = stable_diffusion.StableDiffusion(image_shape=(768, 384, 3))
  mm.to(device)
  with torch.no_grad(), global_context:
      imm = mm.text_to_image('anime draw of a penguin under the moon on the beach.', batch_size=4).cpu().numpy()
  print(f"{imm.shape = }, {imm.min() = }, {imm.max() = }")
  # imm.shape = (4, 3, 768, 384), imm.min() = -1.24831, imm.max() = 1.2017612
  plt.imsave('bb.jpg', np.hstack(np.clip(imm.transpose([0, 2, 3, 1]) / 2 + 0.5, 0, 1)))
  ```
  ![stable_diffusion_384_768](https://github.com/leondgarse/keras_cv_attention_models/assets/5744524/f8f322de-06c4-459e-8411-119b59bbebd2)
