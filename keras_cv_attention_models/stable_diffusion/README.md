# ___Keras Stable Diffusion___
  - [Summary](#summary)
  - [Models](#models)
  - [Usage](#usage)
  - [DDPM training](#ddpm-training)
  - [Test of Encoder and Decoder](#test-of-encoder-and-decoder)
***

## Summary
  - [PDF 2006.11239 Denoising Diffusion Probabilistic Models](https://arxiv.org/pdf/2006.11239.pdf)
  - [PDF 2112.10752 High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/pdf/2112.10752.pdf)
  - [Huggingface runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5)
  - [The Illustrated Stable Diffusion](https://jalammar.github.io/illustrated-stable-diffusion/)
  - [Github CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion)
  - [Github runwayml/stable-diffusion](https://github.com/runwayml/stable-diffusion)
  - [Github stability-ai/stablediffusion](https://github.com/stability-ai/stablediffusion)
  - [Github labmlai/annotated_deep_learning_paper_implementations/stable_diffusion](https://github.com/labmlai/annotated_deep_learning_paper_implementations/tree/master/labml_nn/diffusion/stable_diffusion)
  - [Github zoubohao/DenoisingDiffusionProbabilityModel-ddpm-](https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-)
  - Model weights ported from [Github runwayml/stable-diffusion](https://github.com/runwayml/stable-diffusion) `sd-v1-5.ckpt`.
## Models
  | Model               | Params | FLOPs   | Input               | Download            |
  | ------------------- | ------ | ------- | ------------------- | ------------------- |
  | ViTTextLargePatch14 | 123.1M | 6.67G   | [None, 77]          | [vit_text_large_patch14_clip.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/beit/vit_text_large_patch14_clip.h5) |
  | Encoder             | 34.16M | 559.6G  | [None, 512, 512, 3] | [encoder_v1_5.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/stable_diffusion/encoder_v1_5.h5) |
  | UNet                | 859.5M | 404.4G  | [None, 64, 64, 4]   | [unet_v1_5.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/stable_diffusion/unet_v1_5.h5) |
  | Decoder             | 49.49M | 1259.5G | [None, 64, 64, 4]   | [decoder_v1_5.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/stable_diffusion/decoder_v1_5.h5) |

  **Note: `StableDiffusion` is not defined a model, just a gather of above models and forward functions.**
## Usage
  - **Basic**
    ```py
    import tensorflow as tf
    if len(tf.config.experimental.get_visible_devices('GPU')) > 0:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    from keras_cv_attention_models import stable_diffusion
    mm = stable_diffusion.StableDiffusion()

    imm = mm('Cyberpunk cityscape with towering skyscrapers, neon signs, and flying cars.', batch_size=4).numpy()
    print(f"{imm.shape = }, {imm.min() = }, {imm.max() = }")
    # imm.shape = (4, 512, 512, 3), imm.min() = -2.4545908, imm.max() = 1.851803
    plt.imshow(np.hstack(np.clip(imm.astype("float32") / 2 + 0.5, 0, 1)))
    ```
    ![stabel_diffusion](https://github.com/leondgarse/keras_cv_attention_models/assets/5744524/e565c750-f98a-4d04-a280-0d0aa382ef5f)
  - **Change to other shape** by setting `image_shape`. Should be divisible by `64`, as UNet needs to concatenate down and up samples
    ```py
    import tensorflow as tf
    if len(tf.config.experimental.get_visible_devices('GPU')) > 0:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    from keras_cv_attention_models import stable_diffusion
    mm = stable_diffusion.StableDiffusion()

    prompt = 'mountains, stars and paisley fileed sky, artstation, digital painting, sharp focus.'
    imm = mm(prompt=prompt, image_shape=(512, 1024)).numpy()
    print(f"{imm.shape = }, {imm.min() = }, {imm.max() = }")
    # imm.shape = (1, 512, 1024, 3), imm.min() = -1.5322105, imm.max() = 1.419162
    plt.imsave('aa.jpg', np.hstack(np.clip(imm.astype("float32") / 2 + 0.5, 0, 1)))
    ```
    ![stable_diffusion_512_1024](https://github.com/leondgarse/keras_cv_attention_models/assets/5744524/a10e3b97-38b5-4993-92ff-98f05ac0055d)
  - **Show inner process results** by setting `return_inner=True`
    ```py
    import tensorflow as tf
    if len(tf.config.experimental.get_visible_devices('GPU')) > 0:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    from keras_cv_attention_models import stable_diffusion
    mm = stable_diffusion.StableDiffusion()

    imms = mm('anime cute cat relaxing in the grass.', return_inner=True)
    imms = np.concatenate([ii.numpy().astype("float32") for ii in imms], axis=0)
    print(f"{imms.shape = }, {imms.min() = }, {imms.max() = }")
    # imms.shape = (50, 512, 512, 3), imms.min() = -1.9704391, imms.max() = 1.8913615
    imms = np.clip(imms / 2 + 0.5, 0, 1)
    plt.imshow(np.vstack([np.hstack(imms[id * 10: (id + 1) * 10]) for id in range(5)]))
    ```
    ![stable_diffusion_inner](https://github.com/leondgarse/keras_cv_attention_models/assets/5744524/efb3c8a4-6dea-4e40-b28c-a5bc8dacefbc)
  - **Image to image** by giving an image or image_path to `image`
    ```py
    import tensorflow as tf
    if len(tf.config.experimental.get_visible_devices('GPU')) > 0:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    from keras_cv_attention_models import stable_diffusion, test_images
    mm = stable_diffusion.StableDiffusion()

    image = test_images.cat()
    imm = mm('a tiger', image=image, batch_size=2).numpy()
    print(f"{imm.shape = }, {imm.min() = }, {imm.max() = }")
    # imm.shape = (2, 512, 512, 3), imm.min() = -1.066, imm.max() = 1.191
    plt.imshow(np.hstack([image / 255, *np.clip(imm.astype("float32") / 2 + 0.5, 0, 1)]))
    ```
    ![stable_duffusion_image_2_iamge](https://github.com/leondgarse/keras_cv_attention_models/assets/5744524/ff9b5cbb-6b7c-477d-b2c6-18fad9cf84d9)
  - **Inpaint** by giving an image or image_path to `image`, and `inpaint_mask` for keeping part of the original image. `inpaint_mask` is in same shape with `image_latents`, which is the output of `Encoder` model in shape `[batch_size, image_height // 8, image_width // 8, 4]`, or 4 float value `[top, left, bottom, right]` in `[0, 1]`, like `[0.5, 0, 1, 1]` for keeping bottom half.
    ```py
    import tensorflow as tf
    if len(tf.config.experimental.get_visible_devices('GPU')) > 0:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    from keras_cv_attention_models import stable_diffusion, test_images
    mm = stable_diffusion.StableDiffusion()

    image = test_images.cat()
    imm = mm('a tiger mask', image=image, batch_size=2, inpaint_mask=(0.5, 0, 1, 1)).numpy()
    print(f"{imm.shape = }, {imm.min() = }, {imm.max() = }")
    # imm.shape = (2, 512, 512, 3), imm.min() = -1.0901726, imm.max() = 1.257365
    plt.imshow(np.hstack([image / 255, *np.clip(imm.astype("float32") / 2 + 0.5, 0, 1)]))
    ```
    ![stable_diffusion_inpaint](https://github.com/leondgarse/keras_cv_attention_models/assets/5744524/c44c7585-9949-4826-afff-450258c8bb18)
  - **Using PyTorch backend** by set `KECAM_BACKEND='torch'` environment variable.
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
        imm = mm('anime draw of a penguin under the moon on the beach.', image_shape=(768, 384), batch_size=4).cpu().numpy()
    print(f"{imm.shape = }, {imm.min() = }, {imm.max() = }")
    # imm.shape = (4, 3, 768, 384), imm.min() = -1.24831, imm.max() = 1.2017612
    plt.imshow(np.hstack(np.clip(imm.transpose([0, 2, 3, 1]).astype("float32") / 2 + 0.5, 0, 1)))
    ```
    ![stable_diffusion_384_768](https://github.com/leondgarse/keras_cv_attention_models/assets/5744524/f8f322de-06c4-459e-8411-119b59bbebd2)
## DDPM training
  - **Note: Works better with PyTorch backend, Tensorflow one seems overfitted if training logger like `--epochs 200`, and evaluation runs ~5 times slower. [???]**
  - **Dataset** can be a directory containing images for basic DDPM training using images only, or a recognition json file created following [Custom recognition dataset](https://github.com/leondgarse/keras_cv_attention_models/discussions/52#discussion-3971513), which will train using labels as instruction.
    ```sh
    python custom_dataset_script.py --train_images cifar10/train/ --test_images cifar10/test/
    # >>>> total_train_samples: 50000, total_test_samples: 10000, num_classes: 10
    # >>>> Saved to: cifar10.json
    ```
  - **Train using `ddpm_train_script.py on cifar10 with labels`** Default `--data_path` is builtin `cifar10`.
    ```py
    # Set --eval_interval 50 as TF evaluation is rather slow [???]
    TF_XLA_FLAGS="--tf_xla_auto_jit=2" CUDA_VISIBLE_DEVICES=1 python ddpm_train_script.py --eval_interval 50
    ```
    **Train Using PyTorch backend by setting `KECAM_BACKEND='torch'`**
    ```py
    # Training on gtsrb dataset
    KECAM_BACKEND='torch' CUDA_VISIBLE_DEVICES=1 python ddpm_train_script.py --eval_interval 50 \
    --data_path gtsrb --input_shape 64 --batch_size 64 --disable_horizontal_flip
    ```
    ![ddpm_unet_test_E100](https://github.com/leondgarse/keras_cv_attention_models/assets/5744524/861f4004-4496-4aff-ae9c-706f4c04fef2)
  - **Reload model and run prediction after training**
    ```py
    import numpy as np
    import tensorflow as tf
    if len(tf.config.experimental.get_visible_devices('GPU')) > 0:
        tf.keras.mixed_precision.set_global_policy("mixed_float16")

    from keras_cv_attention_models import stable_diffusion, plot_func
    num_classes, num_samples = 10, 8
    mm = stable_diffusion.UNetTest(num_classes=num_classes, input_shape=(32, 32, 3))
    mm.load_weights("checkpoints/ddpm_unet_test_tensorflow_latest.h5")

    images = mm.run_prediction(labels=np.arange(num_classes).repeat(num_samples))
    print(f"{images.shape = }, {images.min() = }, {images.max() = }")
    plt.imshow(np.vstack([np.hstack(row * num_samples: (row + 1) * num_samples) for row in range(num_classes)]))
    ```
    **Or using PyTorch backend** on gtsrb dataset
    ```py
    os.environ['KECAM_BACKEND'] = 'torch'
    import torch
    import numpy as np
    from contextlib import nullcontext
    device = torch.device("cuda:0") if torch.cuda.is_available() and int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")) >= 0 else torch.device("cpu")
    global_context = nullcontext() if device.type == "cpu" else torch.autocast(device_type=device.type, dtype=torch.float16)

    from keras_cv_attention_models import stable_diffusion, plot_func
    mm = stable_diffusion.UNetTest(num_classes=43, input_shape=(64, 64, 3))
    mm.load_weights("checkpoints/ddpm_unet_test_torch_gtsrb_E200_latest.pt")  # can also be a h5 file
    mm = mm.to(device)

    with torch.no_grad(), global_context:
        images = mm.run_prediction(labels=np.arange(40).repeat(2))
    print(f"{images.shape = }, {images.min() = }, {images.max() = }")
    plt.imshow(np.vstack([np.hstack(row * 8: (row + 1) * 8) for row in range(10)]))
    ```
    ![gtsrb_15](https://github.com/leondgarse/keras_cv_attention_models/assets/5744524/5952ab94-d7ac-426e-a5ae-121a450d4c48)
## Test of Encoder and Decoder
  ```py
  from keras_cv_attention_models import stable_diffusion, test_images

  ee = stable_diffusion.Encoder()
  encoder_outputs = ee(test_images.cat().astype("float32")[None] / 127.5 - 1).numpy()

  mean, log_var = np.split(encoder_outputs, 2, axis=-1)
  log_var = np.clip(log_var, -30.0, 20.0)
  std = np.exp(log_var * 0.5)
  gaussian = np.random.normal(size=std.shape)
  image_latents = mean + std * gaussian

  dd = stable_diffusion.Decoder()
  out = dd(image_latents).numpy()
  plt.imshow(np.clip(out[0] / 2 + 0.5, 0, 1))
  ```
  Output should be same as the input image.
