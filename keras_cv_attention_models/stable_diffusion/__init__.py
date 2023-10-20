from keras_cv_attention_models import backend as __backend__
from keras_cv_attention_models.stable_diffusion.stable_diffusion import StableDiffusion
from keras_cv_attention_models.stable_diffusion.unet import UNet, UNetTest
from keras_cv_attention_models.stable_diffusion.encoder_decoder import Encoder, Decoder
from keras_cv_attention_models.stable_diffusion.eval_func import RunPrediction

if __backend__.is_tensorflow_backend:
    from keras_cv_attention_models.stable_diffusion.data import build_tf_dataset as build_dataset
else:
    from keras_cv_attention_models.stable_diffusion.data import build_torch_dataset as build_dataset


__head_doc__ = """
Keras implementation of [Github CompVis/stable-diffusion](https://github.com/CompVis/stable-diffusion).
Paper [PDF 2112.10752 High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/pdf/2112.10752.pdf).
"""

__tail_doc__ = """  image_shape: model image input shape and generated image shape.
      Should have exactly 3 inputs channels like `(224, 224, 3)`.
      Inner latents inpuit shape for UNet and Decode is `[image_shape[0] // 8, image_shape[1] // 8, 4]`.
  clip_model: str value like `beit.ViTTextLargePatch14` for models from this package under `keras_cv_attention_models`.
      Also can be a built model, or None for not using.
  unet_model: str value like `stable_diffusion.UNet` for models from this package under `keras_cv_attention_models`.
      Also can be a built model, or None for not using.
  decoder_model: str value like `stable_diffusion.Decoder` for models from this package under `keras_cv_attention_models`.
      Also can be a built model, or None for not using.
  encoder_model: str value like `stable_diffusion.Encoder` for models from this package under `keras_cv_attention_models`.
      Also can be a built model, or None for not using.
  clip_model_kwargs: dict value for kwargs used for building `clip_model`.
  unet_model_kwargs: dict value for kwargs used for building `unet_model`.
  decoder_model_kwargs: dict value for kwargs used for building `decoder_model`.
  encoder_model_kwargs: dict value for kwargs used for building `encoder_model`.
  caption_tokenizer: str value in ['GPT2Tokenizer', 'SimpleTokenizer', 'SentencePieceTokenizer'],
      or tiktoken one ['gpt2', 'r50k_base', 'p50k_base', 'cl100k_base'],
      or specified built tokenizer.
  num_steps: int value for the number of DDIM sampling steps, also means total denoising steps.
  num_training_steps: int value for total denoising steps during training.
  ddim_discretize: one of ["uniform", "quad"] for time_steps sampling `num_steps` method from `num_training_steps`.
  linear_start: float value for `beta` start value.
  linear_end: float value for `beta` end value.
  ddim_eta: float value for calculating `ddim_sigma`. 0 makes the sampling process deterministic.

Returns:
    A `StableDiffusion` instance.
"""

StableDiffusion.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__ + """
Model architectures:
  | Model               | Params | FLOPs   | Input               |
  | ------------------- | ------ | ------- | ------------------- |
  | ViTTextLargePatch14 | 123.1M | 6.67G   | [None, 77]          |
  | Encoder             | 34.16M | 559.6G  | [None, 512, 512, 3] |
  | UNet                | 859.5M | 404.4G  | [None, 64, 64, 4]   |
  | Decoder             | 49.49M | 1259.5G | [None, 64, 64, 4]   |
"""
