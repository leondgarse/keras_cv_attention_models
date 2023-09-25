import numpy as np
from tqdm.auto import tqdm
from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, functional, models, initializers, image_data_format
from keras_cv_attention_models.stable_diffusion.unet import UNet
from keras_cv_attention_models.beit.vit import ViTTextLargePatch14
from keras_cv_attention_models.stable_diffusion.encoder_decoder import Encoder, Decoder, gaussian_distribution
from keras_cv_attention_models.clip.tokenizer import SimpleTokenizer
from keras_cv_attention_models.clip.models import add_text_model_index_header


class StableDiffusion:
    def __init__(self, n_steps=50, n_steps_training=1000, ddim_discretize="uniform", linear_start=0.00085, linear_end=0.0120, ddim_eta=0.0):
        self.tokenizer = SimpleTokenizer()
        clip_model = ViTTextLargePatch14(vocab_size=self.tokenizer.vocab_size, include_top=False)
        self.clip_model = add_text_model_index_header(clip_model, latents_dim=0, caption_tokenizer=self.tokenizer)

        # self.encoder = Encoder()
        self.unet_model = UNet()
        self.decoder_model = Decoder()

        self.uncond_prompt = self.clip_model(functional.convert_to_tensor(self.tokenizer("")[None]))
        self.channel_axis = -1 if image_data_format() == "channels_last" else 1
        self.n_steps, self.n_steps_training, self.ddim_discretize, self.ddim_eta = n_steps, n_steps_training, ddim_discretize, ddim_eta
        self.linear_start, self.linear_end = linear_start, linear_end
        self.init_ddim_sampler(n_steps, n_steps_training, ddim_discretize, linear_start, linear_end, ddim_eta)

    def init_ddim_sampler(self, n_steps=50, n_steps_training=1000, ddim_discretize="uniform", linear_start=0.00085, linear_end=0.0120, ddim_eta=0.0):
        # DDIM sampling from the paper [Denoising Diffusion Implicit Models](https://papers.labml.ai/paper/2010.02502)
        # n_steps, n_steps_training, ddim_discretize, linear_start, linear_end, ddim_eta = 50, 1000, "uniform", 0.00085, 0.0120, 0
        if ddim_discretize == "quad":
            time_steps = ((np.linspace(0, np.sqrt(n_steps_training * 0.8), n_steps)) ** 2).astype(int) + 1
        else:  # "uniform"
            interval = n_steps_training // n_steps
            time_steps = np.arange(0, n_steps_training, interval) + 1

        beta = np.linspace(linear_start**0.5, linear_end**0.5, n_steps_training, dtype="float64") ** 2
        alpha = 1.0 - beta
        alpha_bar = np.cumprod(alpha, axis=0).astype("float32")

        ddim_alpha = alpha_bar[time_steps]
        ddim_alpha_sqrt = np.sqrt(ddim_alpha)
        ddim_alpha_prev = np.concatenate([alpha_bar[:1], alpha_bar[time_steps[:-1]]])
        ddim_sigma = ddim_eta * ((1 - ddim_alpha_prev) / (1 - ddim_alpha) * (1 - ddim_alpha / ddim_alpha_prev)) ** 0.5
        ddim_sqrt_one_minus_alpha = (1.0 - ddim_alpha) ** 0.5

        self.time_steps, self.ddim_alpha, self.ddim_alpha_sqrt, self.ddim_alpha_prev = time_steps, ddim_alpha, ddim_alpha_sqrt, ddim_alpha_prev
        self.ddim_sigma, self.ddim_sqrt_one_minus_alpha = ddim_sigma, ddim_sqrt_one_minus_alpha

    def text_to_image(
        self,
        prompt,
        input_shape=[None, 512 // 8, 512 // 8, 4],  # 3 or 4 dimension, will exclude the first dimension if 4
        batch_size=4,
        repeat_noise=False,  # specified whether the noise should be same for all samples in the batch
        temperature=1,  # is the noise temperature (random noise gets multiplied by this)
        init_x0=None,  # If not provided random noise will be used.
        init_step=0,  # is the number of time steps to skip $i'$. We start sampling from $S - i'$. And `x_last` is then $x_{\tau_{S - i'}}$.
        latent_scaling_factor=0.18215,
        uncond_scale=7.5,  # unconditional guidance scale: "eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))"
        return_inner=False,  # boolean value if return inner step results for visualizing the process
    ):
        input_shape = input_shape if len(input_shape) == 3 else input_shape[1:]  # Exclude batch_size dimension
        # assume channel dimension is the one with min value in input_shape, and put it first or last regarding image_data_format
        input_shape = backend.align_input_shape_by_image_data_format(input_shape)
        target_shape = [batch_size, *input_shape]

        cond_prompt = self.clip_model(self.tokenizer(prompt)[None])
        uncond_cond_prompt = functional.concat([self.uncond_prompt] * batch_size + [cond_prompt] * batch_size, axis=0)

        xt = np.random.normal(size=target_shape) if init_x0 is None else init_x0
        xt = functional.convert_to_tensor(xt.astype("float32"))

        rr = []
        for cur_step in tqdm(range(self.n_steps - init_step)[::-1]):
            time_step = functional.convert_to_tensor(np.stack([self.time_steps[cur_step]] * batch_size * 2))
            xt_inputs = functional.concat([xt, xt], axis=0)

            # get_eps
            out = self.unet_model([xt_inputs, time_step, uncond_cond_prompt])
            e_t_uncond, e_t_cond = functional.split(out, 2, axis=0)
            e_t = e_t_uncond + (e_t_cond - e_t_uncond) * uncond_scale

            # get_x_prev_and_pred_x0
            ddim_alpha_prev, ddim_sigma = self.ddim_alpha_prev[cur_step], self.ddim_sigma[cur_step]
            pred_x0 = (xt - e_t * self.ddim_sqrt_one_minus_alpha[cur_step]) / (self.ddim_alpha[cur_step] ** 0.5)  # Current prediction for x_0
            dir_xt = e_t * ((1.0 - ddim_alpha_prev - ddim_sigma**2) ** 0.5)  # Direction pointing to x_t

            if ddim_sigma == 0:
                noise = 0.0
            elif repeat_noise:
                noise = np.random.normal(size=(1, *target_shape[1:])).astype("float32")
            else:
                noise = np.random.normal(size=target_shape).astype("float32")
            xt = (ddim_alpha_prev**0.5) * pred_x0 + dir_xt + ddim_sigma * temperature * functional.convert_to_tensor(noise)
            if return_inner:
                rr.append(xt)

        # Decode the image
        if return_inner:
            return [self.decoder_model(inner / latent_scaling_factor) for inner in rr]
        else:
            return self.decoder_model(xt / latent_scaling_factor)
