import os
import numpy as np
from PIL import Image
from tqdm.auto import tqdm
from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import callbacks, functional


class RunPrediction:
    def __init__(self, model, num_training_steps=1000, num_steps=0, beta_max=0.02):
        self.model, self.num_training_steps = model, num_training_steps
        self.num_steps = num_steps if num_steps > 0 else num_training_steps

        self.timesteps = np.arange(0, num_training_steps, num_training_steps // self.num_steps).astype("int64")
        beta = np.linspace(0.0001, beta_max, num_training_steps).astype("float32")[self.timesteps]
        alpha = 1.0 - beta
        alpha_bar = np.cumprod(alpha, axis=0)
        self.xt_prev_alpha = 1 / (alpha**0.5)
        self.xt_noise_alpha = self.xt_prev_alpha * (1 - alpha) / (1 - alpha_bar) ** 0.5
        self.eps_alpha = beta**0.5

        self.bar_format = "{desc}: {n_fmt}/{total_fmt} - [{elapsed}<{remaining},{rate_fmt}]"
        self.is_channels_last = backend.image_data_format() == "channels_last"
        self.device = None  # Set to actual torch model using device later
        self.to_device = (lambda xx: xx.to(self.device)) if backend.is_torch_backend else (lambda xx: xx)
        self.to_host = (lambda xx: xx.cpu()) if backend.is_torch_backend else (lambda xx: xx)

    def __call__(self, image_size=-1, batch_size=1, labels=None, init_x0=None, init_step=0, labels_guide_weight=1.8, return_inner=False):
        assert (len(self.model.inputs) == 2 and labels is None) or (len(self.model.inputs) == 3 and labels is not None), "labels input not matching with model"

        if backend.is_torch_backend and self.device is None:
            self.device = next(self.model.parameters()).device

        """ Init labels inputs first, for getting batch_size by labels """
        use_labels = False if labels is None else True
        if use_labels:
            if init_x0 is not None:
                assert init_x0.shape[0] == len(labels), "init_x0.shape[0] = {} should be equal with len(labels) = {}".format(init_x0.shape[0], len(labels))
            batch_size = len(labels)
            labels_inputs = self.to_device(functional.convert_to_tensor(labels, dtype="int64")) + 1  # add 1 skipping 0
            labels_inputs_zeros = self.to_device(functional.convert_to_tensor(np.zeros([batch_size]), dtype="int64"))

        """ Create init_x0 """
        if init_x0 is None:
            model_input_shape = self.model.input_shape[0][1:]
            image_size = image_size if None in model_input_shape else (model_input_shape[:-1] if self.is_channels_last else model_input_shape[1:])
            image_size = image_size if isinstance(image_size, (list, tuple)) else (image_size, image_size)
            noise_shape = (batch_size, *image_size, 3) if self.is_channels_last else (batch_size, 3, *image_size)
            init_x0 = np.random.normal(size=noise_shape).astype("float32")
        else:
            noise_shape, batch_size = init_x0.shape, init_x0.shape[0]
        xt = self.to_device(functional.convert_to_tensor(init_x0, dtype="float32"))

        """ Unet loop steps """
        gathered_inner = []
        for cur_step in tqdm(range(self.num_steps - init_step)[::-1], "Eval", bar_format=self.bar_format):  # 1000 -> 0
            timestep_inputs = functional.convert_to_tensor(np.stack([self.timesteps[cur_step]] * batch_size), dtype="int64")
            timestep_inputs = self.to_device(timestep_inputs)
            if use_labels:
                xt_noise_cond = self.model([xt, labels_inputs, timestep_inputs])
                xt_noise_zeros = self.model([xt, labels_inputs_zeros, timestep_inputs])
                xt_noise = xt_noise_cond + labels_guide_weight * (xt_noise_cond - xt_noise_zeros)
            else:
                xt_noise = self.model([xt, timestep_inputs])

            eps = functional.convert_to_tensor(np.random.normal(size=noise_shape).astype("float32"), dtype=xt_noise.dtype)
            xt = self.xt_prev_alpha[cur_step] * xt - self.xt_noise_alpha[cur_step] * xt_noise
            xt += self.eps_alpha[cur_step] * self.to_device(eps)

            if return_inner:
                gathered_inner.append(xt)

        """ Output """
        if return_inner:
            gathered_inner = np.stack([self.to_host(inner).numpy().astype("float32") for inner in gathered_inner], axis=1)
            eval_xt = gathered_inner if self.is_channels_last else gathered_inner.transpose([0, 1, 3, 4, 2])  # [batch, innner_steps, height, width, channel]
        else:
            xt = self.to_host(xt).numpy().astype("float32")
            eval_xt = xt if self.is_channels_last else xt.transpose([0, 2, 3, 1])
        eval_xt = (np.clip(eval_xt / 2 + 0.5, 0, 1) * 255).astype("uint8")
        return eval_xt


class DenoisingEval(callbacks.Callback):
    def __init__(
        self, save_path, image_size=512, num_classes=0, num_training_steps=1000, num_steps=0, labels_guide_weight=1.8, beta_max=0.02, interval=1, rows=5, cols=4
    ):
        super().__init__()
        self.save_path, self.image_size, self.labels_guide_weight, self.interval = save_path, image_size, labels_guide_weight, max(interval, 1)
        self.num_classes, self.rows, self.cols, self.batch_size = num_classes, rows, cols, rows * cols
        self.run_prediction = RunPrediction(model=None, num_training_steps=num_training_steps, num_steps=num_steps, beta_max=beta_max)

        if backend.image_data_format() == "channels_last":
            self.eval_x0 = np.random.normal(size=(self.batch_size, image_size, image_size, 3)).astype("float32")
        else:
            self.eval_x0 = np.random.normal(size=(self.batch_size, 3, image_size, image_size)).astype("float32")

        if num_classes > 0:
            labels_inputs = np.arange(0, num_classes, num_classes / self.batch_size)[: self.batch_size].astype("int64")
            self.labels_inputs = np.pad(labels_inputs, [0, self.batch_size - labels_inputs.shape[0]], constant_values=num_classes - 1)  # Just in case
            print(">>>> Eval labels:", self.labels_inputs)

    def on_epoch_end(self, cur_epoch=0, logs=None):
        if (cur_epoch + 1) % self.interval > 0:
            return
        if self.run_prediction.model is None:
            self.run_prediction.model = self.model
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)

        lables = self.labels_inputs if self.num_classes > 0 else None
        eval_xt = self.run_prediction(labels=lables, init_x0=self.eval_x0, init_step=0, labels_guide_weight=self.labels_guide_weight)
        eval_xt = np.vstack([np.hstack(eval_xt[row * self.cols : (row + 1) * self.cols]) for row in range(self.rows)])
        save_path = os.path.join(self.save_path, "epoch_{}.jpg".format(cur_epoch + 1) if isinstance(cur_epoch, int) else (cur_epoch + ".jpg"))
        Image.fromarray(eval_xt).save(save_path)
        print(">>>> Eval result image saved to {}".format(save_path))
