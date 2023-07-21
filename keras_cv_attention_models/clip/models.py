import math
from keras_cv_attention_models.backend import layers, models, functional
from keras_cv_attention_models.attention_layers import ExpLogitScale


def convert_to_clip_model(image_model, text_model, caption_tokenizer=None):
    image_input, image_output = image_model.inputs[0], image_model.outputs[-1]
    text_input, text_output = text_model.inputs[0], text_model.outputs[-1]

    # image_output = layers.Dense(latents_dim, use_bias=False, dtype="float32", name="image_latents")(image_output)
    eol_index = functional.argmax(functional.pad(text_input[:, 1:], [[0, 0], [1, 0]]), axis=-1)  # Skip <|startoftext|> no matter what it is
    text_output = functional.gather_nd(text_output, functional.expand_dims(eol_index, axis=-1), batch_dims=1)
    text_output = layers.Dense(image_output.shape[-1], use_bias=False, dtype="float32", name="text_latents")(text_output)
    text_model = models.Model(text_input, text_output)

    # Return similarity directly for avoiding re-calculating in metrics again
    text_latents = ExpLogitScale(axis=None, init_value=math.log(1 / 0.07), dtype="float32", name="temperature")(text_output)
    text_latents = text_latents / functional.norm(text_latents, ord="euclidean", axis=-1, keepdims=True)
    image_latents = image_output / functional.norm(image_output, ord="euclidean", axis=-1, keepdims=True)
    similarity = functional.matmul(text_latents, image_latents, transpose_b=True)

    model = models.Model([image_input, text_input], similarity)
    model.run_prediction = RunPrediction(image_model, text_model, caption_tokenizer=caption_tokenizer)
    return model, image_model, text_model


class RunPrediction:
    def __init__(self, image_model, text_model, caption_tokenizer, softmax_scale=10, formatter="a {}", rescale_mode="tf"):
        self.image_model, self.text_model, self.caption_tokenizer = image_model, text_model, caption_tokenizer
        self.reset(softmax_scale=softmax_scale, formatter=formatter, rescale_mode=rescale_mode)

    def reset(self, softmax_scale=10, formatter="a {}", rescale_mode="tf"):
        self.formatter, self.text_labels, self.text_features = formatter, None, None
        self.softmax_scale = softmax_scale
        self._init_image_model_preprocess_input_(rescale_mode)

    def init_image_model_preprocess_input(self, rescale_mode):
        if hasattr(self.image_model, "preprocess_input"):
            self.image_model.preprocess_input.set_rescale_mode(rescale_mode)
        else:
            from keras_cv_attention_models.attention_layers import PreprocessInput

            self.image_model.preprocess_input = PreprocessInput(self.image_model.input_shape[1:], rescale_mode=rescale_mode)
        self.rescale_mode = rescale_mode

    def _if_init_text_features_(self, text_labels):
        if text_labels is None and self.text_labels is None:
            raise ValueError("[Error] text_labels has to be provided for the first time call. Format like ['cat', 'dog', ...]")

        if text_labels is not None:
            self.text_labels = text_labels if self.formatter is None else [self.formatter.format(ii) for ii in text_labels]

        if text_labels is not None or self.text_features is None:
            text_features = functional.stack([self.caption_tokenizer(ii) for ii in self.text_labels])
            text_features = self.text_model(text_features)
            self.text_features = text_features / functional.norm(text_features, axis=-1, keepdims=True)

    def __call__(self, image, text_labels=None):
        self._if_init_text_features_(text_labels=text_labels)

        image_features = self.image_model(self.image_model.preprocess_input(image))
        image_features /= functional.norm(image_features, axis=-1, keepdims=True)
        return functional.softmax(self.softmax_scale * image_features @ functional.transpose(self.text_features))
