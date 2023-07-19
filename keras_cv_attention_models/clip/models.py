import math
from keras_cv_attention_models.backend import layers, models, functional
from keras_cv_attention_models.attention_layers import ExpLogitScale


def convert_to_clip_model(image_model, text_model):
    image_input, image_output = image_model.inputs[0], image_model.outputs[-1]
    text_input, text_output = text_model.inputs[0], text_model.outputs[-1]

    # image_output = layers.Dense(latents_dim, use_bias=False, dtype="float32", name="image_latents")(image_output)
    eol_index = functional.argmax(text_input[:, 1:], axis=-1) + 1  # Skip <|startoftext|> no matter what it is
    text_output = functional.gather_nd(text_output, functional.expand_dims(eol_index, axis=-1), batch_dims=1)
    text_output = layers.Dense(image_output.shape[-1], use_bias=False, name="text_latents")(text_output)
    text_output = ExpLogitScale(axis=None, init_value=math.log(1 / 0.07), name="temperature", dtype="float32")(text_output)
    text_model = models.Model(text_input, text_output)
    return models.Model([image_input, text_input], functional.concat([image_output, text_output], axis=-1)), image_model, text_model
