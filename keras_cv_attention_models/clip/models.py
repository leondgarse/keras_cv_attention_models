import math
from keras_cv_attention_models.backend import layers, models, functional, initializers
from keras_cv_attention_models import attention_layers


def text_model_index_header(text_input, text_output, latents_dim=512, eot_token=None):
    if eot_token is None:
        eol_index = functional.argmax(functional.pad(text_input[:, 1:], [[0, 0], [1, 0]]), axis=-1)  # Skip <|startoftext|> no matter what it is
    else:
        assert eot_token != 0, "eot_token is 0, conflict with the padding value"
        eol_index = functional.argmax(functional.equal(functional.pad(text_input[:, 1:], [[0, 0], [1, 0]]), eot_token), axis=-1)

    text_output = functional.gather_nd(text_output, functional.expand_dims(eol_index, axis=-1), batch_dims=1)
    if latents_dim > 0:
        kernel_initializer = initializers.RandomNormal(stddev=text_output.shape[-1] ** -0.5)
        text_output = layers.Dense(latents_dim, use_bias=False, kernel_initializer=kernel_initializer, dtype="float32", name="text_latents")(text_output)
    return text_output


def add_text_model_index_header(model, latents_dim=512, caption_tokenizer=None):
    # Use None for argmax as eot position for GPT2Tokenizer and SimpleTokenizer. For SentencePieceTokenizer eot==2, pass the actual value
    eot_token = None if caption_tokenizer is None or caption_tokenizer.eot_token >= caption_tokenizer.vocab_size - 2 else caption_tokenizer.eot_token
    text_input, text_output = model.inputs[0], model.outputs[-1]
    text_output = text_model_index_header(text_input, text_output, latents_dim=latents_dim, eot_token=eot_token)
    return models.Model(text_input, text_output, name=model.name)


def convert_to_clip_model(image_model, text_model=None, caption_tokenizer=None, epsilon=1e-6):
    """
    >>> import tensorflow as tf  # Or import torch for PyTroch backend
    >>> from keras_cv_attention_models import clip, gpt2, beit
    >>> image_model = beit.ViT(num_classes=512, classifier_activation=None)
    >>> text_model = gpt2.GPT2_Base(include_top=False)
    >>> caption_tokenizer = clip.SimpleTokenizer()
    >>> model, image_model, text_model = clip.convert_to_clip_model(image_model, text_model, caption_tokenizer)
    >>> model.run_prediction(tf.ones([1, 224, 224, 3]), ['cat', 'dog'])
    """
    if text_model is None:
        print(">>>> Build text_model from image_model")
        kwargs = {"vocab_size": caption_tokenizer.vocab_size} if caption_tokenizer is not None and hasattr(caption_tokenizer, "vocab_size") else {}
        image_model, text_model = build_text_model_from_image_model(image_model, **kwargs)

    image_input, image_output = image_model.inputs[0], image_model.outputs[-1]
    if text_model.output_names[0] != "text_latents":
        text_model = add_text_model_index_header(text_model, latents_dim=image_output.shape[-1], caption_tokenizer=caption_tokenizer)
    text_input, text_output = text_model.inputs[0], text_model.outputs[-1]

    # Return similarity directly for avoiding re-calculating in metrics again
    # text_latents = text_output / functional.norm(text_output, ord="euclidean", axis=-1, keepdims=True)
    text_latents = text_output * functional.rsqrt(functional.reduce_sum(functional.square(text_output), axis=-1, keepdims=True) + epsilon)
    image_latents = layers.Identity(dtype="float32", name="image_latents")(image_output)  # Give a name for locating this layer in split
    # image_latents = image_latents / functional.norm(image_latents, ord="euclidean", axis=-1, keepdims=True)
    image_latents = image_latents * functional.rsqrt(functional.reduce_sum(functional.square(image_latents), axis=-1, keepdims=True) + epsilon)

    similarity = functional.matmul(image_latents, text_latents, transpose_b=True)
    similarity = attention_layers.ExpLogitScale(axis=None, init_value=math.log(1 / 0.07), dtype="float32", name="temperature")(similarity)

    model = models.Model([image_input, text_input], similarity)
    model.run_prediction = RunPrediction(image_model, text_model, caption_tokenizer=caption_tokenizer)
    return model, image_model, text_model


def split_to_image_text_model(model):
    text_model = models.Model(model.inputs[1], model.get_layer("text_latents").output)

    image_latents_layer = model.get_layer("image_latents")
    image_output = image_latents_layer.input if isinstance(image_latents_layer, layers.Identity) else image_latents_layer.output
    image_model = models.Model(model.inputs[0], image_output)
    return image_model, text_model


def build_text_model_from_image_model(image_model, vocab_size=50257, text_dropout=0):
    from keras_cv_attention_models import model_surgery

    head_model, body_model, tail_model = model_surgery.split_model_to_head_body_tail_by_blocks(image_model)
    body_model = model_surgery.convert_to_dynamic_input_shape(body_model)

    image_input, image_head_output = head_model.inputs[0], head_model.outputs[0]
    image_body_output = body_model(image_head_output)
    image_output = tail_model(image_body_output)
    image_model = models.Model(image_input, image_output)  # Or models.Sequential([head_model, body_model, tail_model])

    body_input_rank = len(body_model.inputs[0].shape)
    assert body_input_rank == 3, "input for body has to be rank 3, got input_layer: {}, rank: {}".format(body_start_layer.name, body_input_rank)
    max_block_size, embedding_size = head_model.outputs[0].shape[1], body_model.inputs[0].shape[-1]

    text_input = layers.Input([None], dtype="int64")
    pos_idx = attention_layers.PositionalIndex(block_size=max_block_size, name="pos_idx")(text_input)
    tok_emb = layers.Embedding(vocab_size, embedding_size, name="wte")(text_input)
    pos_emb = layers.Embedding(max_block_size, embedding_size, name="wpe")(pos_idx)
    text_head_output = tok_emb + pos_emb
    text_head_output = layers.Dropout(dropout)(text_head_output) if text_dropout > 0 else text_head_output

    text_body_output = body_model(text_head_output)
    # text_output = text_model_index_header(text_input, text_body_output, latents_dim=image_model.outputs[0].shape[-1])
    text_model = models.Model(text_input, text_body_output, name=image_model.name + "_text")
    return image_model, text_model


class RunPrediction:
    def __init__(self, image_model, text_model, caption_tokenizer, softmax_scale=100, formatter="a {}", rescale_mode="torch"):
        self.image_model, self.text_model, self.caption_tokenizer = image_model, text_model, caption_tokenizer
        self.reset(softmax_scale=softmax_scale, formatter=formatter, rescale_mode=rescale_mode)

    def reset(self, softmax_scale=100, formatter="a {}", rescale_mode="torch"):
        self.formatter, self.text_labels, self.text_features = formatter, None, None
        self.softmax_scale = softmax_scale
        self.init_image_model_preprocess_input(rescale_mode)

    def init_image_model_preprocess_input(self, rescale_mode):
        if hasattr(self.image_model, "preprocess_input"):
            self.image_model.preprocess_input.set_rescale_mode(rescale_mode)
        else:
            self.image_model.preprocess_input = attention_layers.PreprocessInput(self.image_model.input_shape[1:], rescale_mode=rescale_mode)
        self.rescale_mode = rescale_mode

    def _if_init_text_features_(self, text_labels):
        if text_labels is None and self.text_labels is None:
            raise ValueError("[Error] text_labels has to be provided for the first time call. Format like ['cat', 'dog', ...]")

        if text_labels is not None:
            self.text_labels = text_labels if self.formatter is None else [self.formatter.format(ii) for ii in text_labels]

        if text_labels is not None or self.text_features is None:
            text_features = functional.stack([functional.convert_to_tensor(self.caption_tokenizer(ii), dtype="int64") for ii in self.text_labels])
            text_features = self.text_model(text_features)
            self.text_features = text_features / functional.norm(text_features, axis=-1, keepdims=True)

    def __call__(self, image, text_labels=None):
        self._if_init_text_features_(text_labels=text_labels)

        image_features = self.image_model(self.image_model.preprocess_input(image))
        image_features /= functional.norm(image_features, axis=-1, keepdims=True)
        return functional.softmax(self.softmax_scale * image_features @ functional.transpose(self.text_features))
