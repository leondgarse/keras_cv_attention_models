import numpy as np
from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models, functional
from keras_cv_attention_models.models import register_model
from keras_cv_attention_models.download_and_load import reload_model_weights
from keras_cv_attention_models.attention_layers import activation_by_name, CausalMask
from keras_cv_attention_models.gpt2.gpt2 import RunPrediction


PRETRAINED_DICT = {
    "llama2_110m": {"tiny_stories": "00a82a7649e9b29af65b500fa4ceceb8"},
    "llama2_15m": {"tiny_stories": "a8db53783fa9e31ecb821f1fd9f7d966"},
    "llama2_42m": {"tiny_stories": "580bdbf3c8ddf8f907c07dbafe526dfe"},
}


@backend.register_keras_serializable(package="kecam/llama2")
class PositionalEncodingFourierRot1D(layers.Layer):
    def __init__(self, max_block_size, temperature=1e4, **kwargs):
        super().__init__(**kwargs)
        self.temperature, self.max_block_size = float(temperature), max_block_size

    def build(self, input_shape):
        # input: `[batch, ..., attn_height * attn_width, num_heads, channels // num_heads // 2, 2]`.
        # print(input_shape)
        self.channels = input_shape[-2] * input_shape[-1]
        pos_filters = self.channels // 2
        dim_t = self.temperature ** (np.arange(pos_filters, dtype="float32") / pos_filters)  # (filters,)
        grid = np.expand_dims(np.arange(self.max_block_size, dtype="float32"), -1) / dim_t
        pos_sin, pos_cos = np.expand_dims(np.sin(grid), -2), np.expand_dims(np.cos(grid), -2)
        # print(f"{pos_sin.shape = }, {pos_cos.shape = }, {height = }, {width = }, {self.channels = }")

        if hasattr(self, "register_buffer"):  # PyTorch
            self.register_buffer("pos_sin", functional.convert_to_tensor(pos_sin, dtype=self.compute_dtype), persistent=False)
            self.register_buffer("pos_cos", functional.convert_to_tensor(pos_cos, dtype=self.compute_dtype), persistent=False)
        else:
            self.pos_sin = functional.convert_to_tensor(pos_sin, dtype=self.compute_dtype)
            self.pos_cos = functional.convert_to_tensor(pos_cos, dtype=self.compute_dtype)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        left, right = functional.unstack(inputs, axis=-1)
        pos_cos, pos_sin = self.pos_cos[: left.shape[-3]], self.pos_sin[: left.shape[-3]]
        out = functional.stack([left * pos_cos - right * pos_sin, right * pos_cos + left * pos_sin], axis=-1)
        return out

    def get_config(self):
        base_config = super().get_config()
        base_config.update({"temperature": self.temperature, "max_block_size": self.max_block_size})
        return base_config


@backend.register_keras_serializable(package="kecam/llama2")
class RMSNorm(layers.Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super().__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.gamma = self.add_weight(name="gamma", shape=(input_shape[-1],), initializer="ones", trainable=True)
        super().build(input_shape)

    def call(self, inputs):
        norm = inputs * functional.rsqrt(functional.reduce_mean(inputs**2, keepdims=True, axis=-1) + self.epsilon)
        return norm * self.gamma

    def get_config(self):
        base_config = super().get_config()
        base_config.update({"epsilon": self.epsilon})
        return base_config


def apply_positional_encoding_rotary(inputs, pos_emb_layer, with_cls_token=True):
    """Reshape is separated out from PositionalEncodingFourierRot1D for setting as dynamic"""
    num_heads = inputs.shape[-2]
    nn = layers.Reshape([-1, num_heads, inputs.shape[-1] // 2, 2])(inputs)
    nn = pos_emb_layer(nn)
    out = layers.Reshape([-1, num_heads, inputs.shape[-1]])(nn)
    return out


def causal_self_attention(inputs, block_size, num_heads, use_bias=False, dropout=0, name=""):
    input_channels = inputs.shape[-1]
    key_dim = input_channels // num_heads
    qq_scale = 1.0 / (float(key_dim) ** 0.5)

    query = layers.Dense(input_channels, use_bias=use_bias, name=name + "wq")(inputs)
    key = layers.Dense(input_channels, use_bias=use_bias, name=name + "wk")(inputs)
    value = layers.Dense(input_channels, use_bias=use_bias, name=name + "wv")(inputs)

    # Create a new one every time, as there's no weights for this layer
    rope = PositionalEncodingFourierRot1D(max_block_size=block_size, name=name + "rope")
    query = layers.Reshape([-1, num_heads, key_dim])(query)
    query = apply_positional_encoding_rotary(query, rope)
    query = functional.transpose(query, [0, 2, 1, 3])

    key = layers.Reshape([-1, num_heads, key_dim])(key)
    key = apply_positional_encoding_rotary(key, rope)
    key = functional.transpose(key, [0, 2, 3, 1])

    value = functional.transpose(layers.Reshape([-1, num_heads, key_dim])(value), [0, 2, 1, 3])

    attn = (query @ key) * qq_scale
    attn = CausalMask(block_size=block_size)(attn)
    attn = layers.Softmax(axis=-1, name=name + "attention_scores")(attn)
    attn_out = attn @ value

    output = functional.transpose(attn_out, [0, 2, 1, 3])
    output = layers.Reshape([-1, input_channels])(output)
    output = layers.Dense(input_channels, use_bias=use_bias, name=name + "wo")(output)
    output = layers.Dropout(dropout)(output)
    return output


def attention_fft_block(inputs, block_size, num_heads, hidden_divisible=32, use_bias=False, dropout=0, activation="swish", name=""):
    input_channels = inputs.shape[-1]
    attn = RMSNorm(name=name + "attention_norm")(inputs)
    attn = causal_self_attention(attn, block_size, num_heads, use_bias, dropout, name=name + "attention.")
    attn_out = inputs + attn

    hidden_dim = 2 * 4 * input_channels // 3
    hidden_dim = hidden_divisible * ((hidden_dim + hidden_divisible - 1) // hidden_divisible)
    # print(f"{input_channels = }, {hidden_divisible = }, {hidden_dim = }")

    fft = RMSNorm(name=name + "ffn_norm")(attn_out)
    fft_1 = layers.Dense(hidden_dim, use_bias=use_bias, name=name + "feed_forward.w1")(fft)
    fft_1 = activation_by_name(fft_1, activation=activation, name=name + "feed_forward.w1.")
    fft_3 = layers.Dense(hidden_dim, use_bias=use_bias, name=name + "feed_forward.w3")(fft)
    fft = fft_1 * fft_3
    fft = layers.Dense(input_channels, use_bias=use_bias, name=name + "feed_forward.w2")(fft)
    fft = layers.Dropout(dropout)(fft)

    return layers.Add(name=name + "output")([attn_out, fft])


def LLaMA2(
    num_blocks=12,
    embedding_size=768,
    hidden_divisible=32,
    num_heads=12,
    block_use_bias=False,
    vocab_size=50304,
    max_block_size=2048,
    include_top=True,
    dropout=0.0,
    activation="swish",
    pretrained=None,
    model_name="llama2",
    kwargs=None,
):
    inputs = layers.Input([None], dtype="int64")
    tok_emb = layers.Embedding(vocab_size, embedding_size, name="tok_embeddings")(inputs)
    nn = layers.Dropout(dropout)(tok_emb)

    for block_id in range(num_blocks):
        nn = attention_fft_block(nn, max_block_size, num_heads, hidden_divisible, block_use_bias, dropout, activation, name="blocks.{}.".format(block_id))
    nn = RMSNorm(name="norm")(nn)

    if include_top:
        nn = layers.Dense(vocab_size, use_bias=False, name="output")(nn)

    model = models.Model(inputs, nn, name=model_name)
    model.max_block_size = max_block_size  # or model.get_layer('pos_idx').block_size
    model.run_prediction = RunPrediction(model, tokenizer="SentencePieceTokenizer")
    if pretrained.endswith(".pt") or pretrained.endswith(".pth"):
        load_weights_from_pytorch(model, source_pt_path=pretrained, save_path="~/.keras/models")
    else:
        reload_model_weights(model, PRETRAINED_DICT, "llama2", pretrained)
    return model


@register_model
def LLaMA2_15M(max_block_size=256, vocab_size=32000, include_top=True, activation="swish", pretrained="tiny_stories", **kwargs):
    num_blocks = 6
    embedding_size = 288
    num_heads = 6
    return LLaMA2(**locals(), **kwargs, model_name="llama2_15m")


@register_model
def LLaMA2_42M(max_block_size=1024, vocab_size=32000, include_top=True, activation="swish", pretrained="tiny_stories", **kwargs):
    num_blocks = 8
    embedding_size = 512
    num_heads = 8
    return LLaMA2(**locals(), **kwargs, model_name="llama2_42m")


@register_model
def LLaMA2_110M(max_block_size=1024, vocab_size=32000, include_top=True, activation="swish", pretrained="tiny_stories", **kwargs):
    num_blocks = 12
    embedding_size = 768
    num_heads = 12
    return LLaMA2(**locals(), **kwargs, model_name="llama2_110m")


@register_model
def LLaMA2_7B(max_block_size=2048, vocab_size=50257, include_top=True, activation="swish", pretrained=None, **kwargs):
    num_blocks = 32
    embedding_size = 4096
    hidden_divisible = 256
    num_heads = 32
    return LLaMA2(**locals(), **kwargs, model_name="llama2_7b")


def load_weights_from_pytorch(model, source_pt_path, save_name=None, save_path=".", force=False):
    import os

    pretrained = os.path.splitext(os.path.basename(source_pt_path))[0]
    save_name = save_name if save_name is not None else "{}_{}.h5".format(model.name, pretrained)
    save_path = os.path.join(os.path.expanduser(save_path), save_name)
    if not force and os.path.exists(save_path):
        print("Load previously saved model:", save_path)
        model.load_weights(save_path)
        return
    else:
        print("Convert and load weights from", source_pt_path)

    import torch

    source_state_dict = torch.load(source_pt_path, map_location=torch.device("cpu"))
    source_state_dict = source_state_dict.get("state_dict", source_state_dict.get("model", source_state_dict))

    """ state_dict_stack_by_layer """
    stacked_state_dict = {}
    for kk, vv in source_state_dict.items():
        if kk.endswith(".attn.bias") or kk.endswith(".attn.masked_bias"):
            continue

        split_kk = kk.split(".")
        vv = vv.numpy() if hasattr(vv, "numpy") else vv

        # split_kk[-1] in ["weight", "bias", "running_mean", "running_var", "gain"]
        layer_name = ".".join(split_kk[:-1])
        stacked_state_dict.setdefault(layer_name, []).append(vv)

    """ keras_reload_stacked_state_dict """
    target_names = [ii.name for ii in model.layers if len(ii.weights) != 0]
    for target_name in target_names:
        source_name = target_name.replace("blocks.", "layers.")
        print(">>>> Load {} weights from {}".format(target_name, source_name))
        target_layer = model.get_layer(target_name)
        source_weights = stacked_state_dict[source_name]
        if isinstance(target_layer, layers.Dense):
            source_weights = [ii.T for ii in source_weights]
        print("    Source: {}, Target: {}".format([ii.shape for ii in source_weights], [ii.shape for ii in target_layer.get_weights()]))

        if hasattr(target_layer, "set_weights_channels_last"):
            target_layer.set_weights_channels_last(source_weights)  # Kecam PyTorch backend
        else:
            target_layer.set_weights(source_weights)

    print(">>>> Save to:", save_path)
    if hasattr(model, "save"):
        model.save(save_path)
    else:
        model.save_weights(save_path)  # Kecam PyTorch backend


if __name__ == "__test__":
    from keras_cv_attention_models import llama2

    mm = llama2.LLaMA2_42M()
    mm.run_prediction("As evening fell, a maiden stood at the edge of a wood. In her hands,", num_samples=3)
