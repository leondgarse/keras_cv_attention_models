import numpy as np
from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, functional, models, initializers, image_data_format
from keras_cv_attention_models.models import register_model
from keras_cv_attention_models.attention_layers import (
    activation_by_name,
    conv2d_no_bias,
    group_norm,
    add_pre_post_process,
    qkv_to_multi_head_channels_last_format,
    scaled_dot_product_attention,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

LAYER_NORM_EPSILON = 1e-6

PRETRAINED_DICT = {"unet": {"v1_5": "30b5e8755d2a04211980e608df14f133"}}


@backend.register_keras_serializable(package="kecam")
class SinusoidalTimeStepEmbedding(layers.Layer):
    def __init__(self, hidden_channels=320, max_period=10000, **kwargs):
        super().__init__(**kwargs)
        self.hidden_channels, self.max_period = hidden_channels, max_period

    def build(self, input_shape):
        # input_shape: [batch]
        half = self.hidden_channels // 2  # half the channels are sin and the other half is cos
        frequencies = np.exp(-np.log(self.max_period) * np.arange(0, half, dtype="float32") / half)[None]
        if hasattr(self, "register_buffer"):  # PyTorch
            self.register_buffer("frequencies", functional.convert_to_tensor(frequencies, dtype=self.compute_dtype), persistent=False)
        else:
            self.frequencies = functional.convert_to_tensor(frequencies, dtype=self.compute_dtype)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        cur = inputs[:, None] * self.frequencies
        return functional.concat([functional.cos(cur), functional.sin(cur)], axis=-1)

    def compute_output_shape(self, input_shape):
        return [None, self.hidden_channels]

    def get_config(self):
        base_config = super().get_config()
        base_config.update({"hidden_channels": self.hidden_channels, "max_period": self.max_period})
        return base_config


def cross_attention(inputs, condition=None, num_heads=4, head_dim=0, name=""):
    _, bb, cc = inputs.shape
    head_dim = head_dim if head_dim > 0 else cc // num_heads
    emded_dim = int(num_heads * head_dim)
    condition = inputs if condition is None else condition

    query = layers.Dense(emded_dim, use_bias=False, name=name and name + "query")(inputs)
    key = layers.Dense(emded_dim, use_bias=False, name=name and name + "key")(condition)
    value = layers.Dense(emded_dim, use_bias=False, name=name and name + "value")(condition)

    query, key, value = qkv_to_multi_head_channels_last_format(query, key, value, num_heads, data_format="channels_last")
    output_shape = [-1, bb, cc]
    return scaled_dot_product_attention(query, key, value, output_shape, out_weight=True, out_bias=True, name=name)


def attention_mlp_block(inputs, condition=None, mlp_ratio=4, num_heads=4, head_dim=0, name=""):
    nn = layers.LayerNormalization(axis=-1, epsilon=LAYER_NORM_EPSILON, name=name + "attn_ln")(inputs)  # "channels_first" also using axis=-1
    nn = cross_attention(nn, condition=None, num_heads=num_heads, head_dim=head_dim, name=name + "attn_")
    attn_out = layers.Add(name=name + "attn_out")([inputs, nn])

    """ Attention with condition """
    if condition is not None:
        nn = layers.LayerNormalization(axis=-1, epsilon=LAYER_NORM_EPSILON, name=name + "cond_attn_ln")(attn_out)  # "channels_first" also using axis=-1
        nn = cross_attention(nn, condition=condition, num_heads=num_heads, head_dim=head_dim, name=name + "cond_attn_")
        attn_out = layers.Add(name=name + "cond_attn_out")([attn_out, nn])

    """ Feed forward """
    input_channels = inputs.shape[-1]
    nn = layers.LayerNormalization(axis=-1, epsilon=LAYER_NORM_EPSILON, name=name + "ffn_ln")(attn_out)  # "channels_first" also using axis=-1
    nn = layers.Dense(input_channels * mlp_ratio * 2, use_bias=True, name=name + "ffn_gate_dense")(nn)
    fft, fft_gate = functional.split(nn, 2, axis=-1)
    nn = fft * activation_by_name(fft_gate, activation="gelu", name=name + "gate_")

    nn = layers.Dense(input_channels, use_bias=True, name=name + "mlp.down_proj")(nn)
    nn = layers.Add(name=name + "mlp_output")([attn_out, nn])
    return nn


def spatial_transformer_block(inputs, condition=None, num_attention_block=1, mlp_ratio=4, num_heads=4, head_dim=0, name=""):
    input_channels = inputs.shape[-1 if backend.image_data_format() == "channels_last" else 1]
    nn = group_norm(inputs, epsilon=LAYER_NORM_EPSILON, name=name + "in_layers_")
    nn = conv2d_no_bias(nn, input_channels, kernel_size=1, use_bias=True, name=name + "in_layers_")

    nn = nn if backend.image_data_format() == "channels_last" else layers.Permute([2, 3, 1])(nn)
    block_height, block_width = nn.shape[1], nn.shape[2]
    nn = functional.reshape(nn, [-1, block_height * block_width, nn.shape[-1]])

    for attention_block_id in range(num_attention_block):
        block_name = name + "{}_".format(attention_block_id + 1)
        nn = attention_mlp_block(nn, condition, mlp_ratio=mlp_ratio, num_heads=num_heads, head_dim=head_dim, name=block_name)
    nn = functional.reshape(nn, [-1, block_height, block_width, nn.shape[-1]])
    nn = nn if backend.image_data_format() == "channels_last" else layers.Permute([3, 1, 2])(nn)
    nn = conv2d_no_bias(nn, input_channels, kernel_size=1, use_bias=True, name=name + "out_layers_")
    return layers.Add(name=name + "output")([inputs, nn])


def res_block(inputs, time_embedding=None, out_channels=-1, epsilon=1e-5, activation="swish", name=""):
    input_channels = inputs.shape[-1 if backend.image_data_format() == "channels_last" else 1]
    out_channels = out_channels if out_channels > 0 else input_channels
    if input_channels == out_channels:
        short = inputs
    else:
        short = conv2d_no_bias(inputs, out_channels, kernel_size=1, use_bias=True, name=name + "short_")

    nn = group_norm(inputs, epsilon=epsilon, name=name + "in_layers_")
    nn = activation_by_name(nn, activation=activation, name=name + "in_layers_")
    nn = conv2d_no_bias(nn, out_channels, kernel_size=3, use_bias=True, padding="SAME", name=name + "in_layers_")

    if time_embedding is not None:
        emb = activation_by_name(time_embedding, activation=activation, name=name + "emb_layers_")
        emb = layers.Dense(out_channels, name=name + "emb_layers_dense")(emb)
        emb = emb[:, None, None, :] if backend.image_data_format() == "channels_last" else emb[:, :, None, None]
        nn = nn + emb

    nn = group_norm(nn, epsilon=epsilon, name=name + "out_layers_")
    nn = activation_by_name(nn, activation=activation, name=name + "out_layers_")
    nn = conv2d_no_bias(nn, out_channels, kernel_size=3, use_bias=True, padding="SAME", name=name + "out_layers_")
    return layers.Add(name=name + "out")([nn, short])


@register_model
def UNet(
    input_shape=(64, 64, 4),
    num_blocks=[2, 2, 2, 2],
    hidden_channels=320,
    hidden_expands=[1, 2, 4, 4],
    num_attention_blocks=[1, 1, 1, 0],  # attention_blocks after each res_block in each stack
    num_heads=8,
    mlp_ratio=4,
    conditional_embedding=768,
    labels_embedding=0,
    pretrained="v1_5",
    model_name="unet",
    kwargs=None,  # Not using, recieving parameter
):
    # Regard input_shape as force using original shape if len(input_shape) == 4,
    # else assume channel dimension is the one with min value in input_shape, and put it first or last regarding image_data_format
    inputs = layers.Input(backend.align_input_shape_by_image_data_format(input_shape))
    channel_axis = -1 if backend.image_data_format() == "channels_last" else 1
    time_steps = layers.Input([])
    condition = layers.Input([None, conditional_embedding])

    time_embedding = SinusoidalTimeStepEmbedding(hidden_channels=hidden_channels, name="time_embedding")(time_steps)
    time_embedding = layers.Dense(hidden_channels * 4, name="time_embed_1_dense")(time_embedding)
    time_embedding = activation_by_name(time_embedding, activation="swish", name="time_embed_")
    time_embedding = layers.Dense(hidden_channels * 4, name="time_embed_2_dense")(time_embedding)

    nn = conv2d_no_bias(inputs, hidden_channels, kernel_size=3, use_bias=True, padding="SAME", name="latents_")

    """ Down blocks """
    skip_connections = [nn]
    for stack_id, (num_block, hidden_expand, num_attention_block) in enumerate(zip(num_blocks, hidden_expands, num_attention_blocks)):
        stack_name = "stack{}_".format(stack_id + 1)
        out_channels = hidden_expand * hidden_channels
        if stack_id > 0:
            nn = conv2d_no_bias(nn, nn.shape[channel_axis], kernel_size=3, strides=2, use_bias=True, padding="same", name=stack_name + "downsample_")
            skip_connections.append(nn)

        for block_id in range(num_block):
            block_name = stack_name + "down_block{}_".format(block_id + 1)
            nn = res_block(nn, time_embedding, out_channels, name=block_name)
            if num_attention_block > 0:
                nn = spatial_transformer_block(nn, condition, num_attention_block, mlp_ratio=mlp_ratio, num_heads=num_heads, name=block_name + "attn_")
            skip_connections.append(nn)
    # print(f">>>> {[ii.shape for ii in skip_connections] = }")

    """ Middle blocks """
    nn = res_block(nn, time_embedding, name="middle_block_1_")
    nn = spatial_transformer_block(nn, condition, num_attention_block=1, name="middle_block_attn_")
    nn = res_block(nn, time_embedding, name="middle_block_2_")

    """ Up blocks """
    for stack_id, (num_block, hidden_expand, num_attention_block) in enumerate(zip(num_blocks[::-1], hidden_expands[::-1], num_attention_blocks[::-1])):
        stack_name = "stack{}_".format(len(num_blocks) + stack_id + 1)
        out_channels = hidden_expand * hidden_channels
        if stack_id > 0:
            nn = layers.UpSampling2D(size=2, name=stack_name + "upsample_")(nn)
            nn = conv2d_no_bias(nn, nn.shape[channel_axis], kernel_size=3, strides=1, use_bias=True, padding="same", name=stack_name + "upsample_")

        for block_id in range(num_block + 1):
            block_name = stack_name + "up_block{}_".format(block_id + 1)
            skip_connection = skip_connections.pop(-1)
            nn = functional.concat([nn, skip_connection], axis=channel_axis)

            nn = res_block(nn, time_embedding, out_channels, name=block_name)
            if num_attention_block > 0:
                nn = spatial_transformer_block(nn, condition, num_attention_block, mlp_ratio=mlp_ratio, num_heads=num_heads, name=block_name + "attn_")

    """ Output blocks """
    output_channels = inputs.shape[channel_axis]
    nn = group_norm(nn, name="output_")
    nn = activation_by_name(nn, activation="swish", name="output_")
    outputs = conv2d_no_bias(nn, output_channels, kernel_size=3, use_bias=True, padding="SAME", name="output_")

    model = models.Model([inputs, time_steps, condition], outputs, name=model_name)
    reload_model_weights(model, PRETRAINED_DICT, "stable_diffusion", pretrained)
    return model
