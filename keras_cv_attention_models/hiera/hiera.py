import numpy as np
from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models, functional, image_data_format
from keras_cv_attention_models.models import register_model
from keras_cv_attention_models.attention_layers import (
    conv2d_no_bias,
    drop_block,
    mlp_block,
    scaled_dot_product_attention,
    PositionalEmbedding,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

LAYER_NORM_EPSILON = 1e-6

PRETRAINED_DICT = {
    "hiera_base": {"mae_in1k_ft1k": {224: "7998153d4cf4a2571e1268c8be1d9324"}},
    "hiera_base_plus": {"mae_in1k_ft1k": {224: "5492209f8d7f8632e5db218e2b32496d"}},
    "hiera_huge": {"mae_in1k_ft1k": {224: "2226812b40266ab28e4394849ba7690f"}},
    "hiera_large": {"mae_in1k_ft1k": {224: "68a3f6329f54f5f3bf7737989e3b67df"}},
    "hiera_small": {"mae_in1k_ft1k": {224: "aaba43ec87328a487a06f24d67a4e56d"}},
    "hiera_tiny": {"mae_in1k_ft1k": {224: "d7b913744fc2371e489f536dc236ae03"}},
}


def mhsa_with_window_extracted_and_strides(
    inputs, num_heads=4, key_dim=0, out_shape=None, window_size_prod=-1, strides_prod=1, qkv_bias=True, out_bias=True, attn_dropout=0, name=None
):
    _, blocks, cc = inputs.shape
    out_shape = cc if out_shape is None else out_shape
    key_dim = key_dim if key_dim > 0 else out_shape // num_heads  # Note: different from others using input_channels
    qkv_out = num_heads * key_dim
    qk_scale = 1.0 / (float(key_dim) ** 0.5)
    window_size_prod = window_size_prod if window_size_prod > 0 else blocks
    window_blocks = blocks // window_size_prod
    # print(f"{blocks = }, {window_blocks = }, {window_size_prod = }, {num_heads = }")

    qkv = layers.Dense(qkv_out * 3, use_bias=qkv_bias, name=name and name + "qkv")(inputs)
    query, key, value = functional.split(qkv, 3, axis=-1)

    if strides_prod > 1:
        query = functional.reshape(query, [-1, strides_prod, window_size_prod // strides_prod, window_blocks * num_heads, key_dim])
        query = functional.reduce_max(query, axis=1)
        query = functional.transpose(query, [0, 2, 1, 3])
    else:
        query = functional.transpose(functional.reshape(query, [-1, window_size_prod, window_blocks * num_heads, key_dim]), [0, 2, 1, 3])
    key = functional.transpose(functional.reshape(key, [-1, window_size_prod, window_blocks * num_heads, key_dim]), [0, 2, 3, 1])
    value = functional.transpose(functional.reshape(value, [-1, window_size_prod, window_blocks * num_heads, key_dim]), [0, 2, 1, 3])

    output_shape = [-1, blocks // strides_prod, out_shape]
    return scaled_dot_product_attention(query, key, value, output_shape, out_bias=out_bias, dropout=attn_dropout, name=name)


def attention_mlp_block(inputs, out_channels=-1, num_heads=4, window_size_prod=-1, strides_prod=1, mlp_ratio=4, drop_rate=0, activation="gelu", name=""):
    # print(f">>>> {inputs.shape = }, {drop_rate = }")
    input_channels = inputs.shape[-1]
    out_channels = out_channels if out_channels > 0 else input_channels

    pre = layers.LayerNormalization(axis=-1, epsilon=LAYER_NORM_EPSILON, name=name + "attn_ln")(inputs)  # "channels_first" also using axis=-1
    if strides_prod > 1 or out_channels != input_channels:
        short = pre if out_channels == input_channels else layers.Dense(out_channels, name=name + "short_dense")(pre)
        short = functional.reduce_max(functional.reshape(short, [-1, strides_prod, short.shape[1] // strides_prod, short.shape[-1]]), axis=1)
    else:
        short = inputs

    """ Attention """
    attn = mhsa_with_window_extracted_and_strides(
        pre, num_heads=num_heads, out_shape=out_channels, window_size_prod=window_size_prod, strides_prod=strides_prod, name=name + "attn_"
    )
    attn = drop_block(attn, drop_rate)
    attn_out = layers.Add(name=name + "attn_out")([short, attn])

    """ MLP """
    nn = layers.LayerNormalization(axis=-1, epsilon=LAYER_NORM_EPSILON, name=name + "mlp_ln")(attn_out)  # "channels_first" also using axis=-1
    nn = mlp_block(nn, hidden_dim=int(out_channels * mlp_ratio), activation=activation, name=name + "mlp_")
    nn = drop_block(nn, drop_rate)
    nn = layers.Add(name=name + "mlp_output")([attn_out, nn])
    return nn


def unroll(inputs, strides=[2, 2, 2]):
    """
    inputs: [batch, height, width, channels], strides: [2, 2, 2]
    -> [batch, height // 8, 2_h3, 2_h2, 2_h1, width // 8, 2_w3, 2_w2, 2_w1, channels]
    -> [batch, 2_h1, 2_w1, 2_h2, 2_w2, 2_h3, 2_w3, height // 8, width // 8, channels]  # [0, 4, 8, 3, 7, 2, 6, 1, 5, 9]
    -> [batch, height * width, channels]
    """
    height, width, channels = inputs.shape[1:]
    nn = inputs
    for ii in strides:
        nn = functional.reshape(nn, [-1, nn.shape[-3] // ii, ii, nn.shape[-2] // ii, ii, nn.shape[-1]])
        nn = functional.transpose(nn, [0, 2, 4, 1, 3, 5])
    return functional.reshape(nn, [-1, height * width, channels])

    # height_strided = height // np.prod(strides)
    # width_strided = width // np.prod(strides)
    # inner_shape = [-1, height_strided, *strides, width_strided, *strides, channels]
    # nn = functional.reshape(inputs, inner_shape)
    #
    # strides_len = len(strides) + 1
    # perm = [0] + np.ravel([[ii, ii + strides_len] for ii in range(strides_len, 0, -1)]).tolist() + [2 * strides_len + 1]  # [0, 4, 8, 3, 7, 2, 6, 1, 5, 9]
    # nn = functional.transpose(nn, perm)
    #
    # return functional.reshape(nn, [-1, height * width, channels])


def reroll(inputs, strides, height=-1):
    pass
    # for ii in range(strides):
    #     inputs = functional.reshape(inputs, ii, ii, inputs.shape[1:])


def Hiera(
    num_blocks=[1, 2, 7, 2],
    embed_dim=96,
    num_heads=[1, 2, 4, 8],
    use_window_attentions=[True, True, [True, False], False],  # [True, False] means first one is True, others are False
    mlp_ratio=4,
    strides=[1, 2, 2, 2],
    # window_ratios=[8, 4, 1, 1],
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="gelu",
    drop_connect_rate=0,
    dropout=0,
    extract_features=False,
    classifier_activation="softmax",
    pretrained=None,
    model_name="hiera",
    kwargs=None,
):
    # Regard input_shape as force using original shape if len(input_shape) == 4,
    # else assume channel dimension is the one with min value in input_shape, and put it first or last regarding image_data_format
    input_shape = backend.align_input_shape_by_image_data_format(input_shape)
    inputs = layers.Input(input_shape)
    window_size_prod = int(np.prod(strides) ** 2)  # Total downsample rates after stem

    """ forward_embeddings """
    nn = conv2d_no_bias(inputs, embed_dim, 7, strides=4, padding="same", use_bias=True, name="stem_")
    nn = nn if image_data_format() == "channels_last" else layers.Permute([2, 3, 1])(nn)  # channels_first -> channels_last
    nn = PositionalEmbedding(input_height=nn.shape[1], name="positional_embedding")(nn)
    height, width = nn.shape[1:-1]
    nn = unroll(nn, strides=strides[1:])
    # window_ratios = (window_ratios[0] * window_ratios[1]) if isinstance(window_ratios, (list, tuple)) else window_ratios

    """ stages """
    total_blocks = sum(num_blocks)
    global_block_id = 0
    features = []
    for stack_id, (num_block, stride, use_window_attention) in enumerate(zip(num_blocks, strides, use_window_attentions)):
        stack_name = "stack{}_".format(stack_id + 1)
        cur_out_channels = embed_dim * (2**stack_id)
        cur_num_heads = num_heads[stack_id] if isinstance(num_heads, (list, tuple)) else num_heads
        stack_use_window_attention = use_window_attention if isinstance(use_window_attention, (list, tuple)) else [use_window_attention]
        stack_use_window_attention = stack_use_window_attention + stack_use_window_attention[-1:] * (num_block - len(stack_use_window_attention))
        window_size_prod //= stride**2
        for block_id in range(num_block):
            block_name = stack_name + "block{}_".format(block_id + 1)
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            strides_prod = (stride**2) if block_id == 0 else 1
            cur_window_size_prod = int(window_size_prod * strides_prod) if stack_use_window_attention[block_id] else -1
            nn = attention_mlp_block(
                nn, cur_out_channels, cur_num_heads, cur_window_size_prod, strides_prod, mlp_ratio, block_drop_rate, activation=activation, name=block_name
            )
            global_block_id += 1
        height, width = height // stride, width // stride  # [TODO] reroll

    nn = functional.reshape(nn, [-1, height, width, nn.shape[-1]])
    nn = nn if image_data_format() == "channels_last" else layers.Permute([3, 1, 2])(nn)  # channels_last -> channels_first

    if num_classes > 0:
        nn = layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        nn = layers.LayerNormalization(axis=-1, epsilon=LAYER_NORM_EPSILON, name="post_ln")(nn)
        if dropout > 0:
            nn = layers.Dropout(dropout, name="head_drop")(nn)
        nn = layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="predictions")(nn)

    model = models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    if extract_features:
        model.extract_features = lambda: features
    reload_model_weights(model, PRETRAINED_DICT, "hiera", pretrained, PositionalEmbedding)
    return model


@register_model
def HieraTiny(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="mae_in1k_ft1k", **kwargs):
    return Hiera(**locals(), model_name="hiera_tiny", **kwargs)


@register_model
def HieraSmall(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="mae_in1k_ft1k", **kwargs):
    num_blocks = [1, 2, 11, 2]
    return Hiera(**locals(), model_name="hiera_small", **kwargs)


@register_model
def HieraBase(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="mae_in1k_ft1k", **kwargs):
    num_blocks = [2, 3, 16, 3]
    return Hiera(**locals(), model_name="hiera_base", **kwargs)


@register_model
def HieraBasePlus(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="mae_in1k_ft1k", **kwargs):
    num_blocks = [2, 3, 16, 3]
    embed_dim = 112
    num_heads = [2, 4, 8, 16]
    return Hiera(**locals(), model_name="hiera_base_plus", **kwargs)


@register_model
def HieraLarge(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="mae_in1k_ft1k", **kwargs):
    num_blocks = [2, 6, 36, 4]
    embed_dim = 144
    num_heads = [2, 4, 8, 16]
    return Hiera(**locals(), model_name="hiera_large", **kwargs)


@register_model
def HieraHuge(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="mae_in1k_ft1k", **kwargs):
    num_blocks = [2, 6, 36, 4]
    embed_dim = 256
    num_heads = [4, 8, 16, 32]
    return Hiera(**locals(), model_name="hiera_huge", **kwargs)
