import tensorflow as tf
from tensorflow import keras
from keras_cv_attention_models.attention_layers import (
    batchnorm_with_activation,
    conv2d_no_bias,
    ChannelAffine,
    depthwise_conv2d_no_bias,
    drop_block,
    drop_connect_rates_split,
    mlp_block,
    multi_head_self_attention,
    output_block,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

LAYER_NORM_EPSILON = 1e-5
PRETRAINED_DICT = {
    "mobilevit_xxs": {"imagenet": "f9d1d4f7329726b4bb9069cd631a152e"},
    "mobilevit_xs": {"imagenet": "dcd8591668946aa4ddb0159dfe65cc2b"},
    "mobilevit_s": {"imagenet": "55f1051035ecd14e90ae3df80fd0c7f3"},
}


def bottle_in_linear_out_block(inputs, out_channel, strides=1, expand_ratio=4, use_shortcut=False, drop_rate=0, activation="swish", name=""):
    hidden_dim = int(inputs.shape[-1] * expand_ratio)
    deep = conv2d_no_bias(inputs, hidden_dim, kernel_size=1, strides=1, name=name + "deep_1_")
    deep = batchnorm_with_activation(deep, activation=activation, name=name + "deep_1_")
    deep = depthwise_conv2d_no_bias(deep, kernel_size=3, strides=strides, padding="SAME", name=name + "deep_2_")
    deep = batchnorm_with_activation(deep, activation=activation, name=name + "deep_2_")
    deep = conv2d_no_bias(deep, out_channel, kernel_size=1, strides=1, name=name + "deep_3_")
    deep = batchnorm_with_activation(deep, activation=None, name=name + "deep_3_")
    deep = drop_block(deep, drop_rate=drop_rate, name=name + "deep_")

    out = keras.layers.Add()([inputs, deep]) if use_shortcut else deep
    return out


def mhsa_mlp_block(
    inputs, out_channel, num_heads=4, qkv_bias=True, mlp_ratio=4, mlp_drop_rate=0, attn_drop_rate=0, drop_rate=0, layer_scale=-1, activation="gelu", name=None
):
    attn = keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON, name=name and name + "attn_ln")(inputs)
    attn = multi_head_self_attention(attn, num_heads, qkv_bias=qkv_bias, out_bias=True, attn_dropout=attn_drop_rate, name=name and name + "attn_mhsa_")
    attn = ChannelAffine(use_bias=False, weight_init_value=layer_scale, name=name and name + "1_gamma")(attn) if layer_scale >= 0 else attn
    attn = drop_block(attn, drop_rate=drop_rate, name=name and name + "attn_")
    attn_out = keras.layers.Add(name=name and name + "attn_out")([inputs, attn])

    mlp = keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON, name=name and name + "mlp_ln")(attn_out)
    mlp = mlp_block(mlp, int(out_channel * mlp_ratio), drop_rate=mlp_drop_rate, use_conv=False, activation=activation, name=name and name + "mlp_")
    mlp = ChannelAffine(use_bias=False, weight_init_value=layer_scale, name=name and name + "2_gamma")(mlp) if layer_scale >= 0 else mlp
    mlp = drop_block(mlp, drop_rate=drop_rate, name=name and name + "mlp_")
    return keras.layers.Add(name=name and name + "output")([attn_out, mlp])


def transformer_pre_process(inputs, out_channel, patch_size=2, activation="swish", name=""):
    nn = conv2d_no_bias(inputs, inputs.shape[-1], kernel_size=3, strides=1, padding="SAME", name=name + "pre_1_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "pre_1_")
    nn = conv2d_no_bias(nn, out_channel, kernel_size=1, strides=1, name=name + "pre_2_")

    # Extract patchs, limit transpose permute length <= 4
    # [batch, height, width, channel] -> [batch, height // 2, 2, width // 2, 2, channel] -> [batch * 4, height // 2, width // 2, channel]
    patch_hh, patch_ww, channel = int(tf.math.ceil(nn.shape[1] / patch_size)), int(tf.math.ceil(nn.shape[2] / patch_size)), nn.shape[-1]
    # print(f"transformer_pre_process before resize: {nn.shape = }")
    if patch_hh * patch_size != nn.shape[1] or patch_ww * patch_size != nn.shape[2]:
        nn = tf.image.resize(nn, [patch_hh * patch_size, patch_ww * patch_size], method="bilinear")
    # print(f"transformer_pre_process after resize: {nn.shape = }")
    nn = tf.reshape(nn, [-1, patch_ww, patch_size, channel])  # [batch * patch_hh * h_patch_size, patch_ww, w_patch_size, channel]
    nn = tf.transpose(nn, [0, 2, 1, 3])  # [batch * patch_hh * h_patch_size, w_patch_size, patch_ww, channel]
    nn = tf.reshape(nn, [-1, patch_hh, patch_size * patch_size, patch_ww * channel])  # [batch, patch_hh, h_patch_size * w_patch_size, patch_ww * channel]
    nn = tf.transpose(nn, [0, 2, 1, 3])  # [batch, h_patch_size * w_patch_size, patch_hh, patch_ww * channel]
    nn = tf.reshape(nn, [-1, patch_hh, patch_ww, channel])
    return nn


def transformer_post_process(inputs, pre_attn, out_channel, patch_size=2, activation="swish", name=""):
    nn = keras.layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON, name=name + "post_ln")(inputs)

    # [batch * 4, height // 2, width // 2, channel] -> [batch, height // 2, 2, width // 2, 2, channel] -> [batch, height, width, channel]
    patch_hh, patch_ww, channel = nn.shape[1], nn.shape[2], nn.shape[-1]
    nn = tf.reshape(nn, [-1, patch_size * patch_size, patch_hh, patch_ww * channel])  # [batch, h_patch_size * w_patch_size, patch_hh, patch_ww * channel]
    nn = tf.transpose(nn, [0, 2, 1, 3])  # [batch, patch_hh, h_patch_size * w_patch_size, patch_ww * channel]
    nn = tf.reshape(nn, [-1, patch_size, patch_ww, channel])  # [batch * patch_hh * h_patch_size, w_patch_size, patch_ww, channel]
    nn = tf.transpose(nn, [0, 2, 1, 3])  # [batch * patch_hh * h_patch_size, patch_ww, w_patch_size, channel]
    nn = tf.reshape(nn, [-1, patch_hh * patch_size, patch_ww * patch_size, channel])
    # print(f"transformer_post_process before resize: {nn.shape = }")
    if nn.shape[1] != pre_attn.shape[1] or nn.shape[2] != pre_attn.shape[2]:
        nn = tf.image.resize(nn, [pre_attn.shape[1], pre_attn.shape[2]], method="bilinear")
    # print(f"transformer_post_process after resize: {nn.shape = }")

    nn = conv2d_no_bias(nn, out_channel, kernel_size=1, strides=1, name=name + "post_1_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "post_1_")
    nn = tf.concat([pre_attn, nn], axis=-1)
    nn = conv2d_no_bias(nn, out_channel, kernel_size=3, strides=1, padding="SAME", name=name + "post_2_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "post_2_")
    return nn


def stack(
    inputs, num_block, out_channel, is_conv_block=True, stride=2, expand_ratio=4, attn_channel=0, stack_drop=0, layer_scale=-1, activation="swish", name=""
):
    nn = inputs
    for block_id in range(num_block):
        block_name = name + "block{}_".format(block_id + 1)
        stride = stride if block_id == 0 else 1
        use_shortcut = False if stride != 1 or nn.shape[-1] != out_channel else True
        block_drop_rate = stack_drop[block_id] if isinstance(stack_drop, (list, tuple)) else stack_drop
        if is_conv_block or block_id == 0:  # First transformer block is also a conv block .
            nn = bottle_in_linear_out_block(nn, out_channel, stride, expand_ratio, use_shortcut, block_drop_rate, activation=activation, name=block_name)
        else:
            if block_id == 1:  # pre
                pre_attn = nn
                nn = transformer_pre_process(nn, attn_channel, activation=activation, name=block_name)
            num_heads, qkv_bias, mlp_ratio = 4, True, 2
            nn = mhsa_mlp_block(nn, attn_channel, num_heads, qkv_bias, mlp_ratio, layer_scale=layer_scale, activation=activation, name=block_name)
            if block_id == num_block - 1:  # post
                nn = transformer_post_process(nn, pre_attn, out_channel, activation=activation, name=block_name)
    return nn


def MobileViT(
    num_blocks=[1, 3, 3, 5, 4],
    out_channels=[32, 64, 96, 128, 160],
    attn_channels=[0, 0, 144, 192, 240],  # Can be a list matching out_channels, or a float number for expansion ratio of out_channels
    block_types=["conv", "conv", "transform", "transform", "transform"],
    strides=[1, 2, 2, 2, 2],
    expand_ratio=4,
    stem_width=16,
    output_num_features=640,
    layer_scale=-1,
    input_shape=(256, 256, 3),
    num_classes=1000,
    activation="swish",
    drop_connect_rate=0,
    classifier_activation="softmax",
    dropout=0,
    pretrained=None,
    model_name="mobilevit",
    kwargs=None,
):
    inputs = keras.layers.Input(input_shape)
    nn = conv2d_no_bias(inputs, stem_width, kernel_size=3, strides=2, padding="same", name="stem_")
    nn = batchnorm_with_activation(nn, activation=activation, name="stem_")

    """ stages """
    drop_connect_rates = drop_connect_rates_split(num_blocks, start=0.0, end=drop_connect_rate)
    for id, (num_block, out_channel, block_type, stride, drop_connect) in enumerate(zip(num_blocks, out_channels, block_types, strides, drop_connect_rates)):
        stack_name = "stack{}_".format(id + 1)
        is_conv_block = True if block_type[0].lower() == "c" else False
        attn_channel = attn_channels[id] if isinstance(attn_channels, (list, tuple)) else (attn_channels * out_channel)
        nn = stack(nn, num_block, out_channel, is_conv_block, stride, expand_ratio, attn_channel, drop_connect, layer_scale, activation, name=stack_name)

    nn = output_block(nn, output_num_features, activation, num_classes, drop_rate=dropout, classifier_activation=classifier_activation)
    model = keras.models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="raw01")
    reload_model_weights(model, PRETRAINED_DICT, "mobilevit", pretrained)
    return model


def MobileViT_XXS(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [1, 3, 3, 5, 4]
    out_channels = [16, 24, 48, 64, 80]
    attn_channels = [0, 0, 64, 80, 96]
    output_num_features = 320
    expand_ratio = 2
    return MobileViT(**locals(), model_name="mobilevit_xxs", **kwargs)


def MobileViT_XS(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [1, 3, 3, 5, 4]
    out_channels = [32, 48, 64, 80, 96]
    attn_channels = 1.5
    output_num_features = 384
    return MobileViT(**locals(), model_name="mobilevit_xs", **kwargs)


def MobileViT_S(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [1, 3, 3, 5, 4]
    out_channels = [32, 64, 96, 128, 160]
    attn_channels = 1.5
    return MobileViT(**locals(), model_name="mobilevit_s", **kwargs)
