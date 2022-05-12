import tensorflow as tf
from tensorflow import keras
from keras_cv_attention_models.attention_layers import (
    ChannelAffine,
    conv2d_no_bias,
    depthwise_conv2d_no_bias,
    drop_block,
    layer_norm,
    mlp_block,
    multi_head_self_attention,
    output_block,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

PRETRAINED_DICT = {
    "davit_t": {"imagenet": "040a215cbcf1c0ce06db665cdcf6f9ac"},
    "davit_s": {"imagenet": "b95558071c639815f4ab2e9d09a4141f"},
    "davit_b": {"imagenet": "89e50de7a70ea7b2404f8f57369d8015"},
}


def multi_head_self_attention_channel(
    inputs, num_heads=4, key_dim=0, out_shape=None, out_weight=True, qkv_bias=False, out_bias=False, attn_dropout=0, output_dropout=0, name=None
):
    _, hh, ww, cc = inputs.shape
    key_dim = key_dim if key_dim > 0 else cc // num_heads
    qk_scale = float(1.0 / tf.math.sqrt(tf.cast(key_dim, "float32")))
    out_shape = cc if out_shape is None or not out_weight else out_shape
    qkv_out = num_heads * key_dim

    qkv = keras.layers.Dense(qkv_out * 3, use_bias=qkv_bias, name=name and name + "qkv")(inputs)
    qkv = tf.reshape(qkv, [-1, qkv.shape[1] * qkv.shape[2], qkv.shape[-1]])
    value, query, key = tf.split(qkv, 3, axis=-1)  # Matching weights from PyTorch
    query = tf.transpose(tf.reshape(query, [-1, query.shape[1], num_heads, key_dim]), [0, 2, 3, 1])  # [batch, num_heads, key_dim, hh * ww]
    key = tf.transpose(tf.reshape(key, [-1, key.shape[1], num_heads, key_dim]), [0, 2, 1, 3])  # [batch, num_heads, hh * ww, key_dim]
    value = tf.transpose(tf.reshape(value, [-1, value.shape[1], num_heads, key_dim]), [0, 2, 3, 1])  #  [batch, num_heads, key_dim, hh * ww]

    attention_scores = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([query, key]) * qk_scale  # [batch, num_heads, key_dim, key_dim]
    attention_scores = keras.layers.Softmax(axis=-1, name=name and name + "attention_scores")(attention_scores)
    attention_scores = keras.layers.Dropout(attn_dropout, name=name and name + "attn_drop")(attention_scores) if attn_dropout > 0 else attention_scores

    # value = [batch, num_heads, key_dim, hh * ww], attention_output = [batch, num_heads, key_dim, hh * ww]
    attention_output = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([attention_scores, value])
    attention_output = tf.transpose(attention_output, perm=[0, 3, 1, 2])  # [batch, hh * ww, num_heads, key_dim]
    attention_output = tf.reshape(attention_output, [-1, inputs.shape[1], inputs.shape[2], num_heads * key_dim])
    # print(f">>>> {attention_output.shape = }, {attention_scores.shape = }")

    if out_weight:
        # [batch, hh, ww, num_heads * key_dim] * [num_heads * key_dim, out] --> [batch, hh, ww, out]
        attention_output = keras.layers.Dense(out_shape, use_bias=out_bias, name=name and name + "output")(attention_output)
    attention_output = keras.layers.Dropout(output_dropout, name=name and name + "out_drop")(attention_output) if output_dropout > 0 else attention_output
    return attention_output


def window_attention(inputs, window_size, num_heads=4, name=None):
    input_channel = inputs.shape[-1]
    window_size = window_size if isinstance(window_size, (list, tuple)) else [window_size, window_size]
    window_height = window_size[0] if window_size[0] < inputs.shape[1] else inputs.shape[1]
    window_width = window_size[1] if window_size[1] < inputs.shape[2] else inputs.shape[2]

    # window_partition, partition windows, ceil mode
    patch_height, patch_width = int(tf.math.ceil(inputs.shape[1] / window_height)), int(tf.math.ceil(inputs.shape[2] / window_width))
    should_pad_hh, should_pad_ww = patch_height * window_height - inputs.shape[1], patch_width * window_width - inputs.shape[2]
    # print(f">>>> window_attention {inputs.shape = }, {should_pad_hh = }, {should_pad_ww = }")
    if should_pad_hh or should_pad_ww:
        inputs = tf.pad(inputs, [[0, 0], [0, should_pad_hh], [0, should_pad_ww], [0, 0]])

    # print(f">>>> window_attention {inputs.shape = }, {patch_height = }, {patch_width = }, {window_height = }, {window_width = }")
    # [batch * patch_height, window_height, patch_width, window_width * input_channel], limit transpose perm <= 4
    nn = tf.reshape(inputs, [-1, window_height, patch_width, window_width * input_channel])
    nn = tf.transpose(nn, [0, 2, 1, 3])  # [batch * patch_height, patch_width, window_height, window_width * input_channel]
    nn = tf.reshape(nn, [-1, window_height, window_width, input_channel])  # [batch * patch_height * patch_width, window_height, window_width, input_channel]

    nn = multi_head_self_attention(nn, num_heads=num_heads, qkv_bias=True, out_bias=True, name=name)

    # window_reverse, merge windows
    # [batch * patch_height, patch_width, window_height, window_width * input_channel], limit transpose perm <= 4
    nn = tf.reshape(nn, [-1, patch_width, window_height, window_width * input_channel])
    nn = tf.transpose(nn, [0, 2, 1, 3])  # [batch * patch_height, window_height, patch_width, window_width * input_channel]
    nn = tf.reshape(nn, [-1, patch_height * window_height, patch_width * window_width, input_channel])
    if should_pad_hh or should_pad_ww:
        nn = nn[:, : nn.shape[1] - should_pad_hh, : nn.shape[2] - should_pad_ww, :]  # In case should_pad_hh or should_pad_ww is 0

    return nn


def conv_positional_encoding(inputs, kernel_size=3, use_norm=False, activation="gelu", name=""):
    nn = depthwise_conv2d_no_bias(inputs, kernel_size, padding="SAME", use_bias=True, name=name)
    if use_norm:
        nn = layer_norm(nn, name=name)
    if activation is not None:
        nn = activation_by_name(nn, activation, name=name)
    return keras.layers.Add(name=name + "output")([inputs, nn])


def davit_block(
    inputs, window_size, num_heads=4, use_channel_attn=False, mlp_ratio=4, mlp_drop_rate=0, attn_drop_rate=0, drop_rate=0, layer_scale=-1, name=None
):
    input_channel = inputs.shape[-1]

    pre_attn = conv_positional_encoding(inputs, 3, use_norm=False, activation=None, name=name + "pre_attn_cpe_")
    attn = layer_norm(pre_attn, name=name + "attn_")
    if use_channel_attn:
        attn = multi_head_self_attention_channel(attn, num_heads, qkv_bias=True, out_bias=True, name=name + "channel_attn_")
    else:
        attn = window_attention(attn, window_size, num_heads, name=name + "attn_")
    attn = ChannelAffine(use_bias=False, weight_init_value=layer_scale, name=name + "1_gamma")(attn) if layer_scale >= 0 else attn
    attn = drop_block(attn, drop_rate=drop_rate, name=name + "attn_")
    # print(f"{pre_attn.shape = }, {attn.shape = }, {inputs.shape = }")
    attn_out = keras.layers.Add(name=name + "attn_out")([pre_attn, attn])

    pre_ffn = conv_positional_encoding(attn_out, 3, use_norm=False, activation=None, name=name + "pre_ffn_cpe_")
    mlp = layer_norm(pre_ffn, name=name + "mlp_")
    mlp = mlp_block(mlp, int(input_channel * mlp_ratio), drop_rate=mlp_drop_rate, use_conv=False, activation="gelu", name=name + "mlp_")
    mlp = ChannelAffine(use_bias=False, weight_init_value=layer_scale, name=name + "2_gamma")(mlp) if layer_scale >= 0 else mlp
    mlp = drop_block(mlp, drop_rate=drop_rate, name=name + "mlp_")
    return keras.layers.Add(name=name + "output")([pre_ffn, mlp])


def DaViT(
    num_blocks=[2, 2, 6, 2],
    out_channels=[96, 192, 384, 768],
    num_heads=[3, 6, 12, 24],
    stem_width=-1,
    stem_patch_size=4,
    # window_size=7,
    window_ratio=32,
    mlp_ratio=4,
    layer_scale=-1,
    input_shape=(224, 224, 3),
    num_classes=1000,
    drop_connect_rate=0,
    classifier_activation="softmax",
    dropout=0,
    pretrained=None,
    model_name="davit",
    kwargs=None,
):
    """ Patch stem """
    inputs = keras.layers.Input(input_shape)
    stem_width = stem_width if stem_width > 0 else out_channels[0]
    nn = conv2d_no_bias(inputs, stem_width, kernel_size=7, strides=stem_patch_size, use_bias=True, padding="SAME", name="stem_")
    nn = layer_norm(nn, name="stem_")
    # window_size = [input_shape[0] // window_ratio, input_shape[1] // window_ratio]
    window_size = [int(tf.math.ceil(input_shape[0] / window_ratio)), int(tf.math.ceil(input_shape[1] / window_ratio))]
    # window_size = window_size[:2] if isinstance(window_size, (list, tuple)) else [window_size, window_size]

    """ stages """
    total_blocks = sum(num_blocks)
    global_block_id = 0
    for stack_id, (num_block, out_channel, num_head) in enumerate(zip(num_blocks, out_channels, num_heads)):
        stack_name = "stack{}_".format(stack_id + 1)
        if stack_id > 0:
            ds_name = stack_name + "downsample_"
            nn = layer_norm(nn, name=ds_name)
            # Set use_torch_padding=False, as kernel_size == 2, otherwise shape will be enlarged by 1
            nn = conv2d_no_bias(nn, out_channel, kernel_size=2, strides=2, use_bias=True, padding="SAME", use_torch_padding=False, name=ds_name)
        for block_id in range(num_block):
            block_name = stack_name + "block{}_".format(block_id + 1)
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            use_channel_attn = False if block_id % 2 == 0 else True
            nn = davit_block(nn, window_size, num_head, use_channel_attn, mlp_ratio, drop_rate=block_drop_rate, layer_scale=layer_scale, name=block_name)
            global_block_id += 1
    nn = layer_norm(nn, name="pre_output_")

    nn = output_block(nn, num_classes=num_classes, drop_rate=dropout, classifier_activation=classifier_activation)
    model = keras.models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    reload_model_weights(model, PRETRAINED_DICT, "davit", pretrained)
    return model


def DaViT_T(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 2, 6, 2]
    return DaViT(**locals(), model_name="davit_t", **kwargs)


def DaViT_S(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 2, 18, 2]
    return DaViT(**locals(), model_name="davit_s", **kwargs)


def DaViT_B(input_shape=(224, 224, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 2, 18, 2]
    out_channels = [128, 256, 512, 1024]
    num_heads = [4, 8, 16, 32]
    return DaViT(**locals(), model_name="davit_b", **kwargs)


def DaViT_L(input_shape=(384, 384, 3), num_classes=1000, classifier_activation="softmax", pretrained=None, **kwargs):
    num_blocks = [2, 2, 18, 2]
    out_channels = [192, 384, 768, 1536]
    num_heads = [6, 12, 24, 48]
    return DaViT(**locals(), model_name="davit_l", **kwargs)


def DaViT_H(input_shape=(512, 512, 3), num_classes=1000, classifier_activation="softmax", pretrained=None, **kwargs):
    num_blocks = [2, 2, 18, 2]
    out_channels = [256, 512, 1024, 2048]
    num_heads = [8, 16, 32, 64]
    return DaViT(**locals(), model_name="davit_h", **kwargs)


def DaViT_G(input_shape=(512, 512, 3), num_classes=1000, classifier_activation="softmax", pretrained=None, **kwargs):
    num_blocks = [2, 2, 24, 6]
    out_channels = [384, 768, 1536, 3072]
    num_heads = [12, 24, 48, 96]
    return DaViT(**locals(), model_name="davit_g", **kwargs)
