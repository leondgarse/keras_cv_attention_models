import tensorflow as tf
from tensorflow import keras
from keras_cv_attention_models.attention_layers import (
    ChannelAffine,
    activation_by_name,
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
    "edgenext_small": {"imagenet": {256: "0234641a703283de1cb0d935bb0325e4"}},
    "edgenext_small": {"usi": {256: "c237761b5bd5c32041d6b758186a0716"}},
    "edgenext_x_small": {"imagenet": {256: "472df7659422c7feffbec8012a0f6fa4"}},
    "edgenext_xx_small": {"imagenet": {256: "4190ba28c7caa2fe73215448f8abebd6"}},
}
LAYER_NORM_EPSILON = 1e-6

@tf.keras.utils.register_keras_serializable(package="kecam/edgenext")
class PositionalEncodingFourier(keras.layers.Layer):
    def __init__(self, filters=32, temperature=1e4, **kwargs):
        super().__init__(**kwargs)
        self.filters, self.temperature = filters, float(temperature)
        self.epsilon = 1e-6
        self.scale = 2 * tf.acos(-1.0)  # 2 * pi

    def build(self, input_shape):
        _, height, width, channels = input_shape  # ex: height, width, filters = 12, 27, 32
        hh, ww = tf.range(height, dtype="float32"), tf.range(width, dtype="float32")
        hh = (hh + 1) / (tf.cast(height, "float32") + self.epsilon) * self.scale
        ww = (ww + 1) / (tf.cast(width, "float32") + self.epsilon) * self.scale

        dim_t = self.temperature ** (2 * (tf.range(self.filters, dtype="float32") // 2) / self.filters)  # (filters,)
        pos_hh, pos_ww = tf.expand_dims(hh, -1) / dim_t, tf.expand_dims(ww, -1) / dim_t  # pos_hh [12, 32], pos_ww [27, 32]
        pos_hh = tf.stack([tf.sin(pos_hh[:, 0::2]), tf.cos(pos_hh[:, 1::2])], axis=-1)  # pos_hh [12, 16, 2]
        pos_ww = tf.stack([tf.sin(pos_ww[:, 0::2]), tf.cos(pos_ww[:, 1::2])], axis=-1)  # pos_ww [27, 16, 2]
        pos_hh = tf.repeat(tf.reshape(pos_hh, [height, 1, -1]), width, axis=1)  # [12, 27, 32]
        pos_ww = tf.repeat(tf.reshape(pos_ww, [1, width, -1]), height, axis=0)  # [12, 27, 32]
        self.positional_embedding = tf.concat([pos_hh, pos_ww], axis=-1)  # [12, 27, 64]

        self.token_projection_ww = self.add_weight(name="ww", shape=(self.filters * 2, channels), trainable=True, dtype=self.dtype)
        self.token_projection_bb = self.add_weight(name="bb", shape=(channels,), trainable=True, dtype=self.dtype)

    def call(self, inputs, **kwargs):
        pos_emb = self.positional_embedding @ self.token_projection_ww + self.token_projection_bb
        # tf.print(pos_emb.shape, attention_scores.shape)
        return inputs + pos_emb

    def get_config(self):
        base_config = super().get_config()
        base_config.update({"filters": self.filters, "temperature": self.temperature})
        return base_config


def norm_inverted_bottleneck(inputs, mlp_ratio=4, layer_scale=1e-6, drop_rate=0, activation="gelu", name=""):
    input_channel = inputs.shape[-1]
    nn = layer_norm(inputs, epsilon=LAYER_NORM_EPSILON, name=name)
    nn = mlp_block(nn, input_channel * mlp_ratio, activation=activation, name=name)
    nn = ChannelAffine(use_bias=False, weight_init_value=layer_scale, name=name + "gamma")(nn) if layer_scale >= 0 else nn
    nn = drop_block(nn, drop_rate=drop_rate, name=name)
    return nn


def cross_covariance_attention(inputs, num_heads=4, key_dim=0, qkv_bias=True, out_bias=True, attn_dropout=0, out_dropout=0, name=None):
    input_channel = inputs.shape[-1]
    key_dim = key_dim if key_dim > 0 else input_channel // num_heads
    qk_out = key_dim * num_heads

    qkv = keras.layers.Dense(qk_out * 3, use_bias=True, name=name and name + "qkv")(inputs)
    qkv = tf.reshape(qkv, [-1, qkv.shape[1] * qkv.shape[2], qkv.shape[-1]])
    query, key, value = tf.split(qkv, 3, axis=-1)
    query = tf.transpose(tf.reshape(query, [-1, query.shape[1], num_heads, key_dim]), [0, 2, 3, 1])  #  [batch, num_heads, key_dim, hh * ww]
    key = tf.transpose(tf.reshape(key, [-1, key.shape[1], num_heads, key_dim]), [0, 2, 1, 3])  # [batch, num_heads, hh * ww, key_dim]
    value = tf.transpose(tf.reshape(value, [-1, value.shape[1], num_heads, key_dim]), [0, 2, 3, 1])  # [batch, num_heads, key_dim, hh * ww]

    norm_query, norm_key = tf.nn.l2_normalize(query, axis=-1, epsilon=1e-6), tf.nn.l2_normalize(key, axis=-2, epsilon=1e-6)
    attn = tf.matmul(norm_query, norm_key)  # [batch, num_heads, key_dim, key_dim]
    attn = ChannelAffine(axis=1, use_bias=False, name=name and name + "temperature/no_weight_decay")(attn)  # axis=1 means on head dimension
    attention_scores = keras.layers.Softmax(axis=-1, name=name and name + "attention_scores")(attn)

    if attn_dropout > 0:
        attention_scores = keras.layers.Dropout(attn_dropout, name=name and name + "attn_drop")(attention_scores)
    # [batch, num_heads, key_dim, key_dim] * [batch, num_heads, key_dim, hh * ww] -> [batch, num_heads, key_dim, hh * ww]
    attention_output = tf.matmul(attention_scores, value)
    attention_output = tf.transpose(attention_output, perm=[0, 3, 1, 2])  # [batch, hh * ww, num_heads, key_dim]
    attention_output = tf.reshape(attention_output, [-1, inputs.shape[1], inputs.shape[2], num_heads * key_dim])  # [batch, hh, ww, num_heads * key_dim]
    # print(f">>>> {attention_output.shape = }, {attention_scores.shape = }")

    # [batch, hh, ww, num_heads * key_dim] * [num_heads * key_dim, out] --> [batch, hh, ww, out]
    attention_output = keras.layers.Dense(qk_out, use_bias=out_bias, name=name and name + "output")(attention_output)
    attention_output = keras.layers.Dropout(out_dropout, name=name and name + "out_drop")(attention_output) if out_dropout > 0 else attention_output
    return attention_output


def split_depthwise_transpose_attention(
    inputs, split=1, num_heads=4, mlp_ratio=4, use_pos_emb=False, layer_scale=1e-6, drop_rate=0, activation="gelu", name=""
):
    input_channel = inputs.shape[-1]
    sub_channels = int(tf.math.ceil(input_channel / split))

    spx, remainder = inputs[:, :, :, : (split - 1) * sub_channels], inputs[:, :, :, (split - 1) * sub_channels :]
    spx = tf.split(spx, split - 1, axis=-1)
    gathered_result = []
    for id, ii in enumerate(spx):
        sp = ii if id == 0 else (sp + ii)
        sp = depthwise_conv2d_no_bias(sp, kernel_size=3, padding="SAME", use_bias=True, name=name + "spx_{}_".format(id + 1))
        gathered_result.append(sp)
    gathered_result.append(remainder)
    attn = tf.concat(gathered_result, axis=-1)
    # print(f"{inputs.shape = }, {attn.shape = }")

    # XCA
    attn = PositionalEncodingFourier(name=name + "pos")(attn) if use_pos_emb else attn
    nn = layer_norm(attn, epsilon=LAYER_NORM_EPSILON, name=name + "xca_")
    nn = cross_covariance_attention(nn, num_heads, name=name + "xca_")
    nn = ChannelAffine(use_bias=False, weight_init_value=layer_scale, name=name + "xca_gamma")(nn) if layer_scale >= 0 else nn
    nn = drop_block(nn, drop_rate=drop_rate, name=name + "xca_")
    nn = keras.layers.Add(name=name + "xca")([attn, nn])

    # Inverted Bottleneck
    nn = norm_inverted_bottleneck(nn, mlp_ratio, layer_scale, drop_rate, activation=activation, name=name + "ir_")
    return keras.layers.Add(name=name + "output")([inputs, nn])


def conv_encoder(inputs, mlp_ratio=4, kernel_size=7, layer_scale=1e-6, drop_rate=0, activation="gelu", name=""):
    input_channel = inputs.shape[-1]
    nn = depthwise_conv2d_no_bias(inputs, kernel_size, use_bias=True, padding="SAME", name=name)
    nn = norm_inverted_bottleneck(nn, mlp_ratio, layer_scale, drop_rate, activation=activation, name=name)
    # print(f"{nn.shape = }, {inputs.shape = }")
    return keras.layers.Add(name=name + "output")([inputs, nn])


def EdgeNeXt(
    num_blocks=[2, 2, 6, 2],
    out_channels=[24, 48, 88, 168],
    num_heads=4,
    num_stda_layers=[0, 1, 1, 1],
    stda_split=[2, 2, 3, 4],
    stda_use_pos_emb=[False, True, False, False],
    conv_kernel_size=[3, 5, 7, 9],
    stem_width=-1,
    mlp_ratio=4,
    stem_patch_size=4,
    layer_scale=1e-6,
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="gelu",
    drop_connect_rate=0,
    classifier_activation="softmax",
    dropout=0,
    pretrained=None,
    model_name="edgenext",
    kwargs=None,
):
    inputs = keras.layers.Input(input_shape)
    stem_width = stem_width if stem_width > 0 else out_channels[0]
    nn = conv2d_no_bias(inputs, stem_width, kernel_size=stem_patch_size, strides=stem_patch_size, use_bias=True, padding="VALID", name="stem_")
    nn = layer_norm(nn, epsilon=LAYER_NORM_EPSILON, name="stem_")

    """ stages """
    total_blocks = sum(num_blocks)
    global_block_id = 0
    for stack_id, (num_block, out_channel, num_stda_layer) in enumerate(zip(num_blocks, out_channels, num_stda_layers)):
        stack_name = "stack{}_".format(stack_id + 1)
        if stack_id > 0:
            ds_name = stack_name + "downsample_"
            nn = layer_norm(nn, epsilon=LAYER_NORM_EPSILON, name=ds_name)
            # Set use_torch_padding=False, as kernel_size == 2, otherwise shape will be enlarged by 1
            nn = conv2d_no_bias(nn, out_channel, kernel_size=2, strides=2, use_bias=True, padding="VALID", name=ds_name)
        for block_id in range(num_block):
            block_name = stack_name + "block{}_".format(block_id + 1)
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            if block_id > num_block - num_stda_layer - 1:
                split = stda_split[stack_id]
                use_pos_emb = stda_use_pos_emb[stack_id]
                num_head = num_heads[stack_id] if isinstance(num_heads, (list, tuple)) else num_heads
                nn = split_depthwise_transpose_attention(
                    nn, split, num_head, mlp_ratio, use_pos_emb, layer_scale, block_drop_rate, activation, name=block_name + "stda_"
                )
            else:
                kernel_size = conv_kernel_size[stack_id]
                nn = conv_encoder(nn, mlp_ratio, kernel_size, layer_scale, block_drop_rate, activation=activation, name=block_name + "conv_")
            global_block_id += 1

    """ output """
    if num_classes > 0:
        nn = keras.layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        nn = layer_norm(nn, epsilon=LAYER_NORM_EPSILON, name="pre_output_")
        if dropout > 0:
            nn = keras.layers.Dropout(dropout, name="head_drop")(nn)
        nn = keras.layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="predictions")(nn)

    model = keras.models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    reload_model_weights(model, PRETRAINED_DICT, "edgenext", pretrained)
    return model


def EdgeNeXt_XX_Small(input_shape=(256, 256, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return EdgeNeXt(**locals(), model_name="edgenext_xx_small", **kwargs)


def EdgeNeXt_X_Small(input_shape=(256, 256, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 3, 9, 3]
    out_channels = [32, 64, 100, 192]
    return EdgeNeXt(**locals(), model_name="edgenext_x_small", **kwargs)


def EdgeNeXt_Small(input_shape=(256, 256, 3), num_classes=1000, classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 3, 9, 3]
    out_channels = [48, 96, 160, 304]
    num_heads = 8
    return EdgeNeXt(**locals(), model_name="edgenext_small", **kwargs)
