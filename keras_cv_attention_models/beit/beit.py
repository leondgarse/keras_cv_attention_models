import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from keras_cv_attention_models.attention_layers import (
    activation_by_name,
    BiasLayer,
    ChannelAffine,
    ClassToken,
    conv2d_no_bias,
    drop_block,
    drop_connect_rates_split,
    layer_norm,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

LAYER_NORM_EPSILON = 1e-6

PRETRAINED_DICT = {
    "beit_base_patch16": {"imagenet21k-ft1k": {224: "d7102337a13a3983f3b6470de77b5d5c", 384: "76353026477c60f8fdcbcc749fea17b3"}},
    "beit_large_patch16": {
        "imagenet21k-ft1k": {224: "fce2d162e7fa4dba9a1b1fc5e1dec5ce", 384: "158934d07dd8b1e1c6b96883aa00a748", 512: "64d18088e91df243960e5830aab80a6e"}
    },
}


@tf.keras.utils.register_keras_serializable(package="beit")
class MultiHeadRelativePositionalEmbedding(keras.layers.Layer):
    def __init__(self, with_cls_token=True, attn_height=-1, num_heads=-1, **kwargs):
        super(MultiHeadRelativePositionalEmbedding, self).__init__(**kwargs)
        self.with_cls_token, self.attn_height, self.num_heads = with_cls_token, attn_height, num_heads
        if with_cls_token:
            self.cls_token_len = 1
            self.cls_token_pos_len = 3
        else:
            self.cls_token_len = 0
            self.cls_token_pos_len = 0

    def build(self, attn_shape):
        # print(attn_shape)
        if self.attn_height == -1:
            height = width = int(tf.math.sqrt(float(attn_shape[2] - self.cls_token_len)))  # assume hh == ww, e.g. 14
        else:
            height = self.attn_height
            width = int(float(attn_shape[2] - self.cls_token_len) / height)
        num_heads = attn_shape[1] if self.num_heads == -1 else self.num_heads
        num_relative_distance = (2 * height - 1) * (2 * width - 1) + self.cls_token_pos_len
        # pos_shape = (num_relative_distance, num_heads)
        pos_shape = (num_heads, num_relative_distance)
        self.relative_position_bias_table = self.add_weight(name="positional_embedding", shape=pos_shape, initializer="zeros", trainable=True)

        xx, yy = tf.meshgrid(range(height), range(width))  # tf.meshgrid is same with np.meshgrid 'xy' mode, while torch.meshgrid 'ij' mode
        coords = tf.stack([yy, xx], axis=-1)  # [14, 14, 2]
        coords_flatten = tf.reshape(coords, [-1, 2])  # [196, 2]
        relative_coords = coords_flatten[:, None, :] - coords_flatten[None, :, :]  # [196, 196, 2]
        xx = (relative_coords[:, :, 0] + width - 1) * (2 * height - 1)
        yy = relative_coords[:, :, 1] + height - 1
        relative_coords = tf.stack([xx, yy], axis=-1)

        relative_position_index = tf.reduce_sum(relative_coords, axis=-1)  # [196, 196]
        if attn_shape[3] != attn_shape[2]:
            # Choose the small values if value_block != query_block
            relative_position_index = relative_position_index[:, -(attn_shape[3] - self.cls_token_len) :]

        if self.with_cls_token:
            top = tf.ones((1, relative_position_index.shape[1]), dtype=relative_position_index.dtype) * (num_relative_distance - 3)
            left = tf.ones((relative_position_index.shape[0], 1), dtype=relative_position_index.dtype) * (num_relative_distance - 2)
            corner = tf.ones((1, 1), dtype=relative_position_index.dtype) * (num_relative_distance - 1)
            # print(f">>>> {top.shape = }, {left.shape = }, {corner.shape = }")
            # >>>> top.shape = TensorShape([1, 196]), left.shape = TensorShape([196, 1]), corner.shape = TensorShape([1, 1])
            left_corner = tf.concat([corner, left], axis=0)
            relative_position_index = tf.concat([top, relative_position_index], axis=0)
            relative_position_index = tf.concat([left_corner, relative_position_index], axis=1)  # [197, 197]
        self.relative_position_index = relative_position_index

    def call(self, attention_scores, **kwargs):
        pos_emb = tf.gather(self.relative_position_bias_table, self.relative_position_index, axis=1)
        # tf.print(pos_emb.shape, attention_scores.shape)
        return attention_scores + pos_emb

    def get_config(self):
        base_config = super(MultiHeadRelativePositionalEmbedding, self).get_config()
        base_config.update({"with_cls_token": self.with_cls_token, "attn_height": self.attn_height, "num_heads": self.num_heads})
        return base_config

    def load_resized_pos_emb(self, source_layer, method="nearest"):
        if isinstance(source_layer, dict):
            source_tt = source_layer["positional_embedding:0"]  # weights
            # source_tt = source_layer["pos_emb:0"]  # weights
        else:
            source_tt = source_layer.relative_position_bias_table  # layer
        # self.relative_position_bias_table.assign(tf.transpose(source_tt))
        hh = ww = int(tf.math.sqrt(float(source_tt.shape[1] - self.cls_token_pos_len)))  # assume source weights are all square shape
        num_heads = source_tt.shape[0]
        ss = tf.reshape(source_tt[:, : hh * ww], (num_heads, hh, ww))  # [num_heads, hh, ww]
        ss = tf.transpose(ss, [1, 2, 0])  # [hh, ww, num_heads]

        if self.attn_height == -1:
            target_hh = target_ww = int(tf.math.sqrt(float(self.relative_position_bias_table.shape[1] - self.cls_token_pos_len)))
        else:
            target_hh = 2 * self.attn_height - 1
            target_ww = int(float(self.relative_position_bias_table.shape[1] - self.cls_token_pos_len) / target_hh)
        tt = tf.image.resize(ss, [target_hh, target_ww], method=method)  # [target_hh, target_ww, num_heads]
        tt = tf.reshape(tt, (tt.shape[0] * tt.shape[1], num_heads))  # [target_hh * target_ww, num_heads]
        tt = tf.transpose(tt)  # [num_heads, target_hh * target_ww]
        if self.with_cls_token:
            tt = tf.concat([tt, source_tt[:, -self.cls_token_pos_len :]], axis=1)
        self.relative_position_bias_table.assign(tt)

    def show_pos_emb(self, rows=1, base_size=2):
        import matplotlib.pyplot as plt

        hh = ww = int(tf.math.sqrt(float(self.relative_position_bias_table.shape[0] - self.cls_token_pos_len)))
        ss = tf.reshape(self.relative_position_bias_table[: hh * ww], (hh, ww, -1)).numpy()
        cols = int(tf.math.ceil(ss.shape[-1] / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(base_size * cols, base_size * rows))
        for id, ax in enumerate(axes.flatten()):
            ax.imshow(ss[:, :, id])
            ax.set_axis_off()
        fig.tight_layout()
        return fig


def attention_block(inputs, num_heads=4, key_dim=0, out_weight=True, out_bias=False, qv_bias=True, attn_dropout=0, name=None):
    _, bb, cc = inputs.shape
    key_dim = key_dim if key_dim > 0 else cc // num_heads
    qk_scale = 1.0 / tf.math.sqrt(tf.cast(key_dim, inputs.dtype))
    emded_dim = num_heads * key_dim

    qkv = keras.layers.Dense(emded_dim * 3, use_bias=False, name=name and name + "qkv")(inputs)
    qkv = tf.reshape(qkv, [-1, bb, qkv.shape[-1]])
    query, key, value = tf.split(qkv, 3, axis=-1)
    # query = [batch, num_heads, cls_token + hh * ww, key_dim]
    if qv_bias:
        query = BiasLayer(name=name + "query_bias")(query)
    query = tf.reshape(query, [-1, query.shape[1], num_heads, key_dim])
    query = tf.transpose(query, [0, 2, 1, 3])
    # key = [batch, num_heads, key_dim, cls_token + hh * ww]
    key = tf.transpose(tf.reshape(key, [-1, key.shape[1], num_heads, key_dim]), [0, 2, 3, 1])
    # value = [batch, num_heads, cls_token + hh * ww, key_dim]
    if qv_bias:
        value = BiasLayer(name=name + "value_bias")(value)
    value = tf.reshape(value, [-1, value.shape[1], num_heads, key_dim])
    value = tf.transpose(value, [0, 2, 1, 3])

    query *= qk_scale
    # [batch, num_heads, cls_token + hh * ww, cls_token + hh * ww]
    attention_scores = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([query, key])
    attention_scores = MultiHeadRelativePositionalEmbedding(name=name and name + "pos_emb")(attention_scores)
    # attention_scores = tf.nn.softmax(attention_scores, axis=-1, name=name and name + "_attention_scores")
    attention_scores = keras.layers.Softmax(axis=-1, name=name and name + "attention_scores")(attention_scores)

    if attn_dropout > 0:
        attention_scores = keras.layers.Dropout(attn_dropout, name=name and name + "attn_drop")(attention_scores)
    # value = [batch, num_heads, cls_token + hh * ww, key_dim]
    # attention_output = tf.matmul(attention_scores, value)  # [batch, num_heads, cls_token + hh * ww, key_dim]
    attention_output = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([attention_scores, value])
    attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
    attention_output = tf.reshape(attention_output, [-1, bb, emded_dim])
    # print(f">>>> {attention_output.shape = }, {attention_scores.shape = }")

    if out_weight:
        # [batch, cls_token + hh * ww, num_heads * key_dim] * [num_heads * key_dim, out] --> [batch, cls_token + hh * ww, out]
        attention_output = keras.layers.Dense(emded_dim, use_bias=out_bias, name=name and name + "output")(attention_output)
    return attention_output


def attention_mlp_block(inputs, embed_dim, gamma_init_value=0.1, mlp_ratio=4, drop_rate=0, activation="gelu", attn_params={}, name=""):
    # print(f">>>> {drop_rate = }")
    nn = layer_norm(inputs, epsilon=LAYER_NORM_EPSILON, name=name + "attn_")
    nn = attention_block(nn, **attn_params, name=name + "attn_")
    nn = ChannelAffine(use_bias=False, weight_init_value=gamma_init_value, name=name + "attn_gamma")(nn)
    nn = drop_block(nn, drop_rate)
    attn_out = keras.layers.Add(name=name + "attn_out")([inputs, nn])

    """ MLP """
    nn = layer_norm(attn_out, epsilon=LAYER_NORM_EPSILON, name=name + "mlp_")
    nn = keras.layers.Dense(embed_dim * mlp_ratio, name=name + "mlp_dense_1")(nn)
    nn = activation_by_name(nn, activation, name=name + "mlp_" + activation)
    nn = keras.layers.Dense(embed_dim, name=name + "mlp_dense_2")(nn)
    nn = ChannelAffine(use_bias=False, weight_init_value=gamma_init_value, name=name + "mlp_gamma")(nn)
    nn = drop_block(nn, drop_rate)
    nn = keras.layers.Add(name=name + "mlp_output")([attn_out, nn])
    return nn


@tf.keras.utils.register_keras_serializable(package="beit")
class HeadInitializer(tf.initializers.Initializer):
    def __init__(self, stddev=0.02, scale=0.001, **kwargs):
        super().__init__(**kwargs)
        self.stddev, self.scale = stddev, scale

    def __call__(self, shape, dtype="float32"):
        return tf.initializers.TruncatedNormal(stddev=self.stddev)(shape, dtype=dtype) * self.scale

    def get_config(self):
        base_config = super().get_config()
        base_config.update({"stddev": self.stddev, "scale": self.scale})
        return base_config


def Beit(
    depth=12,
    embed_dim=768,
    num_heads=12,
    mlp_ratio=4,
    patch_size=16,
    attn_key_dim=0,
    attn_qv_bias=True,
    attn_out_weight=True,
    attn_out_bias=True,
    attn_dropout=0,
    gamma_init_value=0.1,
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="gelu",
    drop_connect_rate=0,
    use_mean_pooling=True,
    classifier_activation="softmax",
    pretrained="imagenet21k-ft1k",
    model_name="beit",
    kwargs=None,
):
    inputs = keras.layers.Input(input_shape)

    """ forward_embeddings """
    nn = conv2d_no_bias(inputs, embed_dim, patch_size, strides=patch_size, padding="valid", use_bias=True, name="stem_")
    nn = keras.layers.Reshape([-1, nn.shape[-1]])(nn)
    nn = ClassToken(name="cls_token")(nn)

    attn_params = {
        "num_heads": num_heads,
        "key_dim": attn_key_dim,
        "qv_bias": attn_qv_bias,
        "out_weight": attn_out_weight,
        "out_bias": attn_out_bias,
        "attn_dropout": attn_dropout,
    }

    """ forward_tokens """
    drop_connect_rates = drop_connect_rates_split([depth], 0.0, drop_connect_rate)[0]
    for id in range(depth):
        name = "block{}_".format(id)
        block_drop_rate = drop_connect_rates[id]
        nn = attention_mlp_block(nn, embed_dim, gamma_init_value, mlp_ratio, block_drop_rate, activation, attn_params, name=name)

    if use_mean_pooling:
        nn = tf.reduce_mean(nn[:, 1:, :], axis=1)
        nn = layer_norm(nn, epsilon=LAYER_NORM_EPSILON, name="out_")
    else:
        nn = layer_norm(nn, epsilon=LAYER_NORM_EPSILON, name="out_")[:, 0]

    if num_classes > 0:
        head_init = HeadInitializer()
        nn = keras.layers.Dense(
            num_classes, dtype="float32", activation=classifier_activation, kernel_initializer=head_init, bias_initializer=head_init, name="predictions"
        )(nn)
    model = tf.keras.models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="tf")
    reload_model_weights(model, PRETRAINED_DICT, "beit", pretrained, MultiHeadRelativePositionalEmbedding)
    return model


def BeitBasePatch16(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet21k-ft1k", **kwargs):
    embed_dim = 768
    depth = 12
    num_heads = 12
    gamma_init_value = 0.1
    return Beit(**locals(), model_name="beit_base_patch16", **kwargs)


def BeitLargePatch16(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet21k-ft1k", **kwargs):
    embed_dim = 1024
    depth = 24
    num_heads = 16
    gamma_init_value = 1e-5
    return Beit(**locals(), model_name="beit_large_patch16", **kwargs)
