"""
A Keras version of `botnet`.
Original TensorFlow version: https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from keras_cv_attention_models.aotnet import AotNet
from keras_cv_attention_models.attention_layers import conv2d_no_bias
from keras_cv_attention_models.download_and_load import reload_model_weights

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5

PRETRAINED_DICT = {
    # "botnet50": {"imagenet": "b221b45ca316166fc858fda1cf4fd946"},
    "botnet26t": {"imagenet": {256: "6d7a9548f866b4971ca2c9d17dd815fc"}},
    "botnext_eca26t": {"imagenet": {256: "170b9b4d7fba88dbcb41716047c047b9"}},
    "botnet_se33t": {"imagenet": {256: "f612743ec59d430f197bc38b3a7f8837"}},
}


@tf.keras.utils.register_keras_serializable(package="botnet")
class RelativePositionalEmbedding(keras.layers.Layer):
    def __init__(self, position_height=0, position_width=0, use_absolute_pos=False, dynamic_shape=False, **kwargs):
        super(RelativePositionalEmbedding, self).__init__(**kwargs)
        self.position_height = position_height
        self.position_width = position_width if position_width > 0 else position_height
        self.use_absolute_pos = use_absolute_pos
        self.dynamic_shape = dynamic_shape

    def build(self, input_shape):
        _, num_heads, height, width, key_dim = input_shape
        self.position_height = self.position_height if self.position_height > height else height
        self.position_width = self.position_width if self.position_width > width else width
        self.key_dim = key_dim
        stddev = key_dim ** -0.5

        if self.use_absolute_pos:
            hh_shape = (key_dim, self.position_height)
            ww_shape = (key_dim, self.position_width)
        else:
            hh_shape = (key_dim, 2 * self.position_height - 1)
            ww_shape = (key_dim, 2 * self.position_width - 1)

        initializer = tf.random_normal_initializer(stddev=stddev)
        self.pos_emb_h = self.add_weight(name="r_height", shape=hh_shape, initializer=initializer, trainable=True)
        self.pos_emb_w = self.add_weight(name="r_width", shape=ww_shape, initializer=initializer, trainable=True)
        self.input_height, self.input_width = height, width

    def get_config(self):
        base_config = super(RelativePositionalEmbedding, self).get_config()
        base_config.update(
            {
                "position_height": self.position_height,
                "position_width": self.position_width,
                "use_absolute_pos": self.use_absolute_pos,
                "dynamic_shape": self.dynamic_shape,
            }
        )
        return base_config

    def rel_to_abs(self, rel_pos):
        """
        Converts relative indexing to absolute.
        Input: [bs+heads, height, width, 2 * pos_dim - 1]
        Output: [bs+heads, height, width, pos_dim]
        """
        bs_heads, hh, ww, dim = rel_pos.shape  # [bs+heads, height, width, 2 * width - 1]
        pos_dim = (dim + 1) // 2
        if pos_dim == 1:
            return rel_pos
        if ww == 1:
            return rel_pos[:, :, :, -pos_dim:]
        full_rank_gap = pos_dim - ww
        # [bs+heads, height, width * (2 * pos_dim - 1)] --> [bs+heads, height, width * (2 * pos_dim - 1) - width]
        flat_x = tf.reshape(rel_pos, [-1, hh, ww * dim])[:, :, ww - 1 : -1]
        # [bs+heads, height, width, 2 * (pos_dim - 1)] --> [bs+heads, height, width, pos_dim]
        # print(f">>>> {full_rank_gap = }, {flat_x.shape = }")
        return tf.reshape(flat_x, [-1, hh, ww, 2 * (pos_dim - 1)])[:, :, :, full_rank_gap : pos_dim + full_rank_gap]

    def relative_logits(self, inputs):
        bs, heads, hh, ww, cc = inputs.shape  # e.g.: [1, 4, 14, 16, 128]
        inputs = tf.reshape(inputs, [-1, hh, ww, cc])  # Merge bs and heads, for supporting TFLite conversion
        rel_logits_w = tf.matmul(inputs, self.pos_emb_w)  # [4, 14, 16, 31], 2 * 16 - 1 == 31
        rel_logits_w = self.rel_to_abs(rel_logits_w)  # [4, 14, 16, 16]

        query_h = tf.transpose(inputs, [0, 2, 1, 3])  # [4, 16, 14, 128], [bs+heads, ww, hh, dims], Exchange `ww` and `hh`
        rel_logits_h = tf.matmul(query_h, self.pos_emb_h)  # [4, 16, 14, 27], 2 * 14 - 1 == 27
        rel_logits_h = self.rel_to_abs(rel_logits_h)  # [4, 16, 14, 14]
        rel_logits_h = tf.transpose(rel_logits_h, [0, 2, 1, 3])  # [4, 14, 16, 14], transpose back

        logits = tf.expand_dims(rel_logits_w, axis=-2) + tf.expand_dims(rel_logits_h, axis=-1)  # [4, 14, 16, 14, 16]
        return tf.reshape(logits, [-1, heads, hh, ww, self.position_height, self.position_width])  # [1, 4, 14, 16, 14, 16]

    def absolute_logits(self, inputs):
        # pos_emb = tf.expand_dims(self.pos_emb_w, -2) + tf.expand_dims(self.pos_emb_h, -1)
        # return tf.einsum("bxyhd,dpq->bhxypq", inputs, pos_emb)
        rel_logits_w = tf.matmul(inputs, self.pos_emb_w)
        rel_logits_h = tf.matmul(inputs, self.pos_emb_h)
        return tf.expand_dims(rel_logits_w, axis=-2) + tf.expand_dims(rel_logits_h, axis=-1)

    def call(self, inputs):
        pos_emb = self.absolute_logits(inputs) if self.use_absolute_pos else self.relative_logits(inputs)
        if self.dynamic_shape:
            _, _, hh, ww, _ = inputs.shape
            if hh < self.position_height or ww < self.position_width:
                pos_emb = pos_emb[:, :, :, :, :hh, :ww]
        return pos_emb

    def load_resized_pos_emb(self, source_layer, method="nearest"):
        # For input 224 --> [128, 27], convert to 480 --> [128, 30]
        if isinstance(source_layer, dict):
            source_pos_emb_h = source_layer["r_height:0"]  # weights
            source_pos_emb_w = source_layer["r_width:0"]  # weights
        else:
            source_pos_emb_h = source_layer.pos_emb_h  # layer
            source_pos_emb_w = source_layer.pos_emb_w  # layer
        image_like_w = tf.expand_dims(tf.transpose(source_pos_emb_w, [1, 0]), 0)
        resize_w = tf.image.resize(image_like_w, (1, self.pos_emb_w.shape[1]), method=method)[0]
        self.pos_emb_w.assign(tf.transpose(resize_w, [1, 0]))

        image_like_h = tf.expand_dims(tf.transpose(source_pos_emb_h, [1, 0]), 0)
        resize_h = tf.image.resize(image_like_h, (1, self.pos_emb_h.shape[1]), method=method)[0]
        self.pos_emb_h.assign(tf.transpose(resize_h, [1, 0]))

    def show_pos_emb(self, base_size=4):
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 3, figsize=(base_size * 3, base_size * 1))
        axes[0].imshow(self.pos_emb_h)
        axes[1].imshow(self.pos_emb_w)
        hh_sum = tf.ones([1, self.pos_emb_h.shape[0]]) @ self.pos_emb_h
        ww_sum = tf.ones([1, self.pos_emb_w.shape[0]]) @ self.pos_emb_w
        axes[2].imshow(tf.transpose(hh_sum) + ww_sum)
        titles = ["pos_emb_h", "pos_emb_w", "sum"]
        for ax, title in zip(axes.flatten(), titles):
            ax.set_title(title)
            ax.set_axis_off()
        fig.tight_layout()
        return fig


def mhsa_with_relative_position_embedding(
    inputs, num_heads=4, key_dim=0, relative=True, out_shape=None, out_weight=True, out_bias=False, attn_dropout=0, name=None
):
    _, hh, ww, cc = inputs.shape
    key_dim = key_dim if key_dim > 0 else cc // num_heads
    qk_scale = 1.0 / tf.math.sqrt(tf.cast(key_dim, inputs.dtype))
    out_shape = cc if out_shape is None or not out_weight else out_shape
    qk_out = num_heads * key_dim
    vv_dim = out_shape // num_heads

    # qkv = keras.layers.Dense(emb_dim * 3, use_bias=False, name=name and name + "qkv")(inputs)
    qkv = conv2d_no_bias(inputs, qk_out * 2 + out_shape, kernel_size=1, name=name and name + "qkv_")
    qkv = tf.reshape(qkv, [-1, inputs.shape[1] * inputs.shape[2], qkv.shape[-1]])
    query, key, value = tf.split(qkv, [qk_out, qk_out, out_shape], axis=-1)
    # query = [batch, num_heads, hh * ww, key_dim]
    query = tf.transpose(tf.reshape(query, [-1, query.shape[1], num_heads, key_dim]), [0, 2, 1, 3])
    # key = [batch, num_heads, key_dim, hh * ww]
    key = tf.transpose(tf.reshape(key, [-1, key.shape[1], num_heads, key_dim]), [0, 2, 3, 1])
    # value = [batch, num_heads, hh * ww, vv_dim]
    value = tf.transpose(tf.reshape(value, [-1, value.shape[1], num_heads, vv_dim]), [0, 2, 1, 3])

    # query *= qk_scale
    # [batch, num_heads, hh * ww, hh * ww]
    attention_scores = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([query, key]) * qk_scale
    # pos_query = [batch, num_heads, hh, ww, key_dim]
    pos_query = tf.reshape(query, [-1, num_heads, inputs.shape[1], inputs.shape[2], key_dim])
    pos_emb = RelativePositionalEmbedding(use_absolute_pos=not relative, name=name and name + "pos_emb")(pos_query)
    pos_emb = tf.reshape(pos_emb, [-1, *attention_scores.shape[1:]])
    attention_scores = keras.layers.Add()([attention_scores, pos_emb])
    # attention_scores = tf.nn.softmax(attention_scores, axis=-1)
    attention_scores = keras.layers.Softmax(axis=-1, name=name and name + "attention_scores")(attention_scores)

    if attn_dropout > 0:
        attention_scores = keras.layers.Dropout(attn_dropout, name=name and name + "attn_drop")(attention_scores)
    # value = [batch, num_heads, hh * ww, vv_dim]
    # attention_output = tf.matmul(attention_scores, value)  # [batch, num_heads, hh * ww, vv_dim]
    attention_output = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([attention_scores, value])
    attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
    attention_output = tf.reshape(attention_output, [-1, inputs.shape[1], inputs.shape[2], num_heads * vv_dim])
    # print(f">>>> {attention_output.shape = }, {attention_scores.shape = }")

    if out_weight:
        # [batch, hh, ww, num_heads * vv_dim] * [num_heads * vv_dim, out] --> [batch, hh, ww, out]
        attention_output = keras.layers.Dense(out_shape, use_bias=out_bias, name=name and name + "output")(attention_output)
    return attention_output


def BotNet(input_shape=(224, 224, 3), strides=1, pretrained="imagenet", **kwargs):
    attn_types = [None, None, None, "bot"]
    attn_params = {"num_heads": 4, "out_weight": False}
    strides = strides if isinstance(strides, (list, tuple)) else [1, 2, 2, strides]

    model = AotNet(input_shape=input_shape, attn_types=attn_types, attn_params=attn_params, strides=strides, **kwargs)
    reload_model_weights(model, PRETRAINED_DICT, "botnet", pretrained, RelativePositionalEmbedding)
    return model


def BotNet50(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", strides=1, **kwargs):
    num_blocks = [3, 4, 6, 3]
    return BotNet(**locals(), model_name="botnet50", **kwargs)


def BotNet101(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained=None, strides=1, **kwargs):
    num_blocks = [3, 4, 23, 3]
    return BotNet(**locals(), model_name="botnet101", **kwargs)


def BotNet152(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained=None, strides=1, **kwargs):
    num_blocks = [3, 8, 36, 3]
    return BotNet(**locals(), model_name="botnet152", **kwargs)


def BotNet26T(input_shape=(256, 256, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 2, 2, 2]
    attn_types = [None, None, [None, "bot"], "bot"]
    attn_params = {"num_heads": 4, "out_weight": False}
    stem_type = "tiered"

    model = AotNet(model_name="botnet26t", **locals(), **kwargs)
    reload_model_weights(model, PRETRAINED_DICT, "botnet", pretrained, RelativePositionalEmbedding)
    return model


def BotNextECA26T(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 2, 2, 2]
    attn_types = [None, None, [None, "bot"], "bot"]
    attn_params = {"num_heads": 4, "key_dim": 16, "out_weight": False}
    use_eca = True
    group_size = 16
    stem_type = "tiered"
    model = AotNet(model_name="botnext_eca26t", **locals(), **kwargs)
    reload_model_weights(model, PRETRAINED_DICT, "botnet", pretrained, RelativePositionalEmbedding)
    return model


def BotNetSE33T(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    stem_type = "tiered"
    stem_last_strides = 2
    stem_downsample = False
    out_channels = [256, 512, 1024, 1536]
    hidden_channel_ratio = [1 / 4, 1 / 4, 1 / 4, 1 / 3]
    num_blocks = [2, 3, 3, 2]
    attn_types = [None, [None, None, "bot"], [None, None, "bot"], "bot"]
    attn_params = {"num_heads": 4, "out_weight": False}
    se_ratio = 1 / 16
    output_num_features = 1280

    model = AotNet(model_name="botnet_se33t", **locals(), **kwargs)
    reload_model_weights(model, PRETRAINED_DICT, "botnet", pretrained, RelativePositionalEmbedding)
    return model
