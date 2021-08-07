import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras import backend as K
from einops import rearrange
import os

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


@tf.keras.utils.register_keras_serializable(package="Custom")
class HaloAttention(keras.layers.Layer):
    def __init__(self, num_heads=4, key_dim=128, block_size=2, halo_size=1, out_shape=None, out_weight=True, out_bias=False, attn_dropout=0, **kwargs):
        super(HaloAttention, self).__init__(**kwargs)
        self.num_heads, self.key_dim, self.block_size, self.halo_size, self.out_shape = num_heads, key_dim, block_size, halo_size, out_shape
        self.attn_dropout = attn_dropout
        self.out_bias, self.out_weight = out_bias, out_weight
        self.emb_dim = self.num_heads * self.key_dim
        self.qk_scale = 1.0 / tf.math.sqrt(tf.cast(self.key_dim, self._compute_dtype_object))
        self.kv_kernel = self.block_size + self.halo_size * 2

    def build(self, inputs):
        if hasattr(inputs, "shape"):
            _, hh, ww, cc = inputs.shape
        else:
            _, hh, ww, cc = inputs
        stddev = self.key_dim ** -0.5
        self.out_shape = cc if self.out_shape is None or not self.out_weight else self.out_shape
        self.final_out_shape = (None, hh, ww, self.out_shape)

        self.query_dense = self.add_weight("query", shape=[cc, self.emb_dim], trainable=True)
        self.kv_dense = self.add_weight("key_value", shape=[cc, self.emb_dim * 2], trainable=True)
        if self.out_weight:
            self.out_dense_ww = self.add_weight("output_weight", shape=[self.emb_dim, self.out_shape], trainable=True)
        if self.out_bias:
            self.out_dense_bb = self.add_weight("output_bias", shape=self.out_shape, initializer="zeros", trainable=True)

        if self.attn_dropout > 0:
            self.attn_dropout_layer = keras.layers.Dropout(rate=self.attn_dropout)

        self.pos_emb_w = self.add_weight(
            name="r_width",
            shape=(self.key_dim, 2 * self.kv_kernel - 1),
            initializer=tf.random_normal_initializer(stddev=stddev),
            trainable=True,
        )
        self.pos_emb_h = self.add_weight(
            name="r_height",
            shape=(self.key_dim, 2 * self.kv_kernel - 1),
            initializer=tf.random_normal_initializer(stddev=stddev),
            trainable=True,
        )

    def get_config(self):
        base_config = super(HaloAttention, self).get_config()
        base_config.update(
            {
                "num_heads": self.num_heads,
                "key_dim": self.key_dim,
                "block_size": self.block_size,
                "halo_size": self.halo_size,
                "out_shape": self.out_shape,
                "out_weight": self.out_weight,
                "out_bias": self.out_bias,
                "attn_dropout": self.attn_dropout,
            }
        )
        return base_config

    def rel_to_abs(self, rel_pos):
        """
        Converts relative indexing to absolute.
        Input: [bs, heads, height, width, 2 * pos_dim - 1]
        Output: [bs, heads, height, width, pos_dim]
        """
        _, heads, hh, ww, dim = rel_pos.shape  # [bs, heads, height, width, 2 * width - 1]
        pos_dim = (dim + 1) // 2
        full_rank_gap = pos_dim - ww
        # [bs, heads, height, width * (2 * width - 1)] --> [bs, heads, height, width * (2 * width - 1) - width]
        flat_x = tf.reshape(rel_pos, [-1, heads, hh, ww * (pos_dim * 2 - 1)])[:, :, :, ww - 1 : -1]
        # [bs, heads, height, width, 2 * (width - 1)] --> [bs, heads, height, width, width]
        return tf.reshape(flat_x, [-1, heads, hh, ww, 2 * (pos_dim - 1)])[:, :, :, :, full_rank_gap : pos_dim + full_rank_gap]

    def relative_logits(self, query):
        query_w = query  # e.g.: [1, 4, 14, 16, 128], [bs, heads, hh, ww, dims]
        rel_logits_w = tf.matmul(query_w, self.pos_emb_w)  # [1, 4, 14, 16, 31], 2 * 16 - 1 == 31
        rel_logits_w = self.rel_to_abs(rel_logits_w)  # [1, 4, 14, 16, 16]

        query_h = tf.transpose(query, [0, 1, 3, 2, 4])  # [1, 4, 16, 14, 128], [bs, heads, ww, hh, dims], Exchange `ww` and `hh`
        rel_logits_h = tf.matmul(query_h, self.pos_emb_h)  # [1, 4, 16, 14, 27], 2 * 14 - 1 == 27
        rel_logits_h = self.rel_to_abs(rel_logits_h)  # [1, 4, 16, 14, 14]
        rel_logits_h = tf.transpose(rel_logits_h, [0, 1, 3, 2, 4])  # [1, 4, 14, 16, 14], transpose back

        return tf.expand_dims(rel_logits_w, axis=-2) + tf.expand_dims(rel_logits_h, axis=-1)  # [1, 4, 14, 16, 14, 16]

    def call(self, inputs, return_attention_scores=False, training=None):
        # attn_query = [batch, num_heads, hh, ww, block_size * block_size, key_dim]
        # pos_query = [batch, num_heads * hh * ww, block_size, block_size, key_dim]
        query = tf.matmul(inputs, self.query_dense)
        query = tf.multiply(query, self.qk_scale)
        attn_query = rearrange(query, "B (h hb) (w wb) (hd c) -> B hd h w (hb wb) c", hb=self.block_size, wb=self.block_size, hd=self.num_heads)
        pos_query = rearrange(attn_query, "B hd h w (hb wb) c -> B (hd h w) hb wb c", hb=self.block_size, wb=self.block_size)

        # key_value = [batch, num_heads, hh, ww, kv_kernel * kv_kernel, key_dim * 2]
        key_value = tf.matmul(inputs, self.kv_dense)
        kv_padded = tf.pad(key_value, [[0, 0], [self.halo_size, self.halo_size], [self.halo_size, self.halo_size], [0, 0]])
        sizes, strides = [1, self.kv_kernel, self.kv_kernel, 1], [1, self.block_size, self.block_size, 1]
        kv_inp = tf.image.extract_patches(kv_padded, sizes=sizes, strides=strides, rates=[1, 1, 1, 1], padding="VALID")
        kv_inp = rearrange(kv_inp, "B h w (hb wb hd c) -> B hd h w (hb wb) c", hb=self.kv_kernel, wb=self.kv_kernel, hd=self.num_heads)

        # key = value = [batch, num_heads, hh, ww, kv_kernel * kv_kernel, key_dim]
        key, value = tf.split(kv_inp, 2, axis=-1)

        # scaled_dot_product_attention
        attention_scores = tf.matmul(attn_query, key, transpose_b=True)  # [batch, num_heads, hh, ww, block_size * block_size, kv_kernel * kv_kernel]
        pos = self.relative_logits(pos_query)  # [batch, num_heads * hh * ww, block_size, block_size, kv_kernel, kv_kernel]
        attention_scores += tf.reshape(pos, [-1, *attention_scores.shape[1:]])
        attention_scores = tf.nn.softmax(attention_scores, axis=-1)

        if self.attn_dropout > 0:
            attention_scores = self.attn_dropout_layer(attention_scores, training=training)

        attention_output = tf.matmul(attention_scores, value)  # [batch, num_heads, hh, ww, block_size * block_size, key_dim]
        attention_output = rearrange(attention_output, "B hd h w (hb wb) c -> B (h hb) (w wb) (hd c)", hb=self.block_size, wb=self.block_size)

        if self.out_weight:
            # [batch, hh, ww, num_heads * key_dim] * [num_heads * key_dim, out] --> [batch, hh, ww, out]
            attention_output = tf.matmul(attention_output, self.out_dense_ww)
        if self.out_bias:
            attention_output += self.out_dense_bb
        attention_output.set_shape(self.final_out_shape)

        if return_attention_scores:
            return attention_output, attention_scores
        return attention_output


BLOCK_CONFIGS = {
    "b0": { # rv = 1, rb = 0.5
        "halo_block_size": 8,
        "halo_halo_size": 3,
        "halo_key_dim": 16,  # 16 * rv
        "expansion": 1,  # 2 * rb
        "output_conv_channel": -1,  # df
        "num_blocks": [3, 3, 7, 3],
        "num_heads": [4, 8, 8, 8],
    },
    "b1": { # rv = 1, rb = 1
        "halo_block_size": 8,
        "halo_halo_size": 3,
        "halo_key_dim": 16,  # 16 * rv
        "expansion": 2,  # 2 * rb
        "output_conv_channel": -1,  # df
        "num_blocks": [3, 3, 10, 3],
        "num_heads": [4, 8, 8, 8],
    },
    "b2": { # rv = 1, rb = 1.25
        "halo_block_size": 8,
        "halo_halo_size": 3,
        "halo_key_dim": 16,  # 16 * rv
        "expansion": 2.5,  # 2 * rb
        "output_conv_channel": -1,  # df
        "num_blocks": [3, 3, 11, 3],
        "num_heads": [4, 8, 8, 8],
    },
    "b3": {  # rv = 1, rb = 1.5
        "halo_block_size": 10,
        "halo_halo_size": 3,
        "halo_key_dim": 16,  # 16 * rv
        "expansion": 3,  # 2 * rb
        "output_conv_channel": 1024,  # df
        "num_blocks": [3, 3, 12, 3],
        "num_heads": [4, 8, 8, 8],
    },
    "b4": {  # rv = 1, rb = 3
        "halo_block_size": 12,
        "halo_halo_size": 2,
        "halo_key_dim": 16,  # 16 * rv
        "expansion": 6,  # 2 * rb
        "output_conv_channel": 1280,  # df
        "num_blocks": [3, 3, 12, 3],
        "num_heads": [4, 8, 8, 8],
    },
    "b5": {  # rv = 2.5, rb = 2
        "halo_block_size": 14,
        "halo_halo_size": 2,
        "halo_key_dim": 40,  # 16 * rv
        "expansion": 4,  # 2 * rb
        "output_conv_channel": 1536,  # df
        "num_blocks": [3, 3, 23, 3],
        "num_heads": [4, 8, 8, 8],
    },
    "b6": {  # rv = 3, rb = 2.75
        "halo_block_size": 8,
        "halo_halo_size": 4,
        "halo_key_dim": 48,  # 16 * rv
        "expansion": 5.5,  # 2 * rb
        "output_conv_channel": 1536,  # df
        "num_blocks": [3, 3, 24, 3],
        "num_heads": [4, 8, 8, 8],
    },
    "b7": {  # rv = 4, rb = 3.5
        "halo_block_size": 10,
        "halo_halo_size": 3,
        "halo_key_dim": 64,  # 16 * rv
        "expansion": 7,  # 2 * rb
        "output_conv_channel": 2048,  # df
        "num_blocks": [3, 3, 26, 3],
        "num_heads": [4, 8, 8, 8],
    },
}


def batchnorm_with_activation(inputs, activation="relu", zero_gamma=False, name=""):
    """Performs a batch normalization followed by an activation. zero_gamma: https://arxiv.org/abs/1706.02677 """
    bn_axis = 3 if K.image_data_format() == "channels_last" else 1
    gamma_initializer = tf.zeros_initializer() if zero_gamma else tf.ones_initializer()
    nn = layers.BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        gamma_initializer=gamma_initializer,
        name=name + "bn",
    )(inputs)
    if activation:
        nn = layers.Activation(activation=activation, name=name + activation)(nn)
    return nn


def conv2d_no_bias(inputs, filters, kernel_size, strides=1, padding="VALID", name=""):
    if padding == "SAME":
        inputs = layers.ZeroPadding2D(padding=kernel_size // 2, name=name + "pad")(inputs)
    return layers.Conv2D(filters, kernel_size, strides=strides, padding="VALID", use_bias=False, name=name + "conv")(inputs)


def halo_block(inputs, filter, strides=1, shortcut=False, expansion=2, num_heads=4, key_dim=16, block_size=8, halo_size=4, activation="relu", name=""):
    # target_dimension = round(planes * block.expansion * self.rb)
    expanded_filter = round(filter * expansion)
    if shortcut:
        # print(">>>> Downsample")
        shortcut = conv2d_no_bias(inputs, expanded_filter, 1, strides=strides, name=name + "shorcut_")
        shortcut = batchnorm_with_activation(shortcut, activation=None, zero_gamma=False, name=name + "shorcut_")
    else:
        shortcut = inputs

    # width = planes
    nn = conv2d_no_bias(inputs, filter, 1, name=name + "1_")
    nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name=name + "1_")

    # num_heads=4, key_dim=128, block_size=2, halo_size=1, out_shape=None, out_weight=True, out_bias=False, attn_dropout=0
    # key_dim = round( dim_head * rv)
    # print(">>>>", nn.shape, num_heads, key_dim, block_size, halo_size)
    nn = HaloAttention(num_heads, key_dim, block_size, halo_size, out_bias=True, name=name + "halo")(nn)
    # print(">>>>", nn.shape)
    nn = layers.Activation(activation=activation)(nn)
    # round(planes * self.expansion * rb), expansion = 2
    nn = conv2d_no_bias(nn, expanded_filter, 1, name=name + "2_")
    nn = batchnorm_with_activation(nn, activation=None, zero_gamma=True, name=name + "2_")

    # print(">>>>", nn.shape, shortcut.shape)
    nn = layers.Add(name=name + "_add")([shortcut, nn])
    return layers.Activation(activation, name=name + "_out")(nn)


def halo_stack(inputs, blocks, filter, strides=1, expansion=2, num_heads=4, key_dim=32, block_size=8, halo_size=4, activation="relu", name=""):
    shortcut = True if strides != 1 or inputs.shape[-1] != filter * 2 or expansion != 2 else False
    nn = halo_block(inputs, filter, strides, shortcut, expansion, num_heads, key_dim, block_size, halo_size, activation, name=name + "block1_")
    shortcut = False
    for ii in range(2, blocks + 1):
        block_name = name + "block{}_".format(ii)
        nn = halo_block(nn, filter, 1, shortcut, expansion, num_heads, key_dim, block_size, halo_size, activation, name=block_name)
    return nn


def HaloNet(
    model_type="b0", input_shape=(256, 256, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained=None, model_name="halonet", **kwargs
):
    blocks_config = BLOCK_CONFIGS.get(model_type.lower(), BLOCK_CONFIGS["b0"])
    halo_block_size = blocks_config["halo_block_size"]
    halo_halo_size = blocks_config["halo_halo_size"]
    halo_key_dim = blocks_config["halo_key_dim"]
    expansion = blocks_config["expansion"]
    output_conv_channel = blocks_config["output_conv_channel"]
    num_blocks = blocks_config["num_blocks"]
    num_heads = blocks_config["num_heads"]

    inputs = keras.layers.Input(input_shape)
    nn = keras.layers.ZeroPadding2D(padding=3, name="stem_conv_pad")(inputs)
    nn = conv2d_no_bias(nn, 64, 7, strides=2, padding="VALID", name="stem_")
    nn = batchnorm_with_activation(nn, activation=activation, name="stem_")
    nn = keras.layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name="stem_pool_pad")(nn)
    nn = keras.layers.MaxPooling2D(3, strides=2, name="stem_pool")(nn)

    # Input (height, width) should be dividable by `halo_block_size`, assume height == width here
    if nn.shape[1] % halo_block_size != 0:
        gap = halo_block_size - nn.shape[1] % halo_block_size
        pad_head, pad_tail = gap // 2, gap - gap // 2
        # print(">>>> pad_head:", pad_head, "pad_tail:", pad_tail)
        nn = keras.layers.ZeroPadding2D(padding=((pad_head, pad_tail), (pad_head, pad_tail)), name="gap_pad")(nn)

    out_channels = [64, 128, 256, 512]
    for id, (num_block, out_channel, num_head) in enumerate(zip(num_blocks, out_channels, num_heads)):
        name = "stack{}_".format(id + 1)
        nn = halo_stack(nn, num_block, out_channel, 1, expansion, num_head, halo_key_dim, halo_block_size, halo_halo_size, activation=activation, name=name)

    if output_conv_channel != -1:
        nn = conv2d_no_bias(nn, output_conv_channel, 1, name="post_")

    if num_classes > 0:
        nn = keras.layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        nn = keras.layers.Dense(num_classes, activation=classifier_activation, name="predictions")(nn)

    name = model_name + "_" + model_type
    model = keras.models.Model(inputs, nn, name=name)
    return model


def HaloNetB0(input_shape=(256, 256, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return HaloNet(model_type="b0", **locals(), **kwargs)


def HaloNetB1(input_shape=(256, 256, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return HaloNet(model_type="b1", **locals(), **kwargs)


def HaloNetB2(input_shape=(256, 256, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return HaloNet(model_type="b2", **locals(), **kwargs)


def HaloNetB3(input_shape=(320, 320, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return HaloNet(model_type="b3", **locals(), **kwargs)


def HaloNetB4(input_shape=(384, 384, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return HaloNet(model_type="b4", **locals(), **kwargs)


def HaloNetB5(input_shape=(448, 448, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return HaloNet(model_type="b5", **locals(), **kwargs)


def HaloNetB6(input_shape=(512, 512, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return HaloNet(model_type="b6", **locals(), **kwargs)


def HaloNetB7(input_shape=(600, 600, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return HaloNet(model_type="b7", **locals(), **kwargs)
