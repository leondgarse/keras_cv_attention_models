import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

# from einops import rearrange    # Currently einops 0.3.0 is broken for tf.keras 2.6.0...
import os

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


@tf.keras.utils.register_keras_serializable(package="halonet")
class HaloAttention(keras.layers.Layer):
    def __init__(
        self, num_heads=4, key_dim=128, block_size=2, halo_size=1, strides=1, out_shape=None, out_weight=True, out_bias=False, attn_dropout=0, **kwargs
    ):
        super(HaloAttention, self).__init__(**kwargs)
        self.num_heads, self.key_dim, self.block_size, self.halo_size, self.out_shape = num_heads, key_dim, block_size, halo_size, out_shape
        self.strides = strides
        self.attn_dropout = attn_dropout
        self.out_bias, self.out_weight = out_bias, out_weight
        self.emb_dim = self.num_heads * self.key_dim
        self.qk_scale = 1.0 / tf.math.sqrt(tf.cast(self.key_dim, self._compute_dtype_object))
        self.kv_kernel = self.block_size + self.halo_size * 2
        self.query_block = block_size // strides

    def build(self, inputs):
        if hasattr(inputs, "shape"):
            _, hh, ww, cc = inputs.shape
        else:
            _, hh, ww, cc = inputs
        stddev = self.key_dim ** -0.5
        self.out_shape = cc if self.out_shape is None or not self.out_weight else self.out_shape
        self.final_out_shape = (None, hh // self.strides, ww // self.strides, self.out_shape)

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
                "strides": self.strides,
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
        full_rank_gap = pos_dim - ww if ww > 1 else 2
        # [bs, heads, height, width * (2 * width - 1)] --> [bs, heads, height, width * (2 * width - 1) - width]
        flat_x = tf.reshape(rel_pos, [-1, heads, hh, ww * dim])
        flat_x = flat_x[:, :, :, ww - 1 : -1] if ww > 1 else flat_x[:, :, :, 1:]
        # [bs, heads, height, width, 2 * (width - 1)] --> [bs, heads, height, width, width]
        # print(f">>>> {full_rank_gap = }, {flat_x.shape = }")
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
        # attn_query = [batch, num_heads, hh, ww, query_block * query_block, key_dim]
        # pos_query = [batch, num_heads * hh * ww, query_block, query_block, key_dim]
        if self.strides == 1:
            query = tf.matmul(inputs, self.query_dense)
        else:
            query = tf.matmul(inputs[:, :: self.strides, :: self.strides, :], self.query_dense)
        query = tf.multiply(query, self.qk_scale)
        # print(f">>>> {inputs.shape = }, {query.shape = }, {self.final_out_shape = }, {self.strides = }")
        # attn_query = rearrange(query, "B (h hb) (w wb) (hd c) -> B hd h w (hb wb) c", hb=self.query_block, wb=self.query_block, hd=self.num_heads)
        # pos_query = rearrange(attn_query, "B hd h w (hb wb) c -> B (hd h w) hb wb c", hb=self.query_block, wb=self.query_block)
        _, hh, ww, cc = query.shape
        hh_qq, ww_qq, cc_qq = hh // self.query_block, ww // self.query_block, cc // self.num_heads
        query = tf.reshape(query, [-1, hh_qq, self.query_block, ww_qq, self.query_block, self.num_heads, cc_qq])
        query = tf.transpose(query, [0, 5, 1, 3, 2, 4, 6])
        attn_query = tf.reshape(query, [-1, self.num_heads, hh_qq, ww_qq, self.query_block * self.query_block, cc_qq])
        pos_query = tf.reshape(query, [-1, self.num_heads * hh_qq * ww_qq, self.query_block, self.query_block, cc_qq])

        # key_value = [batch, num_heads, hh, ww, kv_kernel * kv_kernel, key_dim * 2]
        key_value = tf.matmul(inputs, self.kv_dense)
        kv_padded = tf.pad(key_value, [[0, 0], [self.halo_size, self.halo_size], [self.halo_size, self.halo_size], [0, 0]])
        sizes, strides = [1, self.kv_kernel, self.kv_kernel, 1], [1, self.block_size, self.block_size, 1]
        kv_inp = tf.image.extract_patches(kv_padded, sizes=sizes, strides=strides, rates=[1, 1, 1, 1], padding="VALID")
        # kv_inp = rearrange(kv_inp, "B h w (hb wb hd c) -> B hd h w (hb wb) c", hb=self.kv_kernel, wb=self.kv_kernel, hd=self.num_heads)
        _, hh_kk, ww_kk, cc = kv_inp.shape
        cc_kk = cc // self.num_heads // self.kv_kernel // self.kv_kernel
        kv_inp = tf.reshape(kv_inp, [-1, hh_kk, ww_kk, self.kv_kernel, self.kv_kernel, self.num_heads, cc_kk])
        kv_inp = tf.transpose(kv_inp, [0, 5, 1, 2, 3, 4, 6])
        kv_inp = tf.reshape(kv_inp, [-1, self.num_heads, hh_kk, ww_kk, self.kv_kernel * self.kv_kernel, cc_kk])

        # key = value = [batch, num_heads, hh, ww, kv_kernel * kv_kernel, key_dim]
        key, value = tf.split(kv_inp, 2, axis=-1)

        # scaled_dot_product_attention
        # print(f">>>> {attn_query.shape = }, {key.shape = }, {value.shape = }, {kv_inp.shape = }, {pos_query.shape = }")
        attention_scores = tf.matmul(attn_query, key, transpose_b=True)  # [batch, num_heads, hh, ww, query_block * query_block, kv_kernel * kv_kernel]
        pos = self.relative_logits(pos_query)  # [batch, num_heads * hh * ww, query_block, query_block, kv_kernel, kv_kernel]
        # print(f">>>> {pos.shape = }, {attention_scores.shape = }")
        attention_scores += tf.reshape(pos, [-1, *attention_scores.shape[1:]])
        attention_scores = tf.nn.softmax(attention_scores, axis=-1)

        if self.attn_dropout > 0:
            attention_scores = self.attn_dropout_layer(attention_scores, training=training)

        attention_output = tf.matmul(attention_scores, value)  # [batch, num_heads, hh, ww, query_block * query_block, key_dim]
        # attention_output = rearrange(attention_output, "B hd h w (hb wb) c -> B (h hb) (w wb) (hd c)", hb=self.query_block, wb=self.query_block)
        _, heads, hh_aa, ww_aa, patch, cc_aa = attention_output.shape
        attention_output = tf.reshape(attention_output, [-1, heads, hh_aa, ww_aa, self.query_block, self.query_block, cc_aa])
        attention_output = tf.transpose(attention_output, [0, 2, 4, 3, 5, 1, 6])
        attention_output = tf.reshape(attention_output, [-1, hh_aa * self.query_block, ww_aa * self.query_block, heads * cc_aa])
        # print(f">>>> {attention_output.shape = }, {attention_scores.shape = }")

        if self.out_weight:
            # [batch, hh, ww, num_heads * key_dim] * [num_heads * key_dim, out] --> [batch, hh, ww, out]
            attention_output = tf.matmul(attention_output, self.out_dense_ww)
        if self.out_bias:
            attention_output += self.out_dense_bb
        attention_output.set_shape(self.final_out_shape)

        if return_attention_scores:
            return attention_output, attention_scores
        return attention_output


def batchnorm_with_activation(inputs, activation="relu", zero_gamma=False, act_first=False, name=""):
    """Performs a batch normalization followed by an activation. zero_gamma: https://arxiv.org/abs/1706.02677 """
    bn_axis = 3 if K.image_data_format() == "channels_last" else 1
    gamma_initializer = tf.zeros_initializer() if zero_gamma else tf.ones_initializer()
    if act_first and activation:
        inputs = keras.layers.Activation(activation=activation, name=name + activation)(inputs)
    nn = keras.layers.BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        gamma_initializer=gamma_initializer,
        name=name + "bn",
    )(inputs)
    if not act_first and activation:
        nn = keras.layers.Activation(activation=activation, name=name + activation)(nn)
    return nn


def conv2d_no_bias(inputs, filters, kernel_size, strides=1, padding="VALID", name=""):
    if padding == "SAME":
        inputs = keras.layers.ZeroPadding2D(padding=kernel_size // 2, name=name + "pad")(inputs)
    return keras.layers.Conv2D(filters, kernel_size, strides=strides, padding="VALID", use_bias=False, name=name + "conv")(inputs)


def halo_block(inputs, filter, strides=1, shortcut=False, expansion=2, num_heads=4, halo_expansion=1, block_size=8, halo_size=4, activation="relu", name=""):
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
    out_shape = int(filter * halo_expansion)
    key_dim = filter // num_heads
    # key_dim = 16
    nn = HaloAttention(num_heads, key_dim, block_size, halo_size, strides=strides, out_shape=out_shape, out_bias=True, name=name + "halo")(nn)
    # print(">>>>", nn.shape)
    # nn = keras.layers.Activation(activation=activation)(nn)
    nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name=name + "halo_")
    # round(planes * self.expansion * rb), expansion = 2
    nn = conv2d_no_bias(nn, expanded_filter, 1, name=name + "2_")
    nn = batchnorm_with_activation(nn, activation=None, zero_gamma=True, name=name + "2_")

    # print(">>>>", nn.shape, shortcut.shape)
    nn = keras.layers.Add(name=name + "_add")([shortcut, nn])
    return keras.layers.Activation(activation, name=name + "_out")(nn)


def halo_stack(inputs, blocks, filter, strides=1, expansion=2, num_heads=4, halo_expansion=1, block_size=8, halo_size=4, activation="relu", name=""):
    shortcut = True if strides != 1 or inputs.shape[-1] != filter * 2 or expansion != 2 else False
    nn = halo_block(inputs, filter, strides, shortcut, expansion, num_heads, halo_expansion, block_size, halo_size, activation, name=name + "block1_")
    shortcut = False
    for ii in range(2, blocks + 1):
        block_name = name + "block{}_".format(ii)
        nn = halo_block(nn, filter, 1, shortcut, expansion, num_heads, halo_expansion, block_size, halo_size, activation, name=block_name)
    return nn


def HaloNet(
    halo_block_size=8,  # b
    halo_halo_size=3,  # h
    halo_expansion=1,  # rv
    expansion=1,  # rb
    output_conv_channel=-1,  # df
    num_blocks=[3, 3, 10, 3],
    num_heads=[4, 8, 8, 8],
    out_channels=[64, 128, 256, 512],
    strides=[1, 2, 2, 2],
    input_shape=(256, 256, 3),
    num_classes=1000,
    activation="relu",
    classifier_activation="softmax",
    pretrained=None,
    model_name="halonet",
    **kwargs
):
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

    for id, (num_block, out_channel, num_head, stride) in enumerate(zip(num_blocks, out_channels, num_heads, strides)):
        name = "stack{}_".format(id + 1)
        nn = halo_stack(
            nn, num_block, out_channel, stride, expansion, num_head, halo_expansion, halo_block_size, halo_halo_size, activation=activation, name=name
        )

    if output_conv_channel > 0:
        nn = conv2d_no_bias(nn, output_conv_channel, 1, name="post_")
        nn = batchnorm_with_activation(nn, activation=activation, name="post_")

    if num_classes > 0:
        nn = keras.layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        nn = keras.layers.Dense(num_classes, activation=classifier_activation, name="predictions")(nn)

    model = keras.models.Model(inputs, nn, name=model_name)
    return model


BLOCK_CONFIGS = {
    "h0": {  # rv = 1, rb = 0.5
        "halo_block_size": 8,  # b
        "halo_halo_size": 3,  # h
        "halo_expansion": 1,  # rv
        "expansion": 0.5,  # rb
        "output_conv_channel": -1,  # df
        "num_blocks": [3, 3, 7, 3],
        "num_heads": [4, 8, 8, 8],
    },
    "h1": {  # rv = 1, rb = 1
        "halo_block_size": 8,
        "halo_halo_size": 3,
        "halo_expansion": 1,  # rv
        "expansion": 1,  # rb
        "output_conv_channel": -1,  # df
        "num_blocks": [3, 3, 10, 3],
        "num_heads": [4, 8, 8, 8],
    },
    "h2": {  # rv = 1, rb = 1.25
        "halo_block_size": 8,
        "halo_halo_size": 3,
        "halo_expansion": 1,  # rv
        "expansion": 1.25,  # rb
        "output_conv_channel": -1,  # df
        "num_blocks": [3, 3, 11, 3],
        "num_heads": [4, 8, 8, 8],
    },
    "h3": {  # rv = 1, rb = 1.5
        "halo_block_size": 10,
        "halo_halo_size": 3,
        "halo_expansion": 1,  # rv
        "expansion": 1.5,  # rb
        "output_conv_channel": 1024,  # df
        "num_blocks": [3, 3, 12, 3],
        "num_heads": [4, 8, 8, 8],
    },
    "h4": {  # rv = 1, rb = 3
        "halo_block_size": 12,
        "halo_halo_size": 2,
        "halo_expansion": 1,  # rv
        "expansion": 3,  # rb
        "output_conv_channel": 1280,  # df
        "num_blocks": [3, 3, 12, 3],
        "num_heads": [4, 8, 8, 8],
    },
    "h5": {  # rv = 2.5, rb = 2
        "halo_block_size": 14,
        "halo_halo_size": 2,
        "halo_expansion": 2.5,  # rv
        "expansion": 2,  # rb
        "output_conv_channel": 1536,  # df
        "num_blocks": [3, 3, 23, 3],
        "num_heads": [4, 8, 8, 8],
    },
    "h6": {  # rv = 3, rb = 2.75
        "halo_block_size": 8,
        "halo_halo_size": 4,
        "halo_expansion": 3,  # rv
        "expansion": 2.75,  # rb
        "output_conv_channel": 1536,  # df
        "num_blocks": [3, 3, 24, 3],
        "num_heads": [4, 8, 8, 8],
    },
    "h7": {  # rv = 4, rb = 3.5
        "halo_block_size": 10,
        "halo_halo_size": 3,
        "halo_expansion": 4,  # rv
        "expansion": 3.5,  # rb
        "output_conv_channel": 2048,  # df
        "num_blocks": [3, 3, 26, 3],
        "num_heads": [4, 8, 8, 8],
    },
}


def HaloNetH0(input_shape=(256, 256, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return HaloNet(**BLOCK_CONFIGS["h0"], model_name="haloneth0", **locals(), **kwargs)


def HaloNetH1(input_shape=(256, 256, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return HaloNet(**BLOCK_CONFIGS["h1"], model_name="haloneth1", **locals(), **kwargs)


def HaloNetH2(input_shape=(256, 256, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return HaloNet(**BLOCK_CONFIGS["h2"], model_name="haloneth2", **locals(), **kwargs)


def HaloNetH3(input_shape=(320, 320, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return HaloNet(**BLOCK_CONFIGS["h3"], model_name="haloneth3", **locals(), **kwargs)


def HaloNetH4(input_shape=(384, 384, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return HaloNet(**BLOCK_CONFIGS["h4"], model_name="haloneth4", **locals(), **kwargs)


def HaloNetH5(input_shape=(448, 448, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return HaloNet(**BLOCK_CONFIGS["h5"], model_name="haloneth5", **locals(), **kwargs)


def HaloNetH6(input_shape=(512, 512, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return HaloNet(**BLOCK_CONFIGS["h6"], model_name="haloneth6", **locals(), **kwargs)


# halo_block_size == 10, strides == [1, 2, 2, 2], 640 % (halo_block_size * 2 * 2 * 2 * 4) == 0
def HaloNetH7(input_shape=(640, 640, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return HaloNet(**BLOCK_CONFIGS["h7"], model_name="haloneth7", **locals(), **kwargs)
