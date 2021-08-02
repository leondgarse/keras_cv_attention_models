import tensorflow as tf
from tensorflow import keras
from einops import rearrange

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


# @tf.keras.utils.register_keras_serializable(package="Custom")
class HaloAttention(keras.layers.Layer):
    def __init__(self, num_heads=4, key_dim=128, block_size=2, halo_size=1, out_shape=None, out_weight=True, out_bias=False, attn_dropout=0, **kwargs):
        super(HaloAttention, self).__init__(**kwargs)
        self.num_heads, self.key_dim, self.block_size, self.halo_size = num_heads, key_dim, block_size, halo_size
        self.attn_dropout = attn_dropout
        self.out_bias, self.out_weight = out_bias, out_weight
        self.emb_dim = self.num_heads * self.key_dim
        self.out_shape = self.emb_dim if out_shape is None or not out_weight else out_shape
        self.qk_scale = 1.0 / tf.math.sqrt(tf.cast(self.key_dim, self._compute_dtype_object))
        self.kv_kernel = self.block_size + self.halo_size * 2

    def build(self, inputs):
        if hasattr(inputs, "shape"):
            _, hh, ww, cc = inputs.shape
        else:
            _, hh, ww, cc = inputs
        stddev = self.key_dim ** -0.5
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

BLOCK_CONFIGS =     {
    "b0": {
        "halo_block_size": 8,
        "halo_halo_size": 3,
        "halo_dims": 16, # 16 * rv
        "output_conv_channel": -1,   # df
        "num_blocks": [3, 3, 7, 3],
        "out_channels": [32, 64, 128, 256], # [64, 128, 256, 512] * rb
        "num_heads": [4, 8, 8, 8],
    }
    # {'block_size':8, 'halo_size':3,'stage0':(3,4), 'stage1':(3,8), 'stage2':(7,8), 'stage3':(3,8), 'rv':1, 'rb':0.5, 'df':-1}
    # {'block_size':8, 'halo_size':3,'stage0':(3,4), 'stage1':(3,8), 'stage2':(10,8), 'stage3':(3,8), 'rv':1, 'rb':1, 'df':-1}
    # {'block_size':8, 'halo_size':3,'stage0':(3,4), 'stage1':(3,8), 'stage2':(11,8), 'stage3':(3,8), 'rv':1, 'rb':1.25, 'df':-1}
    # {'block_size':10, 'halo_size':3,'stage0':(3,4), 'stage1':(3,8), 'stage2':(12,8), 'stage3':(3,8), 'rv':1, 'rb':1.5, 'df':1024}
    # {'block_size':12, 'halo_size':2,'stage0':(3,4), 'stage1':(3,8), 'stage2':(12,8), 'stage3':(3,8), 'rv':1, 'rb':3, 'df':1280}
    # {'block_size':14, 'halo_size':2,'stage0':(3,4), 'stage1':(3,8), 'stage2':(23,8), 'stage3':(3,8), 'rv':2.5, 'rb':2, 'df':1536}
    # {'block_size':8, 'halo_size':4,'stage0':(3,4), 'stage1':(3,8), 'stage2':(24,8), 'stage3':(3,8), 'rv':3, 'rb':2.75, 'df':1536}
    # {'block_size':10, 'halo_size':3,'stage0':(3,4), 'stage1':(3,8), 'stage2':(24,8), 'stage3':(3,8), 'rv':4, 'rb':3.50, 'df':2048}
}

def batchnorm_with_activation(inputs, activation="relu", zero_gamma=False, name=""):
    """Performs a batch normalization followed by an activation. zero_gamma: https://arxiv.org/abs/1706.02677 """
    bn_axis = 3 if backend.image_data_format() == "channels_last" else 1
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
