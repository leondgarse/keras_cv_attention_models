"""
A Keras version of `botnet`.
Original TensorFlow version: https://gist.github.com/aravindsrinivas/56359b79f0ce4449bcb04ab4b56a57a2
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras import backend as K
import os

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5


@tf.keras.utils.register_keras_serializable(package="Custom")
class MHSAWithPositionEmbedding(keras.layers.Layer):
    def __init__(self, num_heads=4, key_dim=128, relative=True, out_shape=None, out_weight=True, out_bias=False, attn_dropout=0, **kwargs):
        super(MHSAWithPositionEmbedding, self).__init__(**kwargs)
        self.num_heads, self.key_dim, self.relative, self.attn_dropout = num_heads, key_dim, relative, attn_dropout
        self.out_bias, self.out_weight = out_bias, out_weight
        self.emb_dim = self.num_heads * self.key_dim
        self.out_shape = self.emb_dim if out_shape is None or not out_weight else out_shape
        self.qk_scale = 1.0 / tf.math.sqrt(tf.cast(self.key_dim, self._compute_dtype_object))
        self._built_from_signature = False

    def _build_from_signature(self, query, value, key=None):
        if hasattr(query, "shape"):
            _, hh, ww, cc = query.shape
        else:
            _, hh, ww, cc = query
        stddev = self.key_dim ** -0.5
        self.final_out_shape = (None, hh, ww, self.out_shape)

        self.query_dense = self.add_weight("query", shape=[cc, self.emb_dim], trainable=True)
        self.key_dense = self.add_weight("key", shape=[cc, self.emb_dim], trainable=True)
        self.value_dense = self.add_weight("value", shape=[cc, self.emb_dim], trainable=True)
        if self.out_weight:
            self.out_dense_ww = self.add_weight("output_weight", shape=[self.emb_dim, self.out_shape], trainable=True)
        if self.out_bias:
            self.out_dense_bb = self.add_weight("output_bias", shape=self.out_shape, initializer='zeros', trainable=True)

        if self.attn_dropout > 0:
            self.attn_dropout_layer = keras.layers.Dropout(rate=self.attn_dropout)

        if self.relative:
            # Relative positional embedding
            pos_emb_w_shape = (self.key_dim, 2 * ww - 1)
            pos_emb_h_shape = (self.key_dim, 2 * hh - 1)
        else:
            # Absolute positional embedding
            pos_emb_w_shape = (self.key_dim, ww)
            pos_emb_h_shape = (self.key_dim, hh)

        self.pos_emb_w = self.add_weight(
            name="r_width",
            shape=pos_emb_w_shape,
            initializer=tf.random_normal_initializer(stddev=stddev),
            trainable=True,
        )
        self.pos_emb_h = self.add_weight(
            name="r_height",
            shape=pos_emb_h_shape,
            initializer=tf.random_normal_initializer(stddev=stddev),
            trainable=True,
        )
        self._built_from_signature = True

    def get_config(self):
        base_config = super(MHSAWithPositionEmbedding, self).get_config()
        base_config.update({
            "num_heads": self.num_heads,
            "key_dim": self.key_dim,
            "relative": self.relative,
            "out_shape": self.out_shape,
            "out_weight": self.out_weight,
            "out_bias": self.out_bias,
            "attn_dropout": self.attn_dropout
        })
        return base_config

    def rel_to_abs(self, rel_pos):
        """
        Converts relative indexing to absolute.
        Input: [bs, heads, height, width, 2*width - 1]
        Output: [bs, heads, height, width, width]
        """
        _, heads, hh, ww, dim = rel_pos.shape  # [bs, heads, height, width, 2 * width - 1]
        # [bs, heads, height, width * (2 * width - 1)] --> [bs, heads, height, width * (2 * width - 1) - width]
        flat_x = tf.reshape(rel_pos, [-1, heads, hh, ww * (ww * 2 - 1)])[:, :, :, ww - 1 : -1]
        # [bs, heads, height, width, 2 * (width - 1)] --> [bs, heads, height, width, width]
        return tf.reshape(flat_x, [-1, heads, hh, ww, 2 * (ww - 1)])[:, :, :, :, :ww]

    def relative_logits(self, query):
        query_w = tf.transpose(query, [0, 3, 1, 2, 4])  # e.g.: [1, 4, 14, 16, 128], [bs, heads, hh, ww, dims]
        rel_logits_w = tf.matmul(query_w, self.pos_emb_w)   # [1, 4, 14, 16, 31], 2 * 16 - 1 == 31
        rel_logits_w = self.rel_to_abs(rel_logits_w)    # [1, 4, 14, 16, 16]

        query_h = tf.transpose(query, [0, 3, 2, 1, 4]) # [1, 4, 16, 14, 128], [bs, heads, ww, hh, dims], Exchange `ww` and `hh`
        rel_logits_h = tf.matmul(query_h, self.pos_emb_h)  # [1, 4, 16, 14, 27], 2 * 14 - 1 == 27
        rel_logits_h = self.rel_to_abs(rel_logits_h)  # [1, 4, 16, 14, 14]
        rel_logits_h = tf.transpose(rel_logits_h, [0, 1, 3, 2, 4]) # [1, 4, 14, 16, 14], transpose back

        return tf.expand_dims(rel_logits_w, axis=-2) + tf.expand_dims(rel_logits_h, axis=-1) # [1, 4, 14, 16, 14, 16]

    def absolute_logits(self, query):
        pos_emb = tf.expand_dims(self.pos_emb_h, 2) + tf.expand_dims(self.pos_emb_w, 1)
        abs_logits = tf.einsum("bxyhd,dpq->bhxypq", query, pos_emb)
        return abs_logits

    def call(self, inputs, return_attention_scores=False, training=None):
        if not self._built_from_signature:
            self._build_from_signature(query=inputs, value=inputs)

        # attn_query = [batch, num_heads, hh * ww, key_dim]
        # pos_query = [batch, hh, ww, num_heads, key_dim]
        target_shape = [-1, inputs.shape[1] * inputs.shape[2], self.num_heads, self.key_dim]
        query = tf.matmul(inputs, self.query_dense)
        query = tf.multiply(query, self.qk_scale)
        attn_query = tf.transpose(tf.reshape(query, target_shape), [0, 2, 1, 3])
        pos_query = tf.reshape(query, [-1, inputs.shape[1], inputs.shape[2], self.num_heads, self.key_dim])

        # key = [batch, num_heads, key_dim, hh * ww], as will matmul with attn_query
        key = tf.matmul(inputs, self.key_dense)
        key = tf.transpose(tf.reshape(key, target_shape), [0, 2, 3, 1])

        # value = [batch, num_heads, hh * ww, key_dim]
        value = tf.matmul(inputs, self.value_dense)
        value = tf.transpose(tf.reshape(value, target_shape), [0, 2, 1, 3])
        # print(query.shape, key.shape, value.sahpe)

        # scaled_dot_product_attention
        attention_scores = tf.matmul(attn_query, key)   # [batch, num_heads, hh * ww, hh * ww]

        if self.relative:
            # Add relative positional embedding
            pos_emb = self.relative_logits(pos_query)
        else:
            # Add absolute positional embedding
            pos_emb = self.absolute_logits(pos_query)
        attention_scores += tf.reshape(pos_emb, [-1, *attention_scores.shape[1:]])
        attention_scores = tf.nn.softmax(attention_scores, axis=-1)

        if self.attn_dropout > 0:
            attention_scores = self.attn_dropout_layer(attention_scores, training=training)

        attention_output = tf.matmul(attention_scores, value)  # [batch, num_heads, hh * ww, key_dim]
        attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
        attention_output = tf.reshape(attention_output, [-1, inputs.shape[1], inputs.shape[2], self.num_heads * self.key_dim])

        if self.out_weight:
            # [batch, hh, ww, num_heads * key_dim] * [num_heads * key_dim, out] --> [batch, hh, ww, out]
            attention_output = tf.matmul(attention_output, self.out_dense_ww)
        if self.out_bias:
            attention_output += self.out_dense_bb
        attention_output.set_shape(self.final_out_shape)

        if return_attention_scores:
            return attention_output, attention_scores
        return attention_output

    def load_resized_pos_emb(self, source_layer):
        # For input 224 --> [128, 27], convert to 480 --> [128, 30]
        image_like_w = tf.expand_dims(tf.transpose(source_layer.pos_emb_w, [1, 0]), 0)
        resize_w = tf.image.resize(image_like_w, (1, self.pos_emb_w.shape[1]))[0]
        self.pos_emb_w.assign(tf.transpose(resize_w, [1, 0]))

        image_like_h = tf.expand_dims(tf.transpose(source_layer.pos_emb_h, [1, 0]), 0)
        resize_h = tf.image.resize(image_like_h, (1, self.pos_emb_h.shape[1]))[0]
        self.pos_emb_h.assign(tf.transpose(resize_h, [1, 0]))


def batchnorm_with_activation(inputs, activation="relu", zero_gamma=False, name=""):
    """Performs a batch normalization followed by an activation. """
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


def bot_block(
    featuremap,
    heads=4,
    proj_factor=4,
    activation="relu",
    relative_pe=True,
    strides=1,
    target_dimension=2048,
    name="all2all",
    use_MHSA=True,
):
    if strides != 1 or featuremap.shape[-1] != target_dimension:
        # padding = "SAME" if strides == 1 else "VALID"
        shortcut = conv2d_no_bias(featuremap, target_dimension, 1, strides=strides, name=name + "_shorcut_")
        bn_act = activation if use_MHSA else None
        # bn_act = None
        shortcut = batchnorm_with_activation(shortcut, activation=bn_act, zero_gamma=False, name=name + "_shorcut_")
    else:
        shortcut = featuremap

    bottleneck_dimension = target_dimension // proj_factor

    nn = conv2d_no_bias(featuremap, bottleneck_dimension, 1, strides=1, padding="VALID", name=name + "_1_")
    nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name=name + "_1_")

    if use_MHSA:  # BotNet block
        key_dim = bottleneck_dimension // heads
        nn = MHSAWithPositionEmbedding(num_heads=heads, key_dim=key_dim, relative=relative_pe, out_weight=False, name=name + "_2_mhsa")(
            nn
        )
        if strides != 1:
            # nn = layers.ZeroPadding2D(padding=1, name=name + "pad")(nn)
            # nn = layers.AveragePooling2D(pool_size=3, strides=strides, name=name + "pool")(nn)
            nn = layers.AveragePooling2D(pool_size=(2, 2), strides=(2, 2), padding="same")(nn)
    else:  # ResNet block
        nn = conv2d_no_bias(nn, bottleneck_dimension, 3, strides=strides, padding="SAME", name=name + "_2_")
    nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name=name + "_2_")

    nn = conv2d_no_bias(nn, target_dimension, 1, strides=1, padding="VALID", name=name + "_3_")
    nn = batchnorm_with_activation(nn, activation=None, zero_gamma=True, name=name + "_3_")

    nn = layers.Add(name=name + "_add")([shortcut, nn])
    return layers.Activation(activation, name=name + "_out")(nn)


def bot_stack(
    featuremap,
    target_dimension=2048,
    num_layers=3,
    strides=2,
    activation="relu",
    heads=4,
    proj_factor=4,
    relative_pe=True,
    name="all2all_stack",
    use_MHSA=True,
):
    """ c5 Blockgroup of BoT Blocks. Use `activation=swish` for `silu` """
    for i in range(num_layers):
        featuremap = bot_block(
            featuremap,
            heads=heads,
            proj_factor=proj_factor,
            activation=activation,
            relative_pe=relative_pe,
            strides=strides if i == 0 else 1,
            target_dimension=target_dimension,
            name=name + "block{}".format(i + 1),
            use_MHSA=use_MHSA,
        )
    return featuremap


def BotNet(
    num_blocks,
    strides=1,
    preact=False,
    use_bias=False,
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="relu",
    classifier_activation="softmax",
    pretrained="imagenet",
    model_name="botnet",
    **kwargs
):
    img_input = layers.Input(shape=input_shape)

    nn = layers.ZeroPadding2D(padding=((3, 3), (3, 3)), name="conv1_pad")(img_input)
    nn = layers.Conv2D(64, 7, strides=2, use_bias=use_bias, name="conv1_conv")(nn)

    if not preact:
        nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name="conv1_")
    nn = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name="pool1_pad")(nn)
    nn = layers.MaxPooling2D(3, strides=2, name="pool1_pool")(nn)

    out_channels = [64, 128, 256, 512]
    stack_strides = [1, 2, 2, strides]
    for id, (num_block, out_channel, stride) in enumerate(zip(num_blocks, out_channels, stack_strides)):
        name = "stack{}_".format(id + 1)
        use_MHSA = True if id == len(num_blocks) - 1 else False # use MHSA in the last stack
        nn = bot_stack(nn, out_channel * 4, num_block, strides=stride, activation=activation, relative_pe=True, name=name, use_MHSA=use_MHSA)

    if preact:
        nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name="post_")
    if num_classes > 0:
        nn = layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        nn = layers.Dense(num_classes, activation=classifier_activation, name="predictions")(nn)

    model = keras.models.Model(img_input, nn, name=model_name)
    reload_model_weights(model, input_shape, pretrained)
    return model


def reload_model_weights(model, input_shape=(224, 224, 3), pretrained="imagenet"):
    if not pretrained in ["imagenet"] or not model.name in ["botnet50"]:
        print(">>>> No pretraind available, model will be random initialized")
        return

    pre_url = "https://github.com/leondgarse/keras_cv_attention_models/releases/download/botnet/{}.h5"
    url = pre_url.format(model.name)
    file_name = os.path.basename(url)
    try:
        pretrained_model = keras.utils.get_file(file_name, url, cache_subdir="models")
    except:
        print("[Error] will not load weights, url not found or download failed:", url)
        return
    else:
        print(">>>> Load pretraind from:", pretrained_model)
        model.load_weights(pretrained_model, by_name=True, skip_mismatch=True)

    if input_shape[0] != 224:
        try:
            print(">>>> Reload mismatched PositionalEmbedding weights: {} -> {}".format(224, input_shape[0]))
            bb = keras.models.load_model(pretrained_model)
            for ii in ['stack4_block1_2_mhsa', 'stack4_block2_2_mhsa', 'stack4_block3_2_mhsa']:
                model.get_layer(ii).load_resized_pos_emb(bb.get_layer(ii))
        except:
            pass


def BotNet50(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", strides=1, **kwargs):
    num_blocks = [3, 4, 6, 4]
    return BotNet(**locals(), model_name="botnet50", **kwargs)


def BotNet101(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained=None, strides=1, **kwargs):
    num_blocks = [3, 4, 23, 4]
    return BotNet(**locals(), model_name="botnet101", **kwargs)


def BotNet152(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained=None, strides=1, **kwargs):
    num_blocks = [3, 8, 36, 4]
    return BotNet(**locals(), model_name="botnet152", **kwargs)
