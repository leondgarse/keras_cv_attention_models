import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.keras import backend as K

try:
    from tensorflow_addons.layers import StochasticDepth
except:
    pass

import sys
sys.path.append("../Keras_insightface/backbones")
from botnet import MHSAWithPositionEmbedding
from resnest import split_attention_conv2d

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5
CONV_KERNEL_INITIALIZER = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")
# CONV_KERNEL_INITIALIZER = 'glorot_uniform'


def batchnorm_with_activation(inputs, activation="relu", name=""):
    """Performs a batch normalization followed by an activation. """
    bn_axis = 3 if K.image_data_format() == "channels_last" else 1
    nn = layers.BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        name=name + "bn",
    )(inputs)
    if activation:
        nn = layers.Activation(activation=activation, name=name + activation)(nn)
    return nn


def conv2d_with_init(inputs, filters, kernel_size, strides=1, padding="VALID", use_bias=False, name="", **kwargs):
    if padding.upper() == "SAME":
        inputs = layers.ZeroPadding2D(kernel_size // 2)(inputs)
    return layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="VALID",
        use_bias=use_bias,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name + "conv",
        **kwargs,
    )(inputs)


def patch_stem(inputs, hidden_dim=64, stem_width=384, patch_size=8, strides=2, activation="relu", name=""):
    nn = conv2d_with_init(inputs, hidden_dim, 7, strides=strides, padding="same", name=name + "1_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "1_")
    nn = conv2d_with_init(nn, hidden_dim, 3, strides=1, padding="same", name=name + "2_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "2_")
    nn = conv2d_with_init(nn, hidden_dim, 3, strides=1, padding="same", name=name + "3_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "3_")

    patch_step = patch_size // strides
    return conv2d_with_init(nn, stem_width, patch_step, strides=patch_step, use_bias=True, name=name + "patch_")

@tf.keras.utils.register_keras_serializable(package="Custom")
class UnfoldMatmulFold(layers.Layer):
    """
    As the name `fold_overlap_1` indicates, works only if overlap happens once in fold, like `kernel_size=3, strides=2`.
    For `kernel_size=3, strides=1`, overlap happens twice, will NOT work...
    """

    def __init__(self, final_output_shape, kernel_size=1, padding=1, strides=1, **kwargs):
        super(UnfoldMatmulFold, self).__init__(**kwargs)
        self.kernel_size, self.padding, self.strides = kernel_size, padding, strides

        height, width, channel = final_output_shape
        self.final_output_shape = (height, width, channel)
        self.out_channel = channel

        self.patch_kernel = [1, kernel_size, kernel_size, 1]
        self.patch_strides = [1, strides, strides, 1]
        overlap_pad = [0, 2 * strides - kernel_size]
        hh_pad = [0, 0] if int(tf.math.ceil(height / 2)) % 2 == 0 else [0, 1]  # Make patches number even
        ww_pad = [0, 0] if int(tf.math.ceil(width / 2)) % 2 == 0 else [0, 1]  # Make patches number even
        self.fold_overlap_pad = [[0, 0], hh_pad, ww_pad, overlap_pad, overlap_pad, [0, 0]]

        if padding == 1:
            pad = kernel_size // 2
            self.patch_pad = [[0, 0], [pad, pad], [pad, pad], [0, 0]]
            self.out_start_h, self.out_start_w = pad, pad
        else:
            self.patch_pad = [[0, 0], [0, 0], [0, 0], [0, 0]]
            self.out_start_h, self.out_start_w = 0, 0
        self.out_end_h, self.out_end_w = self.out_start_h + height, self.out_start_w + width

    def pad_overlap(self, patches, start_h, start_w):
        bb = patches[:, start_h::2, :, start_w::2, :, :]  # [1, 7, 4, 7, 4, 192]
        bb = tf.reshape(bb, [-1, bb.shape[1] * bb.shape[2], bb.shape[3] * bb.shape[4], bb.shape[-1]])  # [1, 28, 28, 192]
        pad_h = [0, self.strides] if start_h == 0 else [self.strides, 0]
        pad_w = [0, self.strides] if start_w == 0 else [self.strides, 0]
        padding = [[0, 0], pad_h, pad_w, [0, 0]]
        bb = tf.pad(bb, padding)  # [1, 30, 30, 192]
        return bb

    def fold_overlap_1(self, patches):
        aa = tf.pad(patches, self.fold_overlap_pad)  # [1, 14, 14, 4, 4, 192], 14 // 2 * 4 == 28
        aa = tf.transpose(aa, [0, 1, 3, 2, 4, 5])  # [1, 14, 4, 14, 4, 192]
        cc = self.pad_overlap(aa, 0, 0) + self.pad_overlap(aa, 0, 1) + self.pad_overlap(aa, 1, 0) + self.pad_overlap(aa, 1, 1)
        return cc[:, self.out_start_h : self.out_end_h, self.out_start_w : self.out_end_w, :]  # [1, 28, 28, 192]

    def call(self, inputs, **kwargs):
        # vv: [1, 28, 28, 192], attn: [1, 14, 14, 6, 9, 9]
        vv, attn = inputs[0], inputs[1]
        hh, ww, num_head = attn.shape[1], attn.shape[2], attn.shape[3]

        """ unfold """
        # [1, 30, 30, 192], do SAME padding
        pad_vv = tf.pad(vv, self.patch_pad)
        # [1, 14, 14, 1728]
        patches = tf.image.extract_patches(pad_vv, self.patch_kernel, self.patch_strides, [1, 1, 1, 1], padding="VALID")

        """ matmul """
        # mm = einops.rearrange(patches, 'D H W (k h p) -> D H W h k p', h=num_head, k=self.kernel_size * self.kernel_size)
        # mm = tf.matmul(attn, mm)
        # mm = einops.rearrange(mm, 'D H W h (kh kw) p -> D H W kh kw (h p)', h=num_head, kh=self.kernel_size, kw=self.kernel_size)
        # [1, 14, 14, 9, 6, 32], the last 2 dimenions are channel 6 * 32 == 192
        mm = tf.reshape(patches, [-1, hh, ww, self.kernel_size * self.kernel_size, num_head, self.out_channel // num_head])
        # [1, 14, 14, 6, 9, 32], meet the dimenion of attn for matmul
        mm = tf.transpose(mm, [0, 1, 2, 4, 3, 5])
        # [1, 14, 14, 6, 9, 32], The last two dimensions [9, 9] @ [9, 32] --> [9, 32]
        mm = tf.matmul(attn, mm)
        # [1, 14, 14, 9, 6, 32], transpose back
        mm = tf.transpose(mm, [0, 1, 2, 4, 3, 5])
        # [1, 14, 14, 3, 3, 192], split kernel_dimension: 9 --> [3, 3], merge channel_dimmension: [6, 32] --> 192
        mm = tf.reshape(mm, [-1, hh, ww, self.kernel_size, self.kernel_size, self.out_channel])

        """ fold """
        # [1, 28, 28, 192]
        output = self.fold_overlap_1(mm)
        output.set_shape((None, *self.final_output_shape))
        return output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], *self.final_output_shape)

    def get_config(self):
        config = super(UnfoldMatmulFold, self).get_config()
        config.update(
            {
                "final_output_shape": self.final_output_shape,
                "kernel_size": self.kernel_size,
                "padding": self.padding,
                "strides": self.strides,
            }
        )
        return config


def outlook_attention(inputs, embed_dim, num_head, kernel_size=3, padding=1, strides=2, attn_dropout=0, output_dropout=0, name=""):
    _, height, width, channel = inputs.shape
    FLOAT_DTYPE = tf.keras.mixed_precision.global_policy().compute_dtype
    qk_scale = tf.math.sqrt(tf.cast(embed_dim // num_head, FLOAT_DTYPE))
    hh, ww = int(tf.math.ceil(height / strides)), int(tf.math.ceil(width / strides))

    vv = layers.Dense(embed_dim, use_bias=False, name=name + "v")(inputs)

    """ attention """
    # [1, 14, 14, 192]
    attn = layers.AveragePooling2D(pool_size=strides, strides=strides)(inputs)
    # [1, 14, 14, 486]
    attn = layers.Dense(kernel_size ** 4 * num_head, name=name + "attn")(attn) / qk_scale
    # [1, 14, 14, 6, 9, 9]
    attn = tf.reshape(attn, (-1, hh, ww, num_head, kernel_size * kernel_size, kernel_size * kernel_size))
    attention_weights = tf.nn.softmax(attn, axis=-1)
    if attn_dropout > 0:
        attention_weights = layers.Dropout(attn_dropout)(attention_weights)

    """ Unfold --> Matmul --> Fold, no weights in this process """
    output = UnfoldMatmulFold((height, width, embed_dim), kernel_size, padding, strides)([vv, attention_weights])
    output = layers.Dense(embed_dim, use_bias=True, name=name + "out")(output)

    if output_dropout > 0:
        output = layers.Dropout(output_dropout)(output)

    return output


def outlook_attention_simple(inputs, embed_dim, kernel_size=3, num_head=6, attn_dropout=0, name=""):
    key_dim = embed_dim // num_head
    FLOAT_DTYPE = tf.keras.mixed_precision.global_policy().compute_dtype
    qk_scale = tf.math.sqrt(tf.cast(key_dim, FLOAT_DTYPE))

    height, width = inputs.shape[1], inputs.shape[2]
    hh, ww = int(tf.math.ceil(height / kernel_size)), int(tf.math.ceil(width / kernel_size)) # 14, 14
    padded = hh * kernel_size - height
    if padded != 0:
        inputs = keras.layers.ZeroPadding2D(((0, padded), (0, padded)))(inputs)

    vv = keras.layers.Dense(embed_dim, use_bias=False, name=name + "v")(inputs)
    # vv = einops.rearrange(vv, "D (h hk) (w wk) (H p) -> D h w H (hk wk) p", hk=kernel_size, wk=kernel_size, H=num_head, p=key_dim)
    vv = tf.reshape(vv, (-1, hh, kernel_size, ww, kernel_size, num_head, key_dim))  # [1, 14, 2, 14, 2, 6, 32]
    vv = tf.transpose(vv, [0, 1, 3, 5, 2, 4, 6])
    vv = tf.reshape(vv, [-1, hh, ww, num_head, kernel_size * kernel_size, key_dim])  # [1, 14, 14, 6, 4, 32]

    # attn = keras.layers.AveragePooling2D(pool_size=3, strides=2, padding='SAME')(inputs)
    attn = keras.layers.AveragePooling2D(pool_size=kernel_size, strides=kernel_size)(inputs)
    attn = keras.layers.Dense(kernel_size ** 4 * num_head, use_bias=True, name=name + "attn")(attn) / qk_scale
    attn = tf.reshape(attn , [-1, hh, ww, num_head, kernel_size * kernel_size, kernel_size * kernel_size]) # [1, 14, 14, 6, 4, 4]
    attn = tf.nn.softmax(attn, axis=-1)
    if attn_dropout > 0:
        attn = keras.layers.Dropout(attn_dropout)(attn)

    out = tf.matmul(attn, vv) # [1, 14, 14, 6, 4, 32]
    # out = einops.rearrange(out, "D h w H (hk wk) p -> D (h hk) (w wk) (H p)", hk=kernel_size, wk=kernel_size)  # [1, 28, 28, 192]
    out = tf.reshape(out, [-1, hh, ww, num_head, kernel_size, kernel_size, key_dim])  # [1, 14, 14, 6, 2, 2, 32]
    out = tf.transpose(out, [0, 1, 4, 2, 5, 3, 6])  # [1, 14, 2, 14, 2, 6, 32]
    out = tf.reshape(out, [-1, inputs.shape[1], inputs.shape[2], embed_dim])  # [1, 28, 28, 192]
    if padded != 0:
        out = out[:, :-padded, :-padded, :]
    out = keras.layers.Dense(embed_dim, use_bias=True, name=name + "out")(out)

    return out

def scaled_dot_product_attention(q, k, v, qk_scale, output_shape, attn_dropout=0, output_dropout=0, name=""):
    scaled_attention_logits = layers.Lambda(lambda aa: tf.matmul(aa[0], aa[1], transpose_b=True))([q, k]) / qk_scale
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    if attn_dropout > 0:
        attention_weights = layers.Dropout(attn_dropout)(attention_weights)

    output = layers.Lambda(lambda bb: tf.matmul(bb[0], bb[1]))([attention_weights, v])
    output = tf.transpose(output, perm=[0, 2, 1, 3])

    output = tf.reshape(output, output_shape)
    output = layers.Dense(output_shape[-1], use_bias=True, name=name + "out")(output)
    if output_dropout > 0:
        output = layers.Dropout(output_dropout)(output)
    return output


def class_attention(inputs, embed_dim, num_head, attn_dropout=0, output_dropout=0, name=""):
    _, width, channel = inputs.shape
    depth = embed_dim // num_head
    FLOAT_DTYPE = tf.keras.mixed_precision.global_policy().compute_dtype
    qk_scale = tf.math.sqrt(tf.cast(depth, FLOAT_DTYPE))

    q = layers.Dense(embed_dim, use_bias=False, name=name + "q")(inputs[:, :1, :])
    q = tf.reshape(q, (-1, num_head, 1, depth))

    kv = layers.Dense(embed_dim * 2, use_bias=False, name=name + "kv")(inputs)
    kv = tf.reshape(kv, (-1, width, 2, num_head, depth))
    kv = tf.transpose(kv, perm=[2, 0, 3, 1, 4])
    k, v = kv[0], kv[1]

    output_shape = (-1, 1, embed_dim)
    return scaled_dot_product_attention(q, k, v, qk_scale, output_shape, attn_dropout, output_dropout, name=name)


def multi_head_self_attention(inputs, embed_dim, num_head, attn_dropout=0, output_dropout=0, name=""):
    _, height, width, channel = inputs.shape
    depth = embed_dim // num_head
    FLOAT_DTYPE = tf.keras.mixed_precision.global_policy().compute_dtype
    qk_scale = tf.math.sqrt(tf.cast(depth, FLOAT_DTYPE))

    qkv = layers.Dense(embed_dim * 3, use_bias=False, name=name + "qkv")(inputs)
    qkv = tf.reshape(qkv, (-1, height * width, 3, num_head, embed_dim // num_head))
    qkv = tf.transpose(qkv, perm=[2, 0, 3, 1, 4])
    q, k, v = qkv[0], qkv[1], qkv[2]

    output_shape = (-1, height, width, embed_dim)
    return scaled_dot_product_attention(q, k, v, qk_scale, output_shape, attn_dropout, output_dropout, name=name)

@tf.keras.utils.register_keras_serializable(package="Custom")
class BiasLayer(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(BiasLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight('bias', shape=input_shape[-1], initializer='zeros', trainable=True)

    def call(self, inputs):
        return inputs + self.bias

def attention_mlp_block(inputs, embed_dim, num_head=1, mlp_ratio=3, attention_type=None, survival=None, mlp_activation="gelu", dropout=0, name=""):
    nn_0 = inputs[:, :1] if attention_type == "class" else inputs
    nn_1 = layers.LayerNormalization(epsilon=BATCH_NORM_EPSILON, name=name + "LN")(inputs)

    if attention_type == "outlook":
        nn_1 = outlook_attention(nn_1, embed_dim, num_head=num_head, name=name + "attn_")
    elif attention_type == "outlook_simple":
        nn_1 = outlook_attention_simple(nn_1, embed_dim, num_head=num_head, name=name + "attn_")
    elif attention_type == "class":
        nn_1 = class_attention(nn_1, embed_dim, num_head=num_head, name=name + "attn_")
    elif attention_type == "mhsa":
        nn_1 = multi_head_self_attention(nn_1, embed_dim, num_head=num_head, name=name + "attn_")
    elif attention_type == "mhsa_rp":
        nn_1 = MHSAWithPositionEmbedding(num_heads=num_head, key_dim=embed_dim//num_head, out_shape=embed_dim, out_bias=True, name=name + "attn")(nn_1)
        # nn_1 = BiasLayer(name=name + "attn_bias")(nn_1) # bias for output dense

    if survival is not None and survival < 1:
        nn_1 = StochasticDepth(float(survival))([nn_0, nn_1])
    else:
        nn_1 = layers.Add()([nn_0, nn_1])

    """ MLP """
    nn_2 = layers.LayerNormalization(epsilon=BATCH_NORM_EPSILON, name=name + "mlp_LN")(nn_1)
    nn_2 = layers.Dense(embed_dim * mlp_ratio, name=name + "mlp_dense_1")(nn_2)
    # gelu with approximate=False using `erf` leads to GPU memory leak...
    # nn_2 = layers.Activation("gelu", name=name + "mlp_" + mlp_activation)(nn_2)
    nn_2 = tf.nn.gelu(nn_2, approximate=True)
    nn_2 = layers.Dense(embed_dim, name=name + "mlp_dense_2")(nn_2)
    if dropout > 0:
        nn_2 = layers.Dropout(dropout)(nn_2)

    if survival is not None and survival < 1:
        out = StochasticDepth(float(survival))([nn_1, nn_2])
    else:
        out = layers.Add()([nn_1, nn_2])
    if attention_type == "class":
        out = tf.concat([out, inputs[:, 1:]], axis=1)
    return out


@tf.keras.utils.register_keras_serializable(package="Custom")
class PositionalEmbedding(layers.Layer):
    def __init__(self, **kwargs):
        super(PositionalEmbedding, self).__init__(**kwargs)
        self.pp_init = tf.initializers.TruncatedNormal(stddev=0.2)

    def build(self, input_shape):
        hh, ww, cc = input_shape[1:]
        self.pp = self.add_weight(name="positional_embedding", shape=(1, hh, ww, cc), initializer=self.pp_init, trainable=True)
        super(PositionalEmbedding, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs + self.pp

    def load_resized_pos_emb(self, source_layer):
        # For input 224 --> [1, 14, 14, 384], convert to 384 --> [1, 24, 24, 384]
        self.pp.assign(tf.image.resize(source_layer.pp, self.pp.shape[1:3]))


@tf.keras.utils.register_keras_serializable(package="Custom")
class ClassToken(layers.Layer):
    def __init__(self, **kwargs):
        super(ClassToken, self).__init__(**kwargs)
        self.token_init = tf.initializers.TruncatedNormal(stddev=0.2)

    def build(self, input_shape):
        self.class_tokens = self.add_weight(name="tokens", shape=(1, 1, input_shape[-1]), initializer=self.token_init, trainable=True)
        super(ClassToken, self).build(input_shape)

    def call(self, inputs, **kwargs):
        class_tokens = tf.tile(self.class_tokens, [tf.shape(inputs)[0], 1, 1])
        return tf.concat([class_tokens, inputs], axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 1, input_shape[2])


@tf.keras.utils.register_keras_serializable(package="Custom")
class MixupToken(layers.Layer):
    def __init__(self, scale=2, beta=1.0, **kwargs):
        super(MixupToken, self).__init__(**kwargs)
        self.scale, self.beta = scale, beta

    def call(self, inputs, training=None, **kwargs):
        height, width = tf.shape(inputs)[1], tf.shape(inputs)[2]
        # tf.print("training:", training)
        def _call_train():
            return tf.stack(self.rand_bbox(height, width))
        def _call_test():
            return tf.cast(tf.stack([0, 0, 0, 0]), "int32") # No mixup area for test
        return K.in_train_phase(_call_train, _call_test, training=training)

    def sample_beta_distribution(self):
        gamma_1_sample = tf.random.gamma(shape=[], alpha=self.beta)
        gamma_2_sample = tf.random.gamma(shape=[], alpha=self.beta)
        return gamma_1_sample / (gamma_1_sample + gamma_2_sample)

    def rand_bbox(self, height, width):
        random_lam = tf.cast(self.sample_beta_distribution(), self.compute_dtype)
        cut_rate = tf.sqrt(1.0 - random_lam)
        s_height, s_width = height // self.scale, width // self.scale

        right_pos = tf.random.uniform(shape=[], minval=0, maxval=s_width, dtype=tf.int32)
        bottom_pos = tf.random.uniform(shape=[], minval=0, maxval=s_height, dtype=tf.int32)
        left_pos = right_pos - tf.cast(tf.cast(s_width, cut_rate.dtype) * cut_rate, "int32") // 2
        top_pos = bottom_pos - tf.cast(tf.cast(s_height, cut_rate.dtype) * cut_rate, "int32") // 2
        left_pos, top_pos = tf.maximum(left_pos, 0), tf.maximum(top_pos, 0)

        return left_pos, top_pos, right_pos, bottom_pos

    def do_mixup_token(self, inputs, bbox):
        left, top, right, bottom = bbox
        # if tf.equal(right, 0) or tf.equal(bottom, 0):
        #     return inputs
        sub_ww = inputs[:, :, left:right]
        mix_sub = tf.concat([sub_ww[:, :top], sub_ww[::-1, top:bottom], sub_ww[:, bottom:]], axis=1)
        output = tf.concat([inputs[:, :, :left], mix_sub, inputs[:, :, right:]], axis=2)
        output.set_shape(inputs.shape)
        return output

    def get_config(self):
        config = super(MixupToken, self).get_config()
        config.update({"scale": self.scale, "beta": self.beta})
        return config


def VOLO(
    input_shape,
    num_blocks,
    embed_dims,
    num_heads,
    mlp_ratios,
    stem_hidden_dim=64,
    patch_size=8,
    survivals=None,
    classfiers=2,
    num_classes=1000,
    mix_token=False,
    token_classifier_top=False,
    mean_classifier_top=False,
    token_label_top=False,
    first_attn_type="outlook",
    model_name="VOLO",
    **kwargs,
):
    """
    survivals: is used for [Deep Networks with Stochastic Depth](https://arxiv.org/abs/1603.09382).
        Can be value like `0.5` or `0.8`, indicates the survival probability linearly changes from `1 --> 0.8` for `top --> bottom` layers.
        A higher value means a higher probability will keep the deep branch.
    mix_token: Set True for training using `mix_token`. Should better with `token_label_top=True`.
    token_classifier_top: Set True for using tokens[0] for classify when `mix_token==False`.
    mean_classifier_top: Set True for using all tokens for classify when `mix_token==False`.
    token_label_top: Set True for returning both `token_head` and `aux_head`.
    """
    inputs = layers.Input(input_shape)

    """ forward_embeddings """
    nn = patch_stem(inputs, hidden_dim=stem_hidden_dim, stem_width=embed_dims[0], patch_size=patch_size, strides=2, name="stem_")

    if mix_token:
        scale = 2
        mixup_token = MixupToken(scale=scale)
        bbox = mixup_token(nn)
        nn = mixup_token.do_mixup_token(nn, bbox * scale)

    outlook_attentions = [True, False, False, False]
    downsamples = [True, False, False, False]

    # StochasticDepth survival_probability values
    total_layers = sum(num_blocks)
    if isinstance(survivals, float):
        start, end = 1, survivals
        survivals = [start - (1 - end) * float(ii) / (total_layers - 1) for ii in range(total_layers)]
    else:
        survivals = [None] * total_layers
    survivals = [survivals[int(sum(num_blocks[:id])) : sum(num_blocks[: id + 1])] for id in range(len(num_blocks))]

    """ forward_tokens """
    for stack_id, (bb, ee, hh, mm, outlook_attention, downsample, survival) in enumerate(
        zip(num_blocks, embed_dims, num_heads, mlp_ratios, outlook_attentions, downsamples, survivals)
    ):
        attention_type = first_attn_type if outlook_attention else "mhsa"
        for ii in range(bb):
            # print(">>>>", ii, bb, ee, hh, mm, outlook_attention, downsample, nn.shape, survival[ii])
            name = "stack{}_block{}_".format(stack_id, ii)
            nn = attention_mlp_block(nn, ee, hh, mlp_ratio=mm, attention_type=attention_type, survival=survival[ii], name=name)

        if downsample:
            nn = layers.Conv2D(ee * 2, kernel_size=2, strides=2, name="stack_{}_downsample".format(stack_id))(nn)

        if outlook_attention:
            nn = PositionalEmbedding(name="stack_{}_positional".format(stack_id))(nn)

    if num_classes == 0:
        return tf.keras.models.Model(inputs, nn, name=model_name)

    _, height, width, channel = nn.shape
    nn = tf.reshape(nn, (-1, height * width, channel))

    """ forward_cls """
    nn = ClassToken(name="class_token")(nn)

    embed_dim, num_head, mlp_ratio = embed_dims[-1], num_heads[-1], mlp_ratios[-1]
    for id in range(classfiers):
        name = "classfiers{}_".format(id)
        nn = attention_mlp_block(nn, embed_dim=embed_dim, num_head=num_head, mlp_ratio=mlp_ratio, attention_type="class", name=name)
    nn = layers.LayerNormalization(epsilon=BATCH_NORM_EPSILON, name="pre_out_LN")(nn)

    if token_label_top:
        # Training with label token
        nn_cls = layers.Dense(num_classes, name="token_head")(nn[:, 0])
        nn_aux = layers.Dense(num_classes, name="aux_head")(nn[:, 1:])

        if mix_token:
            nn_aux = tf.reshape(nn_aux, (-1, height, width, num_classes))
            nn_aux = mixup_token.do_mixup_token(nn_aux, bbox)
            nn_aux = layers.Reshape((height * width, num_classes), dtype="float32", name="aux")(nn_aux)

            left, top, right, bottom = bbox
            lam = 1 - ((right - left) * (bottom - top) / nn_aux.shape[1])
            lam_repeat = tf.expand_dims(tf.repeat(lam, tf.shape(inputs)[0], axis=0), 1)
            nn_cls = layers.Concatenate(axis=-1, dtype="float32", name="class")([nn_cls, lam_repeat])

        # nn_lam = layers.Lambda(lambda ii: tf.cast(tf.stack(ii), tf.float32))([left_pos, top_pos, right_pos, bottom_pos])
        nn = [nn_cls, nn_aux]
    elif mean_classifier_top:
        # Return mean of all tokens
        nn = layers.GlobalAveragePooling1D(name="avg_pool")(nn)
        nn = layers.Dense(num_classes, name="token_head")(nn)
    elif token_classifier_top:
        # Return dense classifier using only first token
        nn = layers.Dense(num_classes, name="token_head")(nn[:, 0])
    else:
        # Return token dense for evaluation
        nn_cls = layers.Dense(num_classes, name="token_head")(nn[:, 0])
        nn_aux = layers.Dense(num_classes, name="aux_head")(nn[:, 1:])
        nn = layers.Add()([nn_cls, tf.reduce_max(nn_aux, 1) * 0.5])
    model = tf.keras.models.Model(inputs, nn, name=model_name)
    return model


def volo_d1(input_shape=(224, 224, 3), model_name="volo_d1", **kwargs):
    num_blocks = [4, 4, 8, 2]
    embed_dims = [192, 384, 384, 384]
    num_heads = [6, 12, 12, 12]
    mlp_ratios = [3, 3, 3, 3]
    return VOLO(**locals(), **kwargs)


def volo_d2(input_shape=(224, 224, 3), model_name="volo_d2", **kwargs):
    num_blocks = [6, 4, 10, 4]
    embed_dims = [256, 512, 512, 512]
    num_heads = [8, 16, 16, 16]
    mlp_ratios = [3, 3, 3, 3]
    return VOLO(**locals(), **kwargs)


def volo_d3(input_shape=(224, 224, 3), model_name="volo_d3", **kwargs):
    num_blocks = [8, 8, 16, 4]
    embed_dims = [256, 512, 512, 512]
    num_heads = [8, 16, 16, 16]
    mlp_ratios = [3, 3, 3, 3]
    return VOLO(**locals(), **kwargs)


def volo_d4(input_shape=(224, 224, 3), model_name="volo_d4", **kwargs):
    num_blocks = [8, 8, 16, 4]
    embed_dims = [384, 768, 768, 768]
    num_heads = [12, 16, 16, 16]
    mlp_ratios = [3, 3, 3, 3]
    return VOLO(**locals(), **kwargs)


def volo_d5(input_shape=(224, 224, 3), model_name="volo_d5", **kwargs):
    num_blocks = [12, 12, 20, 4]
    embed_dims = [384, 768, 768, 768]
    num_heads = [12, 16, 16, 16]
    mlp_ratios = [4, 4, 4, 4]
    stem_hidden_dim = 128
    return VOLO(**locals(), **kwargs)
