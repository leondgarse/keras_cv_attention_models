import math
import numpy as np
from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models, functional, image_data_format, initializers, register_keras_serializable
from keras_cv_attention_models.download_and_load import reload_model_weights
from keras_cv_attention_models.attention_layers import (
    activation_by_name,
    batchnorm_with_activation,
    conv2d_no_bias,
    fold_by_conv2d_transpose,
    CompatibleExtractPatches,
    add_pre_post_process,
)


BATCH_NORM_EPSILON = 1e-5
PRETRAINED_DICT = {
    "volo_d1": {"imagenet": {224: "b642d39b05da9f460035d5d5fa617774", 384: "c7632a783d43278608d84f9463743b2e"}},
    "volo_d2": {"imagenet": {224: "19c6c49d3a1020e9fafbcce775200e30", 384: "fc0435d59925e547d9003010a51e4a16"}},
    "volo_d3": {"imagenet": {224: "42ae5c1be8ceb644d4f7c3d894a0034f", 448: "62304a047f182265617c49f74991e6a0"}},
    "volo_d4": {"imagenet": {224: "b45c6518b5e7624b0f6a61f18a5a7bae", 448: "c3e48df2a555032608d48841d2f4a551"}},
    "volo_d5": {"imagenet": {224: "19c98591fb2a97c2a51d9723c2ff6e1d", 448: "6f9858b667cfef77339901c3121c85a1", 512: "f2aa0cb8e265cabee840a6b83858d086"}},
}


def outlook_attention(inputs, embed_dim, num_heads=8, kernel_size=3, padding=1, strides=2, attn_dropout=0, output_dropout=0, name=""):
    _, height, width, channel = inputs.shape
    # qk_scale = float(1.0 / functional.sqrt(functional.cast(embed_dim // num_heads, "float32")))
    qk_scale = 1.0 / (float(embed_dim // num_heads) ** 0.5)
    hh, ww = int(math.ceil(height / strides)), int(math.ceil(width / strides))

    vv = layers.Dense(embed_dim, use_bias=False, name=name + "v")(inputs)

    """ attention """
    # [1, 14, 14, 192]
    pool_padding = "VALID" if height % strides == 0 and width % strides == 0 else "SAME"
    attn = layers.AvgPool2D(pool_size=strides, strides=strides, padding=pool_padding)(inputs)
    # [1, 14, 14, 486]
    attn = layers.Dense(kernel_size**4 * num_heads, name=name + "attn")(attn) * qk_scale
    # [1, 14, 14, 6, 9, 9]
    attn = functional.reshape(attn, (-1, hh, ww, num_heads, kernel_size * kernel_size, kernel_size * kernel_size))
    # attention_weights = functional.softmax(attn, axis=-1)
    attention_weights = layers.Softmax(axis=-1, name=name and name + "attention_scores")(attn)
    if attn_dropout > 0:
        attention_weights = layers.Dropout(attn_dropout)(attention_weights)

    """ unfold """
    # [1, 14, 14, 1728] if compressed else [1, 14, 14, 3, 3, 192]
    # patches = functional.extract_patches(pad_vv, patch_kernel, patch_strides, [1, 1, 1, 1], padding="VALID")
    patches = CompatibleExtractPatches(kernel_size, strides, padding="SAME", compressed=False, name=name)(vv)

    """ matmul """
    # mm = einops.rearrange(patches, 'D H W (k h p) -> D H W h k p', h=num_head, k=kernel_size * kernel_size)
    # mm = functional.matmul(attn, mm)
    # mm = einops.rearrange(mm, 'D H W h (kh kw) p -> D H W kh kw (h p)', h=num_head, kh=kernel_size, kw=kernel_size)
    # [1, 14, 14, 9, 6, 32], the last 2 dimenions are channel 6 * 32 == 192
    mm = functional.reshape(patches, [-1, hh, ww, kernel_size * kernel_size, num_heads, embed_dim // num_heads])
    # [1, 14, 14, 6, 9, 32], meet the dimenion of attn for matmul
    mm = functional.transpose(mm, [0, 1, 2, 4, 3, 5])
    # [1, 14, 14, 6, 9, 32], The last two dimensions [9, 9] @ [9, 32] --> [9, 32]
    # mm = layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([attention_weights, mm])
    mm = attention_weights @ mm
    # [1, 14, 14, 9, 6, 32], transpose back
    mm = functional.transpose(mm, [0, 1, 2, 4, 3, 5])
    # [1, 14, 14, 3, 3, 192], split kernel_dimension: 9 --> [3, 3], merge channel_dimmension: [6, 32] --> 192
    mm = functional.reshape(mm, [-1, hh, ww, kernel_size, kernel_size, embed_dim])

    """ fold """
    # [1, 28, 28, 192]
    output = fold_by_conv2d_transpose(mm, inputs.shape[1:], kernel_size, strides, padding="SAME", compressed=False, name=name)

    # output = UnfoldMatmulFold((height, width, embed_dim), kernel_size, padding, strides)([vv, attention_weights])
    output = layers.Dense(embed_dim, use_bias=True, name=name + "out")(output)

    if output_dropout > 0:
        output = layers.Dropout(output_dropout)(output)

    return output


def outlook_attention_simple(inputs, embed_dim, num_heads=6, kernel_size=3, attn_dropout=0, name=""):
    """Simple version not using unfold and fold"""
    key_dim = embed_dim // num_heads
    # qk_scale = float(1.0 / tf.math.sqrt(tf.cast(key_dim, "float32")))
    qk_scale = 1.0 / (float(embed_dim // num_heads) ** 0.5)

    height, width = inputs.shape[1], inputs.shape[2]
    hh, ww = int(math.ceil(height / kernel_size)), int(math.ceil(width / kernel_size))  # 14, 14
    padded = hh * kernel_size - height
    if padded != 0:
        inputs = layers.ZeroPadding2D(((0, padded), (0, padded)))(inputs)

    vv = layers.Dense(embed_dim, use_bias=False, name=name + "v")(inputs)
    # vv = einops.rearrange(vv, "D (h hk) (w wk) (H p) -> D h w H (hk wk) p", hk=kernel_size, wk=kernel_size, H=num_heads, p=key_dim)
    vv = functional.reshape(vv, (-1, hh, kernel_size, ww, kernel_size, num_heads, key_dim))  # [1, 14, 2, 14, 2, 6, 32]
    vv = functional.transpose(vv, [0, 1, 3, 5, 2, 4, 6])
    vv = functional.reshape(vv, [-1, hh, ww, num_heads, kernel_size * kernel_size, key_dim])  # [1, 14, 14, 6, 4, 32]

    # attn = layers.AvgPool2D(pool_size=3, strides=2, padding='SAME')(inputs)
    attn = layers.AvgPool2D(pool_size=kernel_size, strides=kernel_size)(inputs)
    attn = layers.Dense(kernel_size**4 * num_heads, use_bias=True, name=name + "attn")(attn) * qk_scale
    attn = functional.reshape(attn, [-1, hh, ww, num_heads, kernel_size * kernel_size, kernel_size * kernel_size])  # [1, 14, 14, 6, 4, 4]
    # attn = functional.softmax(attn, axis=-1)
    attn = layers.Softmax(axis=-1, name=name and name + "attention_scores")(attn)
    if attn_dropout > 0:
        attn = layers.Dropout(attn_dropout)(attn)

    # out = tf.matmul(attn, vv)  # [1, 14, 14, 6, 4, 32]
    out = attn @ vv
    # out = einops.rearrange(out, "D h w H (hk wk) p -> D (h hk) (w wk) (H p)", hk=kernel_size, wk=kernel_size)  # [1, 28, 28, 192]
    out = functional.reshape(out, [-1, hh, ww, num_heads, kernel_size, kernel_size, key_dim])  # [1, 14, 14, 6, 2, 2, 32]
    out = functional.transpose(out, [0, 1, 4, 2, 5, 3, 6])  # [1, 14, 2, 14, 2, 6, 32]
    out = functional.reshape(out, [-1, inputs.shape[1], inputs.shape[2], embed_dim])  # [1, 28, 28, 192]
    if padded != 0:
        out = out[:, :-padded, :-padded, :]
    out = layers.Dense(embed_dim, use_bias=True, name=name + "out")(out)

    return out


@register_keras_serializable(package="volo")
class BiasLayer(layers.Layer):
    def __init__(self, axis=-1, initializer="zeros", **kwargs):
        super(BiasLayer, self).__init__(**kwargs)
        self.axis, self.initializer = axis, initializer

    def build(self, input_shape):
        if self.axis == -1 or self.axis == len(input_shape) - 1:
            bb_shape = (input_shape[-1],)
        else:
            bb_shape = [1] * len(input_shape)
            axis = self.axis if isinstance(self.axis, (list, tuple)) else [self.axis]
            for ii in axis:
                bb_shape[ii] = input_shape[ii]
        self.bb = self.add_weight(name="bias", shape=bb_shape, initializer=self.initializer, trainable=True)
        super(BiasLayer, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs + self.bb

    def get_config(self):
        config = super(BiasLayer, self).get_config()
        config.update({"axis": self.axis})  # Not saving initializer in config
        return config

    def get_weights_channels_last(self):
        # channel_first -> channel_last
        weights = self.get_weights()
        if backend.image_data_format() != "channels_last" and self.axis == 1:
            weights = [np.squeeze(ii) for ii in weights]
        return weights

    def set_weights_channels_last(self, weights):
        # channel_last -> channel_first
        if backend.image_data_format() != "channels_last" and self.axis == 1:
            weights = [np.reshape(ii, self.bb.shape) for ii in weights]
        return self.set_weights(weights)


def attention_mlp_block(inputs, embed_dim, num_heads=1, mlp_ratio=3, attention_type=None, drop_rate=0, mlp_activation="gelu", dropout=0, name=""):
    # print(f">>>> {drop_rate = }")
    nn_0 = inputs[:, :1] if attention_type == "class" else inputs
    nn_1 = layers.LayerNormalization(epsilon=BATCH_NORM_EPSILON, name=name + "LN")(inputs)
    key_dim = embed_dim // num_heads

    if attention_type == "outlook":
        nn_1 = outlook_attention(nn_1, embed_dim, num_heads=num_heads, name=name + "attn_")
    elif attention_type == "outlook_simple":
        nn_1 = outlook_attention_simple(nn_1, embed_dim, num_heads=num_heads, name=name + "attn_")
    elif attention_type == "class":
        # nn_1 = class_attention(nn_1, embed_dim, num_heads=num_heads, name=name + "attn_")
        query = nn_1[:, :1, :]
        nn_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, output_shape=embed_dim, use_bias=False, name=name + "attn_mhsa")(query, nn_1)
        nn_1 = BiasLayer(name=name + "attn_bias")(nn_1)  # bias for output dense
    elif attention_type == "mhsa":
        # nn_1 = multi_head_self_attention(nn_1, num_heads=num_heads, key_dim=key_dim, out_shape=embed_dim, out_weight=True, out_bias=True, name=name + "attn_")
        nn_1 = layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, output_shape=embed_dim, use_bias=False, name=name + "attn_mhsa")(nn_1, nn_1)
        nn_1 = BiasLayer(name=name + "attn_bias")(nn_1)  # bias for output dense

    if drop_rate > 0:
        nn_1 = layers.Dropout(drop_rate, noise_shape=(None, 1, 1, 1), name=name + "drop_1")(nn_1)
    nn_1 = layers.Add()([nn_0, nn_1])

    """ MLP """
    nn_2 = layers.LayerNormalization(epsilon=BATCH_NORM_EPSILON, name=name + "mlp_LN")(nn_1)
    nn_2 = layers.Dense(embed_dim * mlp_ratio, name=name + "mlp_dense_1")(nn_2)
    # gelu with approximate=False using `erf` leads to GPU memory leak...
    # nn_2 = layers.Activation("gelu", name=name + "mlp_" + mlp_activation)(nn_2)
    nn_2 = activation_by_name(nn_2, mlp_activation, name=name + mlp_activation)
    nn_2 = layers.Dense(embed_dim, name=name + "mlp_dense_2")(nn_2)
    if dropout > 0:
        nn_2 = layers.Dropout(dropout)(nn_2)

    if drop_rate > 0:
        nn_2 = layers.Dropout(drop_rate, noise_shape=(None, 1, 1, 1), name=name + "drop_2")(nn_2)
    out = layers.Add(name=name + "output")([nn_1, nn_2])

    if attention_type == "class":
        out = functional.concat([out, inputs[:, 1:]], axis=1)
    return out


@register_keras_serializable(package="volo")
class PositionalEmbedding(layers.Layer):
    def __init__(self, input_height=-1, **kwargs):
        super().__init__(**kwargs)
        self.pp_init = initializers.TruncatedNormal(stddev=0.2)
        self.input_height = input_height

    def build(self, input_shape):
        self.pp = self.add_weight(name="positional_embedding", shape=(1, *input_shape[1:]), initializer=self.pp_init, trainable=True)
        super().build(input_shape)

        if len(input_shape) == 3:
            # height and width in input_shape reshaped as one, like [None, 16 * 16 + 1, 32]
            self.is_fused_height_width = True
            self.height = self.input_height if self.input_height > 0 else int(float(input_shape[1]) ** 0.5)
            self.width = input_shape[1] // self.height
        else:
            self.is_fused_height_width = False
            self.height, self.width = input_shape[1:3]

    def call(self, inputs, **kwargs):
        return inputs + self.pp

    def get_config(self):
        base_config = super().get_config()
        base_config.update({"input_height": self.input_height})
        return base_config

    def load_resized_weights(self, source_layer, method="bilinear"):
        # For input 224 --> [1, 14, 14, 384], convert to 384 --> [1, 24, 24, 384]
        if isinstance(source_layer, dict):
            source_pp = list(source_layer.values())[0]  # weights
        else:
            source_pp = source_layer.pp  # layer

        source_pp = np.array(source_pp).astype("float32")
        if self.is_fused_height_width:
            hh = ww = int(float(source_pp.shape[1]) ** 0.5)  # assume source weights are all square shape
            ss = source_pp[:, -hh * ww :]  # If has cls_token
            ss = ss.reshape([1, hh, ww, -1])

            tt = backend.numpy_image_resize(ss, target_shape=[self.height, self.width], method=method)
            tt = np.reshape(tt, [1, self.height * self.width, -1])
            tt = np.concatenate([source_pp[:, : -hh * ww], tt], axis=1)  # If has cls_token
        else:
            tt = backend.numpy_image_resize(source_pp, target_shape=[self.height, self.width], method=method)
        # functional.assign(self.pp, tt)  # For TF it's `parameter.assign(data)`, for Torch `parameter.data = torch.tensor(data)`
        self.set_weights([tt])

    def show_pos_emb(self, rows=16, base_size=1):
        import matplotlib.pyplot as plt

        ss = self.pp[0]
        cols = int(math.ceil(ss.shape[-1] / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(base_size * cols, base_size * rows))
        for id, ax in enumerate(axes.flatten()):
            ax.imshow(ss[:, :, id])
            ax.set_axis_off()
        fig.tight_layout()
        return fig


@register_keras_serializable(package="volo")
class ClassToken(layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.token_init = initializers.TruncatedNormal(stddev=0.2)

    def build(self, input_shape):
        self.class_tokens = self.add_weight(name="tokens", shape=(1, 1, input_shape[-1]), initializer=self.token_init, trainable=True)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        if backend.backend() == "pytorch":
            class_tokens = self.class_tokens.expand(inputs.shape[0], -1, -1)
        else:
            class_tokens = functional.repeat(self.class_tokens, functional.shape(inputs)[0], axis=0)
        return functional.concat([class_tokens, inputs], axis=1)

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[1] + 1, input_shape[2])


@register_keras_serializable(package="volo")
class MixupToken(layers.Layer):
    def __init__(self, scale=2, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.scale, self.beta = scale, beta

    def call(self, inputs, training=None, **kwargs):
        height, width = functional.shape(inputs)[1], functional.shape(inputs)[2]
        # tf.print("training:", training)
        def _call_train():
            return functional.stack(self.rand_bbox(height, width))

        def _call_test():
            return functional.cast(functional.stack([0, 0, 0, 0]), "int32")  # No mixup area for test

        return backend.in_train_phase(_call_train, _call_test, training=training)

    def sample_beta_distribution(self):
        import tensorflow as tf

        gamma_1_sample = tf.random.gamma(shape=[], alpha=self.beta)
        gamma_2_sample = tf.random.gamma(shape=[], alpha=self.beta)
        return gamma_1_sample / (gamma_1_sample + gamma_2_sample)

    def rand_bbox(self, height, width):
        import tensorflow as tf

        random_lam = functional.cast(self.sample_beta_distribution(), self.compute_dtype)
        cut_rate = functional.sqrt(1.0 - random_lam)
        s_height, s_width = height // self.scale, width // self.scale

        right_pos = tf.random.uniform(shape=[], minval=0, maxval=s_width, dtype="int32")
        bottom_pos = tf.random.uniform(shape=[], minval=0, maxval=s_height, dtype="int32")
        left_pos = right_pos - functional.cast(functional.cast(s_width, cut_rate.dtype) * cut_rate, "int32") // 2
        top_pos = bottom_pos - functional.cast(functional.cast(s_height, cut_rate.dtype) * cut_rate, "int32") // 2
        left_pos, top_pos = functional.maximum(left_pos, 0), functional.maximum(top_pos, 0)

        return left_pos, top_pos, right_pos, bottom_pos

    def do_mixup_token(self, inputs, bbox):
        left, top, right, bottom = bbox
        sub_ww = inputs[:, :, left:right]
        mix_sub = functional.concat([sub_ww[:, :top], sub_ww[::-1, top:bottom], sub_ww[:, bottom:]], axis=1)
        output = functional.concat([inputs[:, :, :left], mix_sub, inputs[:, :, right:]], axis=2)
        output.set_shape(inputs.shape)
        return output

    def get_config(self):
        config = super().get_config()
        config.update({"scale": self.scale, "beta": self.beta})
        return config


def patch_stem(inputs, hidden_dim=64, stem_width=384, patch_size=8, strides=2, activation="relu", name=""):
    nn = conv2d_no_bias(inputs, hidden_dim, 7, strides=strides, padding="same", name=name + "1_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "1_")
    nn = conv2d_no_bias(nn, hidden_dim, 3, strides=1, padding="same", name=name + "2_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "2_")
    nn = conv2d_no_bias(nn, hidden_dim, 3, strides=1, padding="same", name=name + "3_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "3_")

    patch_step = patch_size // strides
    return conv2d_no_bias(nn, stem_width, patch_step, strides=patch_step, use_bias=True, name=name + "patch_")


def VOLO(
    num_blocks,
    embed_dims,
    num_heads,
    mlp_ratios,
    stem_hidden_dim=64,
    patch_size=8,
    mlp_activation="gelu",
    input_shape=(224, 224, 3),
    num_classes=1000,
    drop_connect_rate=0,
    classfiers=2,
    mix_token=False,
    token_classifier_top=False,
    mean_classifier_top=False,
    token_label_top=False,
    first_attn_type="outlook",
    pretrained="imagenet",
    model_name="VOLO",
    kwargs=None,
):
    inputs = layers.Input(input_shape)

    """ forward_embeddings """
    nn = patch_stem(inputs, hidden_dim=stem_hidden_dim, stem_width=embed_dims[0], patch_size=patch_size, strides=2, name="stem_")

    if mix_token:
        scale = 2
        mixup_token = MixupToken(scale=scale)
        bbox = mixup_token(nn)
        nn = mixup_token.do_mixup_token(nn, bbox * scale)

    outlook_attentions = [True, False]
    downsamples = [True, False]

    """ forward_tokens """
    total_blocks = sum(num_blocks)
    global_block_id = 0

    # Outlook attentions
    num_block, embed_dim, num_head, mlp_ratio = num_blocks[0], embed_dims[0], num_heads[0], mlp_ratios[0]
    for ii in range(num_block):
        name = "outlook_block{}_".format(ii)
        block_drop_rate = drop_connect_rate * global_block_id / total_blocks
        nn = attention_mlp_block(nn, embed_dim, num_head, mlp_ratio, first_attn_type, block_drop_rate, mlp_activation, name=name)
        global_block_id += 1

    # downsample
    nn = layers.Conv2D(embed_dim * 2, kernel_size=2, strides=2, name="downsample_conv")(nn)
    # PositionalEmbedding
    nn = PositionalEmbedding(name="positional_embedding")(nn)

    # MHSA attentions
    num_block, embed_dim, num_head, mlp_ratio = num_blocks[1], embed_dims[1], num_heads[1], mlp_ratios[1]
    for ii in range(num_block):
        name = "MHSA_block{}_".format(ii)
        block_drop_rate = drop_connect_rate * global_block_id / total_blocks
        nn = attention_mlp_block(nn, embed_dim, num_head, mlp_ratio, "mhsa", block_drop_rate, mlp_activation, name=name)
        global_block_id += 1

    if num_classes == 0:
        model = models.Model(inputs, nn, name=model_name)
        reload_model_weights(model, PRETRAINED_DICT, "volo", pretrained, PositionalEmbedding)
        return model

    _, height, width, channel = nn.shape
    nn = functional.reshape(nn, (-1, height * width, channel))

    """ forward_cls """
    nn = ClassToken(name="class_token")(nn)

    embed_dim, num_head, mlp_ratio = embed_dims[-1], num_heads[-1], mlp_ratios[-1]
    for id in range(classfiers):
        name = "classfiers{}_".format(id)
        nn = attention_mlp_block(nn, embed_dim, num_head, mlp_ratio, "class", mlp_activation=mlp_activation, name=name)
    nn = layers.LayerNormalization(epsilon=BATCH_NORM_EPSILON, name="pre_out_LN")(nn)

    if token_label_top:
        # Training with label token
        nn_cls = layers.Dense(num_classes, dtype="float32", name="token_head")(nn[:, 0])
        nn_aux = layers.Dense(num_classes, dtype="float32", name="aux_head")(nn[:, 1:])

        if mix_token:
            nn_aux = functional.reshape(nn_aux, (-1, height, width, num_classes))
            nn_aux = mixup_token.do_mixup_token(nn_aux, bbox)
            nn_aux = layers.Reshape((height * width, num_classes), dtype="float32", name="aux")(nn_aux)

            left, top, right, bottom = bbox
            lam = 1 - ((right - left) * (bottom - top) / nn_aux.shape[1])
            lam_repeat = functional.expand_dims(functional.repeat(lam, functional.shape(inputs)[0], axis=0), 1)
            nn_cls = layers.Concatenate(axis=-1, dtype="float32", name="class")([nn_cls, lam_repeat])

        # nn_lam = layers.Lambda(lambda ii: functional.cast(functional.stack(ii), "float32"))([left_pos, top_pos, right_pos, bottom_pos])
        nn = [nn_cls, nn_aux]
    elif mean_classifier_top:
        # Return mean of all tokens
        nn = layers.GlobalAveragePooling1D(name="avg_pool")(nn)
        nn = layers.Dense(num_classes, dtype="float32", name="token_head")(nn)
    elif token_classifier_top:
        # Return dense classifier using only first token
        nn = layers.Dense(num_classes, dtype="float32", name="token_head")(nn[:, 0])
    else:
        # Return token dense for evaluation
        nn_cls = layers.Dense(num_classes, dtype="float32", name="token_head")(nn[:, 0])
        nn_aux = layers.Dense(num_classes, dtype="float32", name="aux_head")(nn[:, 1:])
        nn = layers.Add()([nn_cls, functional.reduce_max(nn_aux, 1) * 0.5])

    model = models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    reload_model_weights(model, PRETRAINED_DICT, "volo", pretrained, PositionalEmbedding)
    return model


def VOLO_d1(input_shape=(224, 224, 3), num_classes=1000, pretrained="imagenet", **kwargs):
    num_blocks = [4, 14]
    embed_dims = [192, 384]
    num_heads = [6, 12]
    mlp_ratios = [3, 3]
    stem_hidden_dim = 64
    return VOLO(**locals(), model_name="volo_d1", **kwargs)


def VOLO_d2(input_shape=(224, 224, 3), num_classes=1000, pretrained="imagenet", **kwargs):
    num_blocks = [6, 18]
    embed_dims = [256, 512]
    num_heads = [8, 16]
    mlp_ratios = [3, 3]
    stem_hidden_dim = 64
    return VOLO(**locals(), model_name="volo_d2", **kwargs)


def VOLO_d3(input_shape=(224, 224, 3), num_classes=1000, pretrained="imagenet", **kwargs):
    num_blocks = [8, 28]
    embed_dims = [256, 512]
    num_heads = [8, 16]
    mlp_ratios = [3, 3]
    stem_hidden_dim = 64
    return VOLO(**locals(), model_name="volo_d3", **kwargs)


def VOLO_d4(input_shape=(224, 224, 3), num_classes=1000, pretrained="imagenet", **kwargs):
    num_blocks = [8, 28]
    embed_dims = [384, 768]
    num_heads = [12, 16]
    mlp_ratios = [3, 3]
    stem_hidden_dim = 64
    return VOLO(**locals(), model_name="volo_d4", **kwargs)


def VOLO_d5(input_shape=(224, 224, 3), num_classes=1000, pretrained="imagenet", **kwargs):
    num_blocks = [12, 36]
    embed_dims = [384, 768]
    num_heads = [12, 16]
    mlp_ratios = [4, 4]
    stem_hidden_dim = 128
    return VOLO(**locals(), model_name="volo_d5", **kwargs)
