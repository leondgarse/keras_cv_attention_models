import numpy as np
from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models, functional, image_data_format, initializers
from keras_cv_attention_models.models import register_model
from keras_cv_attention_models.attention_layers import (
    activation_by_name,
    BiasLayer,
    ChannelAffine,
    ClassToken,
    drop_block,
    drop_connect_rates_split,
    PositionalEmbedding,
    PositionalIndex,
    PositionalEncodingFourierRot1D,
    CausalMask,
    add_pre_post_process,
    RMSNorm,
)
from keras_cv_attention_models.download_and_load import reload_model_weights


PRETRAINED_DICT = {
    "beit_base_patch16": {"imagenet21k-ft1k": {224: "d7102337a13a3983f3b6470de77b5d5c", 384: "76353026477c60f8fdcbcc749fea17b3"}},
    "beit_v2_base_patch16": {"imagenet21k-ft1k": {224: "d001dcb67cdda16bfdbb2873ab9b13c8"}},
    "beit_large_patch16": {
        "imagenet21k-ft1k": {224: "fce2d162e7fa4dba9a1b1fc5e1dec5ce", 384: "158934d07dd8b1e1c6b96883aa00a748", 512: "64d18088e91df243960e5830aab80a6e"}
    },
    "beit_v2_large_patch16": {"imagenet21k-ft1k": {224: "b3cee12a545bfb676f9f426ee7158d27"}},
    "dinov2_vit_base14": {"imagenet": {518: "8943697691a511fa472e853c39666d97"}},
    "dinov2_vit_large14": {"imagenet": {518: "53deb035702a60948688adc858b01af6"}},
    "dinov2_vit_small14": {"imagenet": {518: "9f823ded0e64beccbd4e19a9fd2b501f"}},
    "dinov2_vit_giant14": {"imagenet": {518: "12a7cb1f167acb1036d9bfb4e25e796d"}},
    "eva_giant_patch14": {
        "imagenet21k-ft1k": {224: "5a475db6696d6e36ea896ec5dbd1c20d", 336: "fd8eeec10d6b6cb607ce033ea85b8e80", 560: "0ef0d2961523fb2047fbdb59cc347c17"}
    },
    "eva_large_patch14": {"imagenet21k-ft1k": {196: "bbeea886fbde4bd1c8c9876345273a99", 336: "4928faafd0177fe8f0d02dab4abc8e83"}},
    "eva02_base_patch14": {"mim_in22k_ft22k_ft1k": {448: "fd8cb2e335201fd1925370c78bfb68b0"}},
    "eva02_large_patch14": {"mim_m38m_ft22k_ft1k": {448: "8a5ba7c06e3f0a2e4c985982b2d93671"}},
    "eva02_small_patch14": {"mim_in22k_ft1k": {336: "87d87900e8096c3734278aab44c4b5f4"}},
    "eva02_tiny_patch14": {"mim_in22k_ft1k": {336: "8243463316967ca8f6ee0d0abcc4d236"}},
    "flexivit_small": {"imagenet": {240: "efb73a97d099a491b69ebfaf8a337df8"}},
    "flexivit_base": {"imagenet": {240: "dac627debb194928db01e1b9b7a548fd"}},
    "flexivit_large": {"imagenet": {240: "6faa953227d2ef1df6758f8eb7234490"}},
    "meta_transformer_base_patch16": {"laion_2b": {384: "5daafcdef0895ab292b39173331c12c3"}},
    "meta_transformer_large_patch14": {"laion_2b": {336: "f3a4444bf823ccbaab6e586d9915ffb1"}},
    "vit_text_large_patch14": {"clip": "ebeaed60ecd6685c5aeaba117f1b1737"},
    "vit5_small_patch16": {"imagenet": {224: "f60778dbc03efc0455c2efd197164368"}},
    "vit5_base_patch16": {"imagenet": {224: "5f6b4958b86cabaa5f7c3cb37ee9cc65", 384: "6622efcd8fe321f8321b6cee4e3aa069"}},
    "vit5_large_patch16": {"imagenet": {224: "08f35247ba56768d403e4c58f26f6771"}},
}


@backend.register_keras_serializable(package="beit")
class PositionalEncodingFourierRot(layers.Layer):
    def __init__(
        self,
        with_cls_token=True,
        num_reg_tokens=0,
        attn_height=-1,
        num_heads=-1,
        temperature=1e4,
        reg_temperature=100,
        ref_feature_shape=16,
        reg_ref_feature_shape=14,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.with_cls_token, self.num_reg_tokens = with_cls_token, num_reg_tokens
        self.num_cls_tokens = 1 if with_cls_token else 0
        self.attn_height, self.num_heads = attn_height, num_heads
        self.temperature, self.reg_temperature = float(temperature), float(reg_temperature)
        self.ref_feature_shape, self.reg_ref_feature_shape = ref_feature_shape, reg_ref_feature_shape

        if hasattr(self, "register_buffer"):  # PyTorch
            self.register_buffer("pos_sin", None, persistent=False)
            self.register_buffer("pos_cos", None, persistent=False)
            if self.num_reg_tokens > 0:
                self.register_buffer("reg_pos_sin", None, persistent=False)
                self.register_buffer("reg_pos_cos", None, persistent=False)

    def _get_sin_cos_pos(self, height, width, ref_feature_shape, temperature):
        hh, ww = np.arange(height, dtype="float32"), np.arange(width, dtype="float32")
        if ref_feature_shape is not None and ref_feature_shape > 0:
            hh = hh / height * ref_feature_shape
            ww = ww / width * ref_feature_shape

        actual_num_heads = self.num_heads if self.num_heads > 0 else 1
        dim = self.channels // actual_num_heads // 2
        freqs = 1.0 / (temperature ** (np.arange(0, dim, 2).astype("float32") / dim))

        freqs_h = np.repeat(hh[:, None] * freqs[None, :], 2, axis=-1)
        freqs_w = np.repeat(ww[:, None] * freqs[None, :], 2, axis=-1)

        hh_grid, ww_grid = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
        grid = np.concatenate([freqs_h[hh_grid], freqs_w[ww_grid]], axis=-1)
        grid = grid.reshape([1, height * width, -1])

        pos_sin, pos_cos = np.tile(np.sin(grid), [1, 1, actual_num_heads]), np.tile(np.cos(grid), [1, 1, actual_num_heads])
        # print(f"{pos_sin.shape = }, {pos_cos.shape = }, {height = }, {width = }, {channels = }")
        return functional.convert_to_tensor(pos_sin, dtype=self.compute_dtype), functional.convert_to_tensor(pos_cos, dtype=self.compute_dtype)

    def build(self, input_shape):
        # input (with_cls_token=True): `[batch, ..., attn_height * attn_width + class_token, channels]`.
        # input (with_cls_token=False): `[batch, ..., attn_height * attn_width, channels]`.
        # print(input_shape)
        blocks, self.channels = input_shape[-2], input_shape[-1]
        num_patches = blocks - self.num_cls_tokens - self.num_reg_tokens
        self.blocks_shape = [*input_shape[1:-2], num_patches]
        # print(f"{num_patches = }, {self.num_cls_tokens = }, {self.num_reg_tokens = }, {blocks = }, {self.channels = }")

        if self.attn_height == -1:
            height = width = int(float(num_patches) ** 0.5)
        else:
            height = self.attn_height
            width = int(float(num_patches) / height)
        self.pos_sin, self.pos_cos = self._get_sin_cos_pos(height, width, self.ref_feature_shape, self.temperature)

        if self.num_reg_tokens > 0:
            self.reg_blocks_shape = [*input_shape[1:-2], self.num_reg_tokens]
            reg_height = reg_width = int(self.num_reg_tokens**0.5)
            self.reg_pos_sin, self.reg_pos_cos = self._get_sin_cos_pos(reg_height, reg_width, self.reg_ref_feature_shape, self.reg_temperature)
        super().build(input_shape)

    def _rotate_half(self, inputs, block_shape):
        left, right = functional.split(functional.reshape(inputs, [-1, *block_shape, self.channels // 2, 2]), 2, axis=-1)
        return functional.reshape(functional.concat([-right, left], axis=-1), (-1, *block_shape, self.channels))

    def call(self, inputs, **kwargs):
        out = []
        if self.with_cls_token:
            cls_token, inputs = functional.split(inputs, [self.num_cls_tokens, -1], axis=-2)  # `[batch, num_heads, cls_tokens + attn_h * attn_w, channels]`
            out.append(cls_token)
        if self.num_reg_tokens > 0:
            inputs, registers = functional.split(inputs, [-1, self.num_reg_tokens], axis=-2)  # `[batch, num_heads, attn_h * attn_w + reg_tokens, channels]`

        patches = inputs * self.pos_cos + self._rotate_half(inputs, self.blocks_shape) * self.pos_sin
        out.append(patches)

        if self.num_reg_tokens > 0:  # From ViT-5
            registers = registers * self.reg_pos_cos + self._rotate_half(registers, self.reg_blocks_shape) * self.reg_pos_sin
            out.append(registers)
        return functional.concat(out, axis=-2)

    def get_config(self):
        base_config = super().get_config()
        base_config.update(
            {
                "with_cls_token": self.with_cls_token,
                "num_reg_tokens": self.num_reg_tokens,
                "attn_height": self.attn_height,
                "num_heads": self.num_heads,
                "temperature": self.temperature,
                "reg_temperature": self.reg_temperature,
                "ref_feature_shape": self.ref_feature_shape,
                "reg_ref_feature_shape": self.reg_ref_feature_shape,
            }
        )
        return base_config


@backend.register_keras_serializable(package="beit")
class MultiHeadRelativePositionalEmbedding(layers.Layer):
    def __init__(self, with_cls_token=True, attn_height=-1, num_heads=-1, **kwargs):
        super().__init__(**kwargs)
        self.with_cls_token, self.attn_height, self.num_heads = with_cls_token, attn_height, num_heads
        if with_cls_token:
            self.cls_token_len = 1
            self.cls_token_pos_len = 3
        else:
            self.cls_token_len = 0
            self.cls_token_pos_len = 0

    def build(self, attn_shape):
        # input (with_cls_token=True): `[batch, num_heads, attn_blocks, attn_blocks]`. where `attn_blocks = attn_height * attn_width + class_token`
        # input (with_cls_token=False): `[batch, num_heads, attn_blocks, attn_blocks]`. where `attn_blocks = attn_height * attn_width`
        # print(attn_shape)
        if self.attn_height == -1:
            height = width = int(float(attn_shape[2] - self.cls_token_len) ** 0.5)  # hh == ww, e.g. 14
        else:
            height = self.attn_height
            width = int(float(attn_shape[2] - self.cls_token_len) / height)
        num_heads = attn_shape[1] if self.num_heads == -1 else self.num_heads
        num_relative_distance = (2 * height - 1) * (2 * width - 1) + self.cls_token_pos_len
        # pos_shape = (num_relative_distance, num_heads)
        pos_shape = (num_heads, num_relative_distance)
        # initializer = tf.random_normal_initializer(stddev=0.02)
        self.positional_embedding = self.add_weight(name="positional_embedding", shape=pos_shape, initializer="zeros", trainable=True)

        hh, ww = np.meshgrid(range(height), range(width))  # tf.meshgrid is same with np.meshgrid 'xy' mode, while torch.meshgrid 'ij' mode
        coords = np.stack([hh, ww], axis=-1)  # [14, 14, 2]
        coords_flatten = np.reshape(coords, [-1, 2])  # [196, 2]
        relative_coords = coords_flatten[:, None, :] - coords_flatten[None, :, :]  # [196, 196, 2]
        relative_coords_hh = relative_coords[:, :, 0] + height - 1
        relative_coords_ww = (relative_coords[:, :, 1] + width - 1) * (2 * height - 1)
        relative_coords = np.stack([relative_coords_hh, relative_coords_ww], axis=-1)

        relative_position_index = np.sum(relative_coords, axis=-1).astype("float32")  # [196, 196]
        if attn_shape[3] != attn_shape[2]:
            # Choose the small values if value_block != query_block
            relative_position_index = relative_position_index[:, -(attn_shape[3] - self.cls_token_len) :]

        if self.with_cls_token:
            top = np.ones((1, relative_position_index.shape[1]), dtype=relative_position_index.dtype) * (num_relative_distance - 3)
            left = np.ones((relative_position_index.shape[0], 1), dtype=relative_position_index.dtype) * (num_relative_distance - 2)
            corner = np.ones((1, 1), dtype=relative_position_index.dtype) * (num_relative_distance - 1)
            # print(f">>>> {top.shape = }, {left.shape = }, {corner.shape = }")
            # >>>> top.shape = TensorShape([1, 196]), left.shape = TensorShape([196, 1]), corner.shape = TensorShape([1, 1])
            left_corner = np.concatenate([corner, left], axis=0)
            relative_position_index = np.concatenate([top, relative_position_index], axis=0)
            relative_position_index = np.concatenate([left_corner, relative_position_index], axis=1)  # [197, 197]
        if hasattr(self, "register_buffer"):  # PyTorch
            self.register_buffer("relative_position_index", functional.convert_to_tensor(relative_position_index, dtype="int64"), persistent=False)
        else:
            self.relative_position_index = functional.convert_to_tensor(relative_position_index, dtype="int64")
        super().build(attn_shape)

    def call(self, inputs, **kwargs):
        pos_emb = functional.gather(self.positional_embedding, self.relative_position_index[: inputs.shape[2], : inputs.shape[3]], axis=1)
        # tf.print(pos_emb.shape, inputs.shape)
        return inputs + pos_emb

    def get_config(self):
        base_config = super().get_config()
        base_config.update({"with_cls_token": self.with_cls_token, "attn_height": self.attn_height, "num_heads": self.num_heads})
        return base_config

    def load_resized_weights(self, source_layer, method="bilinear"):
        if isinstance(source_layer, dict):
            source_tt = list(source_layer.values())[0]  # weights
            # source_tt = source_layer["pos_emb:0"]  # weights
        else:
            source_tt = source_layer.get_weights()[0]  # layer
        source_tt = np.array(source_tt.detach() if hasattr(source_tt, "detach") else source_tt).astype("float32")
        # self.positional_embedding.assign(tf.transpose(source_tt))
        hh = ww = int(float(source_tt.shape[1] - self.cls_token_pos_len) ** 0.5)  # assume source weights are all square shape
        num_heads = source_tt.shape[0]
        ss = source_tt[:, : hh * ww].reshape((num_heads, hh, ww))  # [num_heads, hh, ww]

        if self.attn_height == -1:
            target_hh = target_ww = int(float(self.positional_embedding.shape[1] - self.cls_token_pos_len) ** 0.5)
        else:
            target_hh = 2 * self.attn_height - 1
            target_ww = int(float(self.positional_embedding.shape[1] - self.cls_token_pos_len) / target_hh)

        tt = backend.numpy_image_resize(ss, target_shape=[target_hh, target_ww], method=method, is_source_channels_last=False)
        tt = tt.reshape((num_heads, tt.shape[1] * tt.shape[2]))  # [num_heads, target_hh * target_ww]
        if self.with_cls_token:
            tt = np.concatenate([tt, source_tt[:, -self.cls_token_pos_len :]], axis=1)
        self.set_weights([tt])

    def show_pos_emb(self, rows=1, base_size=2):
        import math
        import matplotlib.pyplot as plt

        num_heads = self.positional_embedding.shape[0]
        # pos_emb = tf.gather(self.positional_embedding, self.relative_position_index, axis=1).numpy()
        hh = ww = int(float(self.positional_embedding.shape[1] - self.cls_token_pos_len) ** 0.5)
        pos_emb = self.positional_embedding[:, : hh * ww]
        pos_emb = pos_emb.detach() if hasattr(pos_emb, "detach") else pos_emb
        pos_emb = pos_emb.numpy() if hasattr(pos_emb, "numpy") else np.array(pos_emb)
        pos_emb = pos_emb.reshape((num_heads, hh, ww))
        cols = int(math.ceil(num_heads / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(base_size * cols, base_size * rows))
        for id, ax in enumerate(axes.flatten()):
            if id >= num_heads:
                break
            ax.imshow(pos_emb[id])
            ax.set_axis_off()
        fig.tight_layout()
        return fig


def scaled_dot_product_attention(query, key, value, output_shape, pos_emb=None, out_weight=True, out_bias=False, qk_scale=-1, dropout=0, name=None):
    output_dim = output_shape[-1]
    blocks = output_shape[1:-1] if output_shape[0] is None or output_shape[0] < 1 else output_shape[:-1]
    # query, value: [batch, num_heads, blocks, key_dim], key: [batch, num_heads, key_dim, blocks]
    qk_scale = qk_scale if qk_scale > 0 else (1.0 / (float(query.shape[-1]) ** 0.5))
    # print(f"{query.shape = }, {key.shape = }")
    # attention_scores = layers.Lambda(lambda xx: functional.matmul(xx[0], xx[1]))([query, key]) * qk_scale  # [batch, num_heads, q_blocks, k_blocks]
    attention_scores = query @ key
    if qk_scale != 1:
        attention_scores = attention_scores * qk_scale
    # print(f"{attention_scores.shape = }")
    if pos_emb is not None:
        # attention_scores = MultiHeadPositionalEmbedding(query_height=height, name=name and name + "attn_pos")(attention_scores)
        attention_scores = pos_emb(attention_scores)
    attention_scores = layers.Softmax(axis=-1, name=name and name + "attention_scores")(attention_scores)
    if dropout > 0:
        attention_scores = layers.Dropout(dropout, name=name and name + "attn_drop")(attention_scores)

    # output = functional.matmul(attention_scores, value)    # [batch, num_heads, q_blocks, key_dim * attn_ratio]
    # output = layers.Lambda(lambda xx: functional.matmul(xx[0], xx[1]))([attention_scores, value])
    attention_output = attention_scores @ value
    output = functional.transpose(attention_output, [0, 2, 1, 3])  # [batch, q_blocks, num_heads, key_dim * attn_ratio]
    if -1 in blocks:
        output = layers.Reshape([*blocks, int(np.prod(output.shape[2:]))])(output)
    else:
        output = functional.reshape(output, [-1, *blocks, np.prod(output.shape[1:]) // np.prod(blocks)])  # [batch, q_blocks, channel * attn_ratio]
        # output = layers.Reshape([*blocks, np.prod(output.shape[1:]) // np.prod(blocks)])(output)

    if out_weight:
        # [batch, hh, ww, num_heads * key_dim] * [num_heads * key_dim, out] --> [batch, hh, ww, out]
        output_dim = output_dim if output_dim > 0 else value.shape[-1]
        output = layers.Dense(output_dim, use_bias=out_bias, name=name and name + "output")(output)
    return output


def qkv_to_multi_head_channels_last_format(query, key, value, num_heads, data_format=None):
    data_format = image_data_format() if data_format is None else data_format
    # print(f">>>> {query.shape = }, {key.shape = }, {value.shape = }, {num_heads = }, {data_format = }")
    if data_format == "channels_last":
        if query is not None:
            # query [batch, hh, ww, channel] -> [batch, num_heads, hh * ww, key_dim]
            query = layers.Reshape([-1, num_heads, query.shape[-1] // num_heads])(query)
            query = functional.transpose(query, [0, 2, 1, 3])
        if key is not None:
            # key [batch, hh, ww, channel] -> [batch, num_heads, key_dim, hh * ww]
            key = layers.Reshape([-1, num_heads, key.shape[-1] // num_heads])(key)
            key = functional.transpose(key, [0, 2, 3, 1])
        if value is not None:
            # value [batch, hh, ww, channel] -> [batch, num_heads, hh * ww, vv_dim]
            value = layers.Reshape([-1, num_heads, value.shape[-1] // num_heads])(value)
            value = functional.transpose(value, [0, 2, 1, 3])
    else:
        if query is not None:
            # query [batch, channel, hh, ww] -> [batch, num_heads, hh * ww, key_dim]
            query = layers.Reshape([num_heads, query.shape[1] // num_heads, -1])(query)
            query = functional.transpose(query, [0, 1, 3, 2])
        if key is not None:
            # key [batch, channel, hh, ww] -> [batch, num_heads, key_dim, hh * ww]
            key = layers.Reshape([num_heads, key.shape[1] // num_heads, -1])(key)
        if value is not None:
            # value [batch, channel, hh, ww] -> [batch, num_heads, hh * ww, vv_dim]
            value = layers.Reshape([num_heads, value.shape[1] // num_heads, -1])(value)
            value = functional.transpose(value, [0, 1, 3, 2])
    # print(f">>>> {query.shape = }, {key.shape = }, {value.shape = }, {num_heads = }, {data_format = }")
    return query, key, value


def attention_block(
    inputs,
    num_heads=4,
    key_dim=0,
    qkv_bias=False,
    qv_bias=False,
    out_weight=True,
    out_bias=False,
    use_qk_norm=False,
    use_rot_pos_emb=False,
    with_cls_token=True,
    num_reg_tokens=0,
    attn_height=-1,
    shared_pos_emb=None,
    use_pos_emb=False,
    text_max_block_size=0,  # Also a mark if this is a text inputs
    attn_dropout=0,
    epsilon=1e-6,
    temperature=1e4,
    ref_feature_shape=16,
    reg_ref_feature_shape=14,
    name=None,
):
    _, bb, cc = inputs.shape
    key_dim = key_dim if key_dim > 0 else cc // num_heads
    emded_dim = int(num_heads * key_dim)
    is_text_inputs = text_max_block_size > 0
    # print(f">>>> {bb = }, {cc = }, {emded_dim = }")

    # if qkv_bias, just use bias in qkv_dense, and set qv_bias False
    qkv_bias, qv_bias = (True, False) if qkv_bias else (False, qv_bias)

    # qv_bias = False
    # query = layers.Dense(emded_dim, use_bias=True, name=name and name + "query")(inputs)  # For loading eva02 base and large
    # key = layers.Dense(emded_dim, use_bias=False, name=name and name + "key")(inputs)  # For loading eva02 base and large
    # value = layers.Dense(emded_dim, use_bias=True, name=name and name + "value")(inputs)  # For loading eva02 base and large
    qkv = layers.Dense(emded_dim * 3, use_bias=qkv_bias, name=name and name + "qkv")(inputs)
    # qkv = functional.reshape(qkv, [-1, bb, qkv.shape[-1]])
    query, key, value = functional.split(qkv, 3, axis=-1)
    # query = [batch, num_heads, cls_token + hh * ww, key_dim]
    # print(f">>>> {query.shape = }, {key.shape = }, {value.shape = }")
    if qv_bias:
        query = BiasLayer(name=name + "query_bias")(query)
        value = BiasLayer(name=name + "value_bias")(value)

    if use_qk_norm:
        query = functional.reshape(query, [-1, bb, num_heads, key_dim])
        key = functional.reshape(key, [-1, bb, num_heads, key_dim])
        query = RMSNorm(epsilon=epsilon, axis=-1, name=name + "q_rmsnorm")(query)
        key = RMSNorm(epsilon=epsilon, axis=-1, name=name + "k_rmsnorm")(key)
        query = functional.reshape(query, [-1, bb, cc])
        key = functional.reshape(key, [-1, bb, cc])

    if use_rot_pos_emb and is_text_inputs:
        rope = PositionalEncodingFourierRot1D(max_block_size=text_max_block_size, name=name + "rope")
        query = rope(layers.Reshape([-1, num_heads, key_dim // 2, 2])(query))
        key = rope(layers.Reshape([-1, num_heads, key_dim // 2, 2])(key))
    elif use_rot_pos_emb:
        # Create a new one every time, as there's no weights for this layer
        rope = PositionalEncodingFourierRot(
            with_cls_token=with_cls_token,
            num_reg_tokens=num_reg_tokens,
            attn_height=attn_height,
            num_heads=num_heads,
            temperature=temperature,
            ref_feature_shape=ref_feature_shape,
            reg_ref_feature_shape=reg_ref_feature_shape,
            name=name + "rope",
        )
        query, key = rope(query), rope(key)

    query = functional.transpose(layers.Reshape([-1, num_heads, key_dim])(query), [0, 2, 1, 3])
    key = functional.transpose(layers.Reshape([-1, num_heads, key_dim])(key), [0, 2, 3, 1])
    value = functional.transpose(layers.Reshape([-1, num_heads, key_dim])(value), [0, 2, 1, 3])

    if is_text_inputs:
        pos_emb = CausalMask(block_size=text_max_block_size)
    elif shared_pos_emb:
        pos_emb = shared_pos_emb
    elif use_pos_emb:
        pos_emb = MultiHeadRelativePositionalEmbedding(attn_height=attn_height, name=name and name + "pos_emb")
    else:
        pos_emb = None
    output_shape = [-1, -1, emded_dim]
    return scaled_dot_product_attention(query, key, value, output_shape, pos_emb, out_weight, out_bias, dropout=attn_dropout, name=name)


def mlp_block(inputs, mlp_ratio=4, is_gated=False, use_norm=False, epsilon=1e-6, activation="gelu", name=""):
    input_channels = inputs.shape[-1]
    if is_gated:
        nn = layers.Dense(int(input_channels * mlp_ratio) * 2, name=name + "dense_gate")(inputs)
        gate, nn = functional.split(nn, 2, axis=-1)
        # nn = layers.Dense(int(input_channels * mlp_ratio), name=name + "dense_1")(inputs)  # For loading eva02 base and large
        # gate = layers.Dense(int(input_channels * mlp_ratio), name=name + "dense_gate")(inputs)  # For loading eva02 base and large
        gate = activation_by_name(gate, activation=activation, name=name + "gate_")
        nn = gate * nn
    else:
        nn = layers.Dense(int(input_channels * mlp_ratio), name=name + "dense_1")(inputs)
        nn = activation_by_name(nn, activation, name=name)
    if use_norm:
        nn = layers.LayerNormalization(axis=-1, epsilon=epsilon, name=name + "scale_ln")(nn)
    nn = layers.Dense(input_channels, name=name + "dense_2")(nn)
    return nn


def attention_mlp_block(
    inputs,
    layer_scale=0.1,
    mlp_ratio=4,
    use_gated_mlp=False,
    use_norm_mlp=False,
    use_rms_norm=False,
    drop_rate=0,
    epsilon=1e-6,
    activation="gelu",
    attn_params={},
    name="",
):
    # print(f">>>> {inputs.shape = }, {drop_rate = }")
    norm_layer = RMSNorm if use_rms_norm else layers.LayerNormalization
    nn = norm_layer(axis=-1, epsilon=epsilon, name=name + "attn_ln")(inputs)  # "channels_first" also using axis=-1
    nn = attention_block(nn, **attn_params, epsilon=epsilon, name=name + "attn_")
    nn = ChannelAffine(use_bias=False, weight_init_value=layer_scale, name=name + "attn_gamma")(nn) if layer_scale > 0 else nn
    nn = drop_block(nn, drop_rate)
    attn_out = layers.Add(name=name + "attn_out")([inputs, nn])

    """ MLP """
    nn = norm_layer(axis=-1, epsilon=epsilon, name=name + "mlp_ln")(attn_out)  # "channels_first" also using axis=-1
    nn = mlp_block(nn, mlp_ratio=mlp_ratio, is_gated=use_gated_mlp, use_norm=use_norm_mlp, epsilon=epsilon, activation=activation, name=name + "mlp_")
    nn = ChannelAffine(use_bias=False, weight_init_value=layer_scale, name=name + "mlp_gamma")(nn) if layer_scale > 0 else nn
    nn = drop_block(nn, drop_rate)
    # print(f">>>> {attn_out.shape = }, {nn.shape = }")
    nn = layers.Add(name=name + "mlp_output")([attn_out, nn])
    return nn


def patch_merging(inputs, num_tokens_out=8, use_cls_token=True, epsilon=1e-6, name=None):
    if use_cls_token:
        cls_token, inputs = functional.split(inputs, [1, -1], axis=1)
    input_channels = inputs.shape[-1]  # inputs: [batch, input_blocks, channels]
    scale = float(input_channels) ** -0.5
    pre_norm = layers.LayerNormalization(axis=-1, center=False, epsilon=epsilon, name=name)(inputs)
    nn = layers.Dense(num_tokens_out, use_bias=False, name=name and name + "queries")(pre_norm)  # -> [batch, input_blocks, num_tokens_out]
    nn = functional.transpose(nn, [0, 2, 1])  # -> [batch, num_tokens_out, input_blocks]
    nn = functional.softmax(nn, axis=-1)
    out = nn @ pre_norm  # -> [batch, num_tokens_out, channels]
    if use_cls_token:
        out = functional.concat([cls_token, out], axis=1)
    return out


@backend.register_keras_serializable(package="beit")
class HeadInitializer(initializers.Initializer):
    def __init__(self, stddev=0.02, scale=0.001, **kwargs):
        super().__init__(**kwargs)
        self.stddev, self.scale = stddev, scale

    def __call__(self, shape, dtype="float32"):
        return initializers.TruncatedNormal(stddev=self.stddev)(shape, dtype=dtype) * self.scale

    def get_config(self):
        base_config = super().get_config()
        base_config.update({"stddev": self.stddev, "scale": self.scale})
        return base_config


@backend.register_keras_serializable(package="beit")
class PatchConv2DWithResampleWeights(layers.Conv2D):
    def load_resized_weights(self, source_layer, method="bilinear"):
        import numpy as np

        # print("source_layer:", {kk: vv.shape for kk, vv in source_layer.items()})
        if isinstance(source_layer, dict):
            source_kernel, source_bias = list(source_layer.values())  # weights
        else:
            source_kernel, source_bias = source_layer.get_weights()  # layer

        # From FlexiViT https://github.com/google-research/big_vision/blob/main/big_vision/models/proj/flexi/vit.py#L30
        # Paper [PDF 2212.08013 FlexiViT: One Model for All Patch Sizes](https://arxiv.org/pdf/2212.08013.pdf)

        # channels_last source_kernel shape `[patch_size, patch_size, in_channel, out_channel]`
        # channels_first source_kernel shape `[out_channel, in_channel, patch_size, patch_size]`
        source_kernel, source_bias = np.array(source_kernel).astype("float32"), np.array(source_bias).astype("float32")
        source_shape, target_shape = source_kernel.shape[:2], self.kernel_size  # source_kernel is from h5, must be channels_last format
        # print(f"{source_shape = }, {target_shape = }")

        # get_resize_mat(old_shape, target_shape)
        # NOTE: we are using tf.image.resize here to match the resize operations in
        # the data preprocessing pipeline.
        mat = []
        for idx in range(source_shape[0] * source_shape[1]):
            basis_vec = np.zeros(source_shape).astype("float32")
            basis_vec[np.unravel_index(idx, source_shape)] = 1.0
            vec = np.expand_dims(basis_vec, -1 if image_data_format() == "channels_last" else 0)
            vec = functional.resize(vec, target_shape, method=method)
            vec = vec.detach() if hasattr(vec, "detach") else vec
            vec = vec.numpy() if hasattr(vec, "numpy") else np.array(vec)
            mat.append(vec.reshape(-1))
        resize_mat_pinv = np.linalg.pinv(np.stack(mat))

        # v_resample_kernel = jax.vmap(jax.vmap(lambda kernel: (resize_mat_pinv @ kernel.reshape(-1)).reshape(new_hw), 2, 2), 3, 3)
        # cc = v_resample_kernel(old)
        # As it's only one weight, just using two loop here, instead of `jax.vmap`
        target_weights = np.stack([[(resize_mat_pinv @ jj.reshape(-1)).reshape(target_shape) for jj in ii] for ii in source_kernel.transpose([3, 2, 0, 1])])
        if image_data_format() == "channels_last":
            target_weights = target_weights.transpose([2, 3, 1, 0])
        self.set_weights([target_weights, source_bias])


def Beit(
    depth=12,
    embed_dim=768,
    num_heads=12,
    patch_size=16,
    use_patch_bias=True,  # False for MetaTransFormer, True for others
    use_pre_norm=False,  # True for MetaTransFormer, False for others
    use_mask_inputs=False,  # For BeitV2 training model https://github.com/microsoft/unilm/blob/master/beit2/modeling_pretrain.py#L126
    patch_merging_block_id=-1,  # >=0 value to enable. https://arxiv.org/abs/2202.12015, https://github.com/conceptofmind/ViT-Patch-Merger
    patch_merging_num_tokens=8,  # Should better be a square number, expecially if use_rot_pos_emb=True
    with_cls_token=True,  # [Register tokens] https://arxiv.org/abs/2309.16588. Default 1 for cls_token, 0 for disable.
    num_reg_tokens=0,  # [Register tokens] tokens added at the end of sequence, [CLS, patches, registers]
    reg_temperature=None,  # [Register tokens] temperature for RoPE on registers
    attn_key_dim=0,  # [Attention args]
    attn_qv_bias=True,  # Default False for Vit, True for Beit, if True and attn_qkv_bias being False, will add BiasLayer for query and key.
    attn_qkv_bias=False,  # True for Vit, False for Beit, if True, will just use bias in qkv_dense, and set qv_bias False.
    use_qk_norm=False,  # [Attention args] True for ViT-5, apply RMSNorm on query and key
    attn_out_weight=True,
    attn_out_bias=True,
    attn_dropout=0,
    use_abs_pos_emb=False,  # [Pos emb args] True for Vit, False for Beit, whether use abcolute positional embedding or relative one in attention blocks
    use_abs_pos_emb_on_cls_token=True,  # [Pos emb args] False for FlexiViT, no_embed_class in timm. If use_abs_pos_emb is True, whether apply pos_emb on cls_token.
    use_rot_pos_emb=False,  # [Pos emb args] True for EVA02, False for others
    use_shared_pos_emb_for_attn=False,  # [Pos emb args] True for beit raw model without any finetune
    mlp_ratio=4,  # [MLP args]
    use_gated_mlp=False,  # [MLP args] True for DINOv2 and EVA02
    use_norm_mlp=False,  # [MLP args] True for EVA02 base and large, False for others.
    use_rms_norm=False,  # [MLP args] True for ViT-5, use RMSNorm instead of LayerNormalization
    include_top=True,  # [Head args] boolean value if include header and top output Dense layer. False for a LayerNorm layer only
    use_mean_pooling_head=True,  # [Head args] False for Vit, True for Beit, whether use use mean output or `class_token` output
    use_cat_head=False,  # [Head args] True for DINOv2
    vocab_size=0,  # [Text model] Set value > 0 for building text model. Will also set num_classes = vocab_size if include_top is True
    max_block_size=77,  # [Text model] max block size, works only if vocab_size > 0
    text_positional_dropout=0,  # [Text model] dropout for text model embedding layers
    text_use_positional_embedding=True,  # [Text model] boolean value if use Embedding positional layer after inputs
    input_shape=(224, 224, 3),  # [Common args] Not taking effect for text model
    num_classes=1000,  # For text model, equals to vocab_size if include_top is True
    layer_norm_epsilon=1e-6,  # 1e-5 for ViT clip models, 1e-6 for others
    activation="gelu",
    layer_scale=0.1,  # 0 for Vit, 0.1 for Beit, if > 0 will use `layer_scale` on block output
    temperature=1e4,  # [Pos emb args] temperature for RoPE on patches
    ref_feature_shape=16,  # [Pos emb args] reference feature shape for RoPE on patches
    reg_ref_feature_shape=14,  # [Pos emb args] reference feature shape for RoPE on registers
    drop_connect_rate=0,
    classifier_activation="softmax",
    pretrained=None,
    force_reload_mismatch=False,  # set True if patch_size changed, will force reloading pos_emb and stem_conv weights
    model_name="beit",
    kwargs=None,
):
    norm_layer = RMSNorm if use_rms_norm else layers.LayerNormalization
    is_text_model = vocab_size > 0
    if is_text_model:
        """Text inputs"""
        inputs = layers.Input([None], dtype="int64")
        tok_emb = layers.Embedding(vocab_size, embed_dim, name="embed_tokens")(inputs)

        if text_use_positional_embedding:
            pos_idx = PositionalIndex(block_size=max_block_size, name="pos_idx")(inputs)
            pos_emb = layers.Embedding(max_block_size, embed_dim, name="wpe")(pos_idx)
            nn = tok_emb + pos_emb
        else:
            nn = tok_emb
        nn = layers.Dropout(text_positional_dropout)(nn) if text_positional_dropout > 0 else nn
        patch_height = -1
        num_classes = vocab_size
    else:
        # Regard input_shape as force using original shape if len(input_shape) == 4,
        # else assume channel dimension is the one with min value in input_shape, and put it first or last regarding image_data_format
        input_shape = backend.align_input_shape_by_image_data_format(input_shape)
        inputs = layers.Input(input_shape)
        num_cls_tokens = 1 if with_cls_token else 0
        """ Patch embedding """
        # torch conv kernel initializer: uniform(-1/sqrt(k), 1/sqrt(k)), where k = weight.size(1) * prod(*kernel_size)
        # Not using same one as torch, as current one works better in some tests, decreasing loss 3.7055 -> 3.4224 in the first epoch clip training for TF
        kernel_initializer = None if backend.is_torch_backend else initializers.RandomUniform(minval=-1 / (patch_size**0.5), maxval=1 / (patch_size**0.5))
        nn = PatchConv2DWithResampleWeights(
            embed_dim, patch_size, strides=patch_size, padding="valid", kernel_initializer=kernel_initializer, use_bias=use_patch_bias, name="stem_conv"
        )(inputs)
        nn = nn if image_data_format() == "channels_last" else layers.Permute([2, 3, 1])(nn)  # channels_first -> channels_last
        patch_height = nn.shape[1]
        nn = layers.Reshape([-1, nn.shape[-1]])(nn)

        """ mask_inputs for BeitV2 training model, also affecting output block """
        if use_mask_inputs:
            mask_inputs = layers.Input([nn.shape[1]])
            mask_inputs_expanded = functional.expand_dims(mask_inputs, axis=-1)
            weight_init_value = initializers.TruncatedNormal(stddev=0.02)
            masked_pos = ChannelAffine(weight_init_value=weight_init_value, name="mask_token")(functional.repeat(mask_inputs_expanded, embed_dim, axis=-1))
            nn = nn * (1 - mask_inputs_expanded) + masked_pos

        """ Positional embedding """
        if use_abs_pos_emb and use_abs_pos_emb_on_cls_token:
            nn = ClassToken(name="cls_token")(nn) if with_cls_token else nn
            nn = PositionalEmbedding(input_height=patch_height, name="positional_embedding")(nn)
        elif use_abs_pos_emb:
            nn = PositionalEmbedding(input_height=patch_height, name="positional_embedding")(nn)
            nn = ClassToken(name="cls_token")(nn) if with_cls_token else nn
        elif with_cls_token:
            nn = ClassToken(name="cls_token")(nn)

        if num_reg_tokens > 0:
            nn = ClassToken(num_tokens=num_reg_tokens, is_at_end=True, name="reg_token")(nn)

        nn = norm_layer(axis=-1, epsilon=layer_norm_epsilon, name="pre_ln")(nn) if use_pre_norm else nn

    """ Attention MLP body """
    attn_params = {
        "num_heads": num_heads,
        "key_dim": attn_key_dim,
        "qv_bias": attn_qv_bias,
        "qkv_bias": attn_qkv_bias,
        "use_qk_norm": use_qk_norm,
        "out_weight": attn_out_weight,
        "out_bias": attn_out_bias,
        "attn_height": patch_height,
        "use_pos_emb": not use_abs_pos_emb,
        "use_rot_pos_emb": use_rot_pos_emb,
        "with_cls_token": with_cls_token,
        "num_reg_tokens": num_reg_tokens,
        "temperature": temperature,
        "ref_feature_shape": ref_feature_shape,
        "reg_ref_feature_shape": reg_ref_feature_shape,
        "text_max_block_size": max_block_size if vocab_size > 0 else 0,
        "attn_dropout": attn_dropout,
        "shared_pos_emb": MultiHeadRelativePositionalEmbedding(attn_height=patch_height, name="shared_pos_emb") if use_shared_pos_emb_for_attn else None,
    }

    drop_connect_rates = drop_connect_rates_split([depth], 0.0, drop_connect_rate)[0]
    for id in range(depth):
        name = "block{}_".format(id)
        block_drop_rate = drop_connect_rates[id]
        nn = attention_mlp_block(
            nn, layer_scale, mlp_ratio, use_gated_mlp, use_norm_mlp, use_rms_norm, block_drop_rate, layer_norm_epsilon, activation, attn_params, name
        )

        if patch_merging_block_id == id:
            print(">>>> Before patch merging: blocks: {}, attn_height: {}".format(nn.shape[1], attn_params["attn_height"]))
            if attn_params["attn_height"] > 0:
                attn_params["attn_height"] = int(np.ceil(attn_params["attn_height"] / ((nn.shape[1] - 1) / patch_merging_num_tokens) ** 0.5))
            nn = patch_merging(nn, num_tokens_out=patch_merging_num_tokens, use_cls_token=not is_text_model, epsilon=layer_norm_epsilon, name="patch_merging_")
            print(">>>> After patch merging: blocks: {}, attn_height: {}".format(nn.shape[1], attn_params["attn_height"]))
            force_reload_mismatch = True

    """ Head """
    if is_text_model or not include_top:  # Text model
        nn = norm_layer(axis=-1, epsilon=layer_norm_epsilon, name="out_ln")(nn)  # "channels_first" also using axis=-1
    elif use_mask_inputs:  # mask_inputs for BeitV2 training model
        nn = functional.gather_nd(nn[:, 1:, :], functional.where(functional.equal(mask_inputs, 1)))
    elif use_cat_head:  # DINOv2
        nn = norm_layer(axis=-1, epsilon=layer_norm_epsilon, name="out_ln")(nn)  # "channels_first" also using axis=-1
        nn = functional.concat([nn[:, 0], functional.reduce_mean(nn[:, num_cls_tokens:, :], axis=1)], axis=-1)
    elif use_mean_pooling_head:
        nn = functional.reduce_mean(nn[:, num_cls_tokens:, :], axis=1)
        nn = norm_layer(axis=-1, epsilon=layer_norm_epsilon, name="out_ln")(nn)  # "channels_first" also using axis=-1
    else:  # FlexiViT
        nn = norm_layer(axis=-1, epsilon=layer_norm_epsilon, name="out_ln")(nn)  # "channels_first" also using axis=-1
        nn = nn[:, 0]

    """ Output """
    if num_classes > 0 and include_top:
        head_init = initializers.TruncatedNormal(stddev=0.02)  # HeadInitializer() -> Unknown initializer 'HeadInitializer' when loading [???]
        bias_init = initializers.TruncatedNormal(stddev=0.02)  # HeadInitializer() -> Unknown initializer 'HeadInitializer' when loading [???]
        nn = layers.Dense(
            num_classes, dtype="float32", activation=classifier_activation, kernel_initializer=head_init, bias_initializer=bias_init, name="predictions"
        )(nn)
    model = models.Model([inputs, mask_inputs] if use_mask_inputs else inputs, nn, name=model_name)
    add_pre_post_process(model, input_shape=input_shape, rescale_mode="tf")
    mismatch_class = [PatchConv2DWithResampleWeights, PositionalEmbedding if use_abs_pos_emb else MultiHeadRelativePositionalEmbedding]
    reload_model_weights(model, PRETRAINED_DICT, "beit", pretrained, mismatch_class, force_reload_mismatch)
    return model


@register_model
def BeitBasePatch16(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet21k-ft1k", **kwargs):
    embed_dim = 768
    depth = 12
    num_heads = 12
    layer_scale = kwargs.get("layer_scale", 0.1)
    force_reload_mismatch = kwargs.get("patch_size", 16) != 16  # If patch_size not 16, force reload pos_emb and stem_conv weights
    kwargs.pop("kwargs", None)  # From BeitV2BasePatch16
    return Beit(**locals(), model_name=kwargs.pop("model_name", "beit_base_patch16"), **kwargs)


@register_model
def BeitV2BasePatch16(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet21k-ft1k", **kwargs):
    return BeitBasePatch16(**locals(), **kwargs, model_name="beit_v2_base_patch16")


@register_model
def BeitLargePatch16(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet21k-ft1k", **kwargs):
    embed_dim = 1024
    depth = 24
    num_heads = 16
    layer_scale = kwargs.get("layer_scale", 1e-5)
    force_reload_mismatch = kwargs.get("patch_size", 16) != 16  # If patch_size not 16, force reload pos_emb and stem_conv weights
    kwargs.pop("kwargs", None)  # From BeitV2LargePatch16
    return Beit(**locals(), model_name=kwargs.pop("model_name", "beit_large_patch16"), **kwargs)


@register_model
def BeitV2LargePatch16(
    input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet21k-ft1k", **kwargs
):
    return BeitLargePatch16(**locals(), **kwargs, model_name="beit_v2_large_patch16")


""" keras_model_load_weights_from_pytorch_model """


def keras_model_load_weights_from_pytorch_model(keras_model, timm_vit_model, save_name=None):
    from keras_cv_attention_models import download_and_load, attention_layers

    skip_weights = ["relative_position_index", "mask_token"]
    unstack_weights = ["cls_token", "gamma_1", "gamma_2", "relative_position_bias_table", "q_bias", "v_bias", "pos_embed"]
    if "dinov2" in keras_model.name:
        tail_align_dict = {}
    else:
        tail_align_dict = {"attn_gamma": -6, "mlp_gamma": -9, "attn_query_bias": -1, "attn_value_bias": -1, "attn_pos_emb": -1}
    if "flexivit" in keras_model.name:
        full_name_align_dict = {"cls_token": -2, "positional_embedding": -1}
    else:
        full_name_align_dict = {"cls_token": -1, "positional_embedding": -1}

    if "shared_pos_emb" in [ii.name for ii in keras_model.layers]:
        full_name_align_dict["shared_pos_emb"] = -4
        if "attn_gamma" in tail_align_dict:
            tail_align_dict["attn_gamma"] += 1
        if "mlp_gamma" in tail_align_dict:
            tail_align_dict["mlp_gamma"] += 1

    additional_transfer = {attention_layers.MultiHeadRelativePositionalEmbedding: lambda ww: [ww[0].T]}

    download_and_load.keras_reload_from_torch_model(
        torch_model=timm_vit_model,
        keras_model=keras_model,
        input_shape=keras_model.input_shape[1:-1],
        skip_weights=skip_weights,
        unstack_weights=unstack_weights,
        tail_align_dict=tail_align_dict,
        full_name_align_dict=full_name_align_dict,
        tail_split_position=1,
        additional_transfer=additional_transfer,
        save_name=save_name if save_name is not None else (keras_model.name + "_{}.h5".format(keras_model.input_shape[1])),
        do_convert=True,
        # do_predict=False if "eva_giant" in keras_model.name else True,
    )
