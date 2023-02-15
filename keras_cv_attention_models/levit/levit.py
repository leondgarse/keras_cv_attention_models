import math
import numpy as np
from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models, functional, image_data_format, initializers, register_keras_serializable
from keras_cv_attention_models.download_and_load import reload_model_weights
from keras_cv_attention_models.attention_layers import (
    activation_by_name,
    batchnorm_with_activation,
    conv2d_no_bias,
    scaled_dot_product_attention,
    add_pre_post_process,
)


PRETRAINED_DICT = {
    "levit128s": {"imagenet": "5e35073bb6079491fb0a1adff833da23"},
    "levit128": {"imagenet": "730c100fa4d5a10cf48fb923bb7da5c3"},
    "levit192": {"imagenet": "b078d2fe27857d0bdb26e101703210e2"},
    "levit256": {"imagenet": "9ada767ba2798c94aa1c894a00ae40fd"},
    "levit384": {"imagenet": "520f207f7f4c626b83e21564dd0c92a3"},
}


@register_keras_serializable(package="levit")
class MultiHeadPositionalEmbedding(layers.Layer):
    def __init__(self, query_height=-1, key_height=-1, **kwargs):
        super(MultiHeadPositionalEmbedding, self).__init__(**kwargs)
        self.query_height, self.key_height = query_height, key_height

    def build(self, input_shape, **kwargs):
        _, num_heads, qq_blocks, kk_blocks = input_shape
        self.bb = self.add_weight(name="positional_embedding", shape=(kk_blocks, num_heads), initializer="zeros", trainable=True)

        if self.query_height == -1:
            q_blocks_h = q_blocks_w = int(float(qq_blocks) ** 0.5)  # hh == ww
        else:
            q_blocks_h, q_blocks_w = self.query_height, int(qq_blocks / self.query_height)

        strides = int(math.ceil(float(kk_blocks / qq_blocks) ** 0.5))
        if self.key_height == -1:
            k_blocks_h = q_blocks_h * strides
            while kk_blocks % k_blocks_h != 0:
                k_blocks_h -= 1
            k_blocks_w = int(kk_blocks / k_blocks_h)
        else:
            k_blocks_h, k_blocks_w = self.key_height, int(kk_blocks / self.key_height)
        self.k_blocks_h, self.k_blocks_w = k_blocks_h, k_blocks_w
        # print(f"{q_blocks_h = }, {q_blocks_w = }, {k_blocks_h = }, {k_blocks_w = }, {strides = }")

        x1, y1 = np.meshgrid(range(q_blocks_h), range(q_blocks_w))
        x2, y2 = np.meshgrid(range(k_blocks_h), range(k_blocks_w))
        aa = np.concatenate([np.reshape(x1, (-1, 1)), np.reshape(y1, (-1, 1))], axis=-1)
        bb = np.concatenate([np.reshape(x2, (-1, 1)), np.reshape(y2, (-1, 1))], axis=-1)
        # print(f">>>> {aa.shape = }, {bb.shape = }") # aa.shape = (16, 2), bb.shape = (49, 2)
        cc = [np.abs(bb - ii * strides) for ii in aa]
        bb_pos = np.stack([ii[:, 0] + ii[:, 1] * k_blocks_h for ii in cc])

        if hasattr(self, "register_buffer"):  # PyTorch
            self.register_buffer("bb_pos", functional.convert_to_tensor(bb_pos, dtype="int64"), persistent=False)
        else:
            self.bb_pos = functional.convert_to_tensor(bb_pos, dtype="int64")
        # print(f">>>> {self.bb_pos.shape = }")    # self.bb_pos.shape = (16, 49)

        super(MultiHeadPositionalEmbedding, self).build(input_shape)

    def call(self, inputs, **kwargs):
        pos_bias = functional.transpose(functional.gather(self.bb, self.bb_pos), [2, 0, 1])
        return inputs + pos_bias

    def get_config(self):
        base_config = super().get_config()
        base_config.update({"query_height": self.query_height, "key_height": self.key_height})
        return base_config

    def load_resized_weights(self, source_layer, method="bilinear"):
        if isinstance(source_layer, dict):
            source_bb = list(source_layer.values())[0]  # weights
        else:
            source_bb = source_layer.bb  # layer
        source_bb = np.array(source_bb.detach() if hasattr(source_bb, "detach") else source_bb).astype("float32")
        hh = ww = int(float(source_bb.shape[0]) ** 0.5)
        ss = source_bb.reshape((hh, ww, source_bb.shape[-1]))  # [hh, ww, num_heads]
        # target_hh = target_ww = int(float(self.bb.shape[0]) ** 0.5)
        tt = backend.numpy_image_resize(ss, target_shape=[self.k_blocks_h, self.k_blocks_w], method=method)  # [target_hh, target_ww, num_heads]
        tt = tt.reshape((self.bb.shape))  # [target_hh * target_ww, num_heads]
        self.set_weights([tt])

    def show_pos_emb(self, rows=1, base_size=2):
        import matplotlib.pyplot as plt

        hh = ww = int(float(self.bb.shape[0]) ** 0.5)
        ss = np.array(self.bb.detach() if hasattr(self.bb, "detach") else self.bb)
        ss = ss.reshape((hh, ww, -1))
        cols = int(math.ceil(ss.shape[-1] / rows))
        fig, axes = plt.subplots(rows, cols, figsize=(base_size * cols, base_size * rows))
        for id, ax in enumerate(axes.flatten()):
            ax.imshow(ss[:, :, id])
            ax.set_axis_off()
        fig.tight_layout()
        return fig


def mhsa_with_multi_head_position(
    inputs, num_heads, key_dim=-1, output_dim=-1, attn_ratio=1, use_bn=False, qkv_bias=False, out_bias=False, activation=None, name=None
):
    _, height, width, input_channels = inputs.shape
    key_dim = key_dim if key_dim > 0 else input_channels // num_heads
    output_dim = output_dim if output_dim > 0 else input_channels
    embed_dim = key_dim * num_heads

    qkv_dim = (attn_ratio + 1 + 1) * embed_dim
    qkv = layers.Dense(qkv_dim, use_bias=qkv_bias, name=name and name + "qkv")(inputs)
    qkv = batchnorm_with_activation(qkv, activation=None, axis=-1, name=name and name + "qkv_") if use_bn else qkv
    qkv = functional.reshape(qkv, (-1, qkv.shape[1] * qkv.shape[2], num_heads, qkv_dim // num_heads))
    qq, kk, vv = functional.split(qkv, [key_dim, key_dim, key_dim * attn_ratio], axis=-1)
    qq, kk, vv = functional.transpose(qq, [0, 2, 1, 3]), functional.transpose(kk, [0, 2, 3, 1]), functional.transpose(vv, [0, 2, 1, 3])

    output_shape = (height, width, output_dim)
    pos_emb = MultiHeadPositionalEmbedding(query_height=height, name=name and name + "attn_pos")
    output = scaled_dot_product_attention(qq, kk, vv, output_shape, pos_emb=pos_emb, out_weight=False, name=name)

    if activation:
        output = activation_by_name(output, activation=activation, name=name)
    # [batch, cls_token + hh * ww, num_heads * key_dim] * [num_heads * key_dim, out] --> [batch, cls_token + hh * ww, out]
    output = layers.Dense(output_dim, use_bias=out_bias, name=name and name + "out")(output)
    if use_bn:
        output = batchnorm_with_activation(output, activation=None, zero_gamma=True, axis=-1, name=name and name + "out_")
    return output


def mhsa_with_multi_head_position_and_strides(
    inputs, num_heads, key_dim=-1, output_dim=-1, attn_ratio=1, strides=1, use_bn=True, qkv_bias=False, out_bias=False, activation=None, name=None
):
    _, _, _, input_channels = inputs.shape
    key_dim = key_dim if key_dim > 0 else input_channels // num_heads
    output_dim = output_dim if output_dim > 0 else input_channels
    embed_dim = num_heads * key_dim

    qq = inputs[:, ::strides, ::strides, :] if strides > 1 else inputs
    height, width = qq.shape[1], qq.shape[2]
    # print(f"{height = }, {width = }, {strides = }, {inputs.shape = }")
    qq = layers.Dense(embed_dim, use_bias=qkv_bias, name=name and name + "q")(qq)
    qq = batchnorm_with_activation(qq, activation=None, axis=-1, name=name and name + "q_") if use_bn else qq
    qq = functional.reshape(qq, [-1, qq.shape[1] * qq.shape[2], num_heads, key_dim])
    qq = functional.transpose(qq, [0, 2, 1, 3])

    kv_dim = (attn_ratio + 1) * embed_dim
    kv = layers.Dense(kv_dim, use_bias=qkv_bias, name=name and name + "kv")(inputs)
    kv = batchnorm_with_activation(kv, activation=None, axis=-1, name=name and name + "kv_") if use_bn else kv
    kv = functional.reshape(kv, (-1, kv.shape[1] * kv.shape[2], num_heads, kv_dim // num_heads))
    kk, vv = functional.split(kv, [key_dim, key_dim * attn_ratio], axis=-1)
    kk, vv = functional.transpose(kk, [0, 2, 3, 1]), functional.transpose(vv, [0, 2, 1, 3])

    output_shape = (height, width, output_dim)
    # print(f"{qq.shape = }, {kk.shape = }, {vv.shape = }, {output_shape = }")
    pos_emb = MultiHeadPositionalEmbedding(query_height=height, name=name and name + "attn_pos")
    output = scaled_dot_product_attention(qq, kk, vv, output_shape, pos_emb=pos_emb, out_weight=False, name=name)

    if activation:
        output = activation_by_name(output, activation=activation, name=name)
    # [batch, cls_token + hh * ww, num_heads * key_dim] * [num_heads * key_dim, out] --> [batch, cls_token + hh * ww, out]
    output = layers.Dense(output_dim, use_bias=out_bias, name=name and name + "out")(output)
    if use_bn:
        output = batchnorm_with_activation(output, activation=None, zero_gamma=True, axis=-1, name=name and name + "out_")
    return output


def res_mhsa_with_multi_head_position(inputs, embed_dim, num_heads, key_dim, attn_ratio, drop_rate=0, activation="hard_swish", name=""):
    nn = mhsa_with_multi_head_position(inputs, num_heads, key_dim, embed_dim, attn_ratio, use_bn=True, activation=activation, name=name)
    if drop_rate > 0:
        nn = layers.Dropout(drop_rate, noise_shape=(None, 1, 1), name=name + "drop")(nn)
    return layers.Add(name=name + "add")([inputs, nn])


def res_mlp_block(inputs, mlp_ratio, drop_rate=0, use_bias=False, activation="hard_swish", name=""):
    in_channels = inputs.shape[-1]
    nn = layers.Dense(in_channels * mlp_ratio, use_bias=use_bias, name=name + "1_dense")(inputs)
    nn = batchnorm_with_activation(nn, activation=activation, axis=-1, name=name + "1_")  # "channels_first" also using axis=-1
    nn = layers.Dense(in_channels, use_bias=use_bias, name=name + "2_dense")(nn)
    nn = batchnorm_with_activation(nn, activation=None, axis=-1, name=name + "2_")  # "channels_first" also using axis=-1
    if drop_rate > 0:
        nn = layers.Dropout(drop_rate, noise_shape=(None, 1, 1), name=name + "drop")(nn)
    return layers.Add(name=name + "add")([inputs, nn])


def attention_mlp_stack(inputs, out_channel, num_heads, depth, key_dim, attn_ratio, mlp_ratio, strides, stack_drop=0, activation="hard_swish", name=""):
    nn = inputs
    embed_dim = nn.shape[-1]
    stack_drop_s, stack_drop_e = stack_drop if isinstance(stack_drop, (list, tuple)) else [stack_drop, stack_drop]
    for ii in range(depth):
        block_name = name + "block{}_".format(ii + 1)
        drop_rate = stack_drop_s + (stack_drop_e - stack_drop_s) * ii / depth
        nn = res_mhsa_with_multi_head_position(nn, embed_dim, num_heads, key_dim, attn_ratio, drop_rate, activation=activation, name=block_name)
        if mlp_ratio > 0:
            nn = res_mlp_block(nn, mlp_ratio, drop_rate, activation=activation, name=block_name + "mlp_")
    if embed_dim != out_channel:
        block_name = name + "downsample_"
        ds_num_heads = embed_dim // key_dim
        ds_attn_ratio = attn_ratio * strides
        nn = mhsa_with_multi_head_position_and_strides(nn, ds_num_heads, key_dim, out_channel, ds_attn_ratio, strides, activation=activation, name=block_name)
        if mlp_ratio > 0:
            nn = res_mlp_block(nn, mlp_ratio, drop_rate, activation=activation, name=block_name + "mlp_")
    return layers.Activation("linear", name=name + "output")(nn)  # Identity, Just need a name here


def patch_stem(inputs, stem_width, activation="hard_swish", name=""):
    nn = conv2d_no_bias(inputs, stem_width // 8, 3, strides=2, padding="same", name=name + "1_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "1_")
    nn = conv2d_no_bias(nn, stem_width // 4, 3, strides=2, padding="same", name=name + "2_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "2_")
    nn = conv2d_no_bias(nn, stem_width // 2, 3, strides=2, padding="same", name=name + "3_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "3_")
    nn = conv2d_no_bias(nn, stem_width, 3, strides=2, padding="same", name=name + "4_")
    nn = batchnorm_with_activation(nn, activation=None, name=name + "4_")
    return nn


def LeViT(
    patch_channel=128,
    out_channels=[256, 384, 384],  # C
    num_heads=[4, 6, 8],  # N
    depthes=[2, 3, 4],  # X
    key_dims=[16, 16, 16],  # D
    attn_ratios=[2, 2, 2],  # attn_ratio
    mlp_ratios=[2, 2, 2],  # mlp_ratio
    strides=[2, 2, 0],  # down_ops, strides
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="hard_swish",
    drop_connect_rate=0,
    dropout=0,
    classifier_activation=None,
    use_distillation=True,
    pretrained="imagenet",
    model_name="levit",
    kwargs=None,
):
    # Regard input_shape as force using original shape if len(input_shape) == 4,
    # else assume channel dimention is the one with min value in input_shape, and put it first or last regarding image_data_format
    input_shape = backend.align_input_shape_by_image_data_format(input_shape)
    inputs = layers.Input(input_shape)
    nn = patch_stem(inputs, patch_channel, activation=activation, name="stem_")
    nn = nn if image_data_format() == "channels_last" else layers.Permute([2, 3, 1])(nn)  # channels_first -> channels_last

    global_block_id = 0
    total_blocks = sum(depthes)
    drop_connect_s, drop_connect_e = drop_connect_rate if isinstance(drop_connect_rate, (list, tuple)) else (drop_connect_rate, drop_connect_rate)
    for id, (out_channel, num_head, depth, key_dim, attn_ratio, mlp_ratio, stride) in enumerate(
        zip(out_channels, num_heads, depthes, key_dims, attn_ratios, mlp_ratios, strides)
    ):
        name = "stack{}_".format(id + 1)
        stack_drop_s = drop_connect_s + (drop_connect_e - drop_connect_s) * global_block_id / total_blocks
        stack_drop_e = drop_connect_s + (drop_connect_e - drop_connect_s) * (global_block_id + depth) / total_blocks
        stack_drop = (stack_drop_s, stack_drop_e)
        nn = attention_mlp_stack(nn, out_channel, num_head, depth, key_dim, attn_ratio, mlp_ratio, stride, stack_drop, activation, name=name)
        global_block_id += depth
    nn = nn if image_data_format() == "channels_last" else layers.Permute([3, 1, 2])(nn)  # channels_first -> channels_last

    if num_classes == 0:
        out = nn
    else:
        nn = layers.GlobalAveragePooling2D(keepdims=True)(nn)
        if dropout > 0 and dropout < 1:
            nn = layers.Dropout(dropout)(nn)
        out = batchnorm_with_activation(nn, activation=None, name="head_")
        out = layers.Flatten()(out)
        out = layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="head")(out)

        if use_distillation:
            distill = batchnorm_with_activation(nn, activation=None, name="distill_head_")
            distill = layers.Flatten()(distill)
            distill = layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="distill_head")(distill)
            out = [out, distill]

    model = models.Model(inputs, out, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    reload_model_weights(model, PRETRAINED_DICT, "levit", pretrained, MultiHeadPositionalEmbedding)
    return model


def LeViT128S(input_shape=(224, 224, 3), num_classes=1000, use_distillation=True, classifier_activation=None, pretrained="imagenet", **kwargs):
    return LeViT(**locals(), model_name="levit128s", **kwargs)


def LeViT128(input_shape=(224, 224, 3), num_classes=1000, use_distillation=True, classifier_activation=None, pretrained="imagenet", **kwargs):
    num_heads = [4, 8, 12]
    depthes = [4, 4, 4]
    return LeViT(**locals(), model_name="levit128", **kwargs)


def LeViT192(input_shape=(224, 224, 3), num_classes=1000, use_distillation=True, classifier_activation=None, pretrained="imagenet", **kwargs):
    patch_channel = 192
    out_channels = [288, 384, 384]
    num_heads = [3, 5, 6]
    depthes = [4, 4, 4]
    key_dims = [32, 32, 32]
    return LeViT(**locals(), model_name="levit192", **kwargs)


def LeViT256(input_shape=(224, 224, 3), num_classes=1000, use_distillation=True, classifier_activation=None, pretrained="imagenet", **kwargs):
    patch_channel = 256
    out_channels = [384, 512, 512]
    num_heads = [4, 6, 8]
    depthes = [4, 4, 4]
    key_dims = [32, 32, 32]
    return LeViT(**locals(), model_name="levit256", **kwargs)


def LeViT384(input_shape=(224, 224, 3), num_classes=1000, use_distillation=True, classifier_activation=None, pretrained="imagenet", **kwargs):
    patch_channel = 384
    out_channels = [512, 768, 768]
    num_heads = [6, 9, 12]
    depthes = [4, 4, 4]
    key_dims = [32, 32, 32]
    return LeViT(**locals(), model_name="levit384", **kwargs)
