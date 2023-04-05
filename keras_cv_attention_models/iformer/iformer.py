from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models, functional, image_data_format
from keras_cv_attention_models.attention_layers import (
    activation_by_name,
    add_with_layer_scale_and_drop_block,
    batchnorm_with_activation,
    conv2d_no_bias,
    depthwise_conv2d_no_bias,
    layer_norm,
    mlp_block,
    multi_head_self_attention,
    output_block,
    PositionalEmbedding,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

LAYER_NORM_EPSILON = 1e-6

PRETRAINED_DICT = {
    "iformer_base": {"imagenet": {224: "90c8fe1307bc8cbd73f5a7358e65956d", 384: "11d150c128a65f8dcc95871e87209491"}},
    "iformer_large": {"imagenet": {224: "2a116d5780a551846761aa43b1d74395", 384: "b21ae4484d1f5da7f1425ac301248e29"}},
    "iformer_small": {"imagenet": {224: "de2f8262da0cc1c6df43b32b827444f1", 384: "e1c7f4e52abf3514437138c62194d2d7"}},
}


def attention_low_frequency_mixer(inputs, num_heads=4, pool_size=1, dropout=0, name=""):
    height_axis, width_axis, channel_axis = (1, 2, 3) if image_data_format() == "channels_last" else (2, 3, 1)
    # print(f"{inputs.shape = }, {pool_size = }")
    if pool_size > 1:
        orign_height, orign_width = inputs.shape[height_axis], inputs.shape[width_axis]
        inputs = layers.AvgPool2D(pool_size, strides=pool_size, padding="same", name=name + "avg_down")(inputs)
    nn = inputs if image_data_format() == "channels_last" else layers.Permute([2, 3, 1])(inputs)  # channels_first -> channels_last
    nn = multi_head_self_attention(nn, num_heads=num_heads, qkv_bias=True, out_weight=False, attn_dropout=dropout, name=name + "attn_")
    nn = nn if image_data_format() == "channels_last" else layers.Permute([3, 1, 2])(nn)  # channels_last -> channels_first

    if pool_size > 1:
        nn = layers.UpSampling2D(size=pool_size, name=name + "up")(nn)
        if nn.shape[height_axis] != orign_height or nn.shape[width_axis] != orign_width:
            nn = nn[:, :orign_height, :orign_width] if image_data_format() == "channels_last" else nn[:, :, :orign_height, :orign_width]
    return nn


def conv_high_frequency_mixer(inputs, activation="gelu", name=""):
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    nn = conv2d_no_bias(inputs, inputs.shape[channel_axis] * 2, kernel_size=1, name=name)
    nn = depthwise_conv2d_no_bias(nn, kernel_size=3, padding="same", name=name)
    nn = activation_by_name(nn, activation=activation, name=name)
    return nn


def pool_high_frequency_mixer(inputs, activation="gelu", name=""):
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    nn = layers.MaxPool2D(3, strides=1, padding="SAME", name=name + "max")(inputs)
    nn = conv2d_no_bias(nn, inputs.shape[channel_axis] * 2, kernel_size=1, use_bias=True, name=name)
    nn = activation_by_name(nn, activation=activation, name=name)
    return nn


def conv_pool_attention_mixer(inputs, num_heads=4, key_dim=0, num_attn_low_heads=1, pool_size=1, dropout=0, activation="gelu", name=""):
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    input_channel = inputs.shape[channel_axis]
    key_dim = key_dim if key_dim > 0 else input_channel // num_heads
    attention_channels = num_attn_low_heads * key_dim
    conv_channels = (input_channel - attention_channels) // 2
    pool_channels = input_channel - attention_channels - conv_channels
    conv_branch, pool_branch, attention_branch = functional.split(inputs, [conv_channels, pool_channels, attention_channels], axis=channel_axis)
    # print(f"{key_dim = }, {num_heads = }, {num_attn_low_heads = }, {attention_channels = }")

    conv_branch = conv_high_frequency_mixer(conv_branch, activation=activation, name=name + "high_conv_branch_")
    pool_branch = pool_high_frequency_mixer(pool_branch, activation=activation, name=name + "high_pool_branch_")
    attention_branch = attention_low_frequency_mixer(attention_branch, num_heads=num_attn_low_heads, pool_size=pool_size, dropout=dropout, name=name + "low_")
    high_low = functional.concat([conv_branch, pool_branch, attention_branch], axis=channel_axis)
    # print(f"{conv_branch.shape = }, {pool_branch.shape = }, {attention_branch.shape = }, {high_low.shape = }")

    high_low_fused = depthwise_conv2d_no_bias(high_low, kernel_size=3, padding="same", name=name + "fuse_")
    high_low_out = layers.Add()([high_low, high_low_fused])

    out = conv2d_no_bias(high_low_out, input_channel, kernel_size=1, use_bias=True, name=name + "output_")
    if dropout > 0:
        out = layers.Dropout(dropout)(out)
    return out


def attention_mlp_block(inputs, num_heads=8, num_attn_low_heads=1, pool_size=1, mlp_ratio=4, layer_scale=0.1, drop_rate=0, activation="gelu", name=""):
    input_channel = inputs.shape[-1]  # Channels_last only

    """ attention """
    nn = layer_norm(inputs, epsilon=LAYER_NORM_EPSILON, axis=-1, name=name + "attn_")
    nn = nn if image_data_format() == "channels_last" else layers.Permute([3, 1, 2])(nn)  # channels_last -> channels_first
    nn = conv_pool_attention_mixer(nn, num_heads, num_attn_low_heads=num_attn_low_heads, pool_size=pool_size, activation=activation, name=name + "attn_")
    nn = nn if image_data_format() == "channels_last" else layers.Permute([2, 3, 1])(nn)  # channels_first -> channels_last
    attn_out = add_with_layer_scale_and_drop_block(inputs, nn, layer_scale=layer_scale, drop_rate=drop_rate, axis=-1, name=name + "attn_")

    """ MLP """
    nn = layer_norm(attn_out, epsilon=LAYER_NORM_EPSILON, axis=-1, name=name + "mlp_")
    nn = mlp_block(nn, input_channel * mlp_ratio, activation=activation, name=name + "mlp_")
    nn = add_with_layer_scale_and_drop_block(attn_out, nn, layer_scale=layer_scale, drop_rate=drop_rate, axis=-1, name=name + "mlp_")
    return nn


def InceptionTransformer(
    num_blocks=[3, 3, 9, 3],
    embed_dims=[96, 192, 320, 384],
    num_heads=[3, 6, 10, 12],
    num_attn_low_heads=[1, 3, [7] * 4 + [9] * 5, 11],
    pool_sizes=[2, 2, 1, 1],
    mlp_ratios=4,
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="gelu",
    drop_connect_rate=0,
    dropout=0,
    layer_scales=[0, 0, 1e-6, 1e-6],
    classifier_activation="softmax",
    pretrained=None,
    model_name="iformer",
    kwargs=None,
):
    # Regard input_shape as force using original shape if len(input_shape) == 4,
    # else assume channel dimention is the one with min value in input_shape, and put it first or last regarding image_data_format
    input_shape = backend.align_input_shape_by_image_data_format(input_shape)
    inputs = layers.Input(input_shape)

    """ Stem """
    nn = conv2d_no_bias(inputs, embed_dims[0] // 2, kernel_size=3, strides=2, padding="same", use_bias=True, name="stem_1_")
    nn = batchnorm_with_activation(nn, activation=activation, name="stem_1_")
    nn = conv2d_no_bias(nn, embed_dims[0], kernel_size=3, strides=2, padding="same", use_bias=True, name="stem_2_")
    nn = batchnorm_with_activation(nn, activation=None, name="stem_2_")

    """ stacks """
    total_blocks = sum(num_blocks)
    global_block_id = 0
    for stack_id, (num_block, embed_dim) in enumerate(zip(num_blocks, embed_dims)):
        stack_name = "stack{}_".format(stack_id + 1)
        if stack_id > 0:
            nn = conv2d_no_bias(nn, embed_dim, 3, strides=2, padding="same", use_bias=True, name=stack_name + "downsample_")
            nn = batchnorm_with_activation(nn, activation=None, name=stack_name + "downsample_")  # Using epsilon=1e-5
        nn = nn if image_data_format() == "channels_last" else layers.Permute([2, 3, 1])(nn)  # channels_first -> channels_last
        nn = PositionalEmbedding(name=stack_name + "positional_embedding")(nn)

        stack_num_attn_low_head = num_attn_low_heads[stack_id] if isinstance(num_attn_low_heads, (list, tuple)) else num_attn_low_heads
        num_head = num_heads[stack_id] if isinstance(num_heads, (list, tuple)) else num_heads
        mlp_ratio = mlp_ratios[stack_id] if isinstance(mlp_ratios, (list, tuple)) else mlp_ratios
        pool_size = pool_sizes[stack_id] if isinstance(pool_sizes, (list, tuple)) else pool_sizes
        layer_scale = layer_scales[stack_id] if isinstance(layer_scales, (list, tuple)) else layer_scales
        for block_id in range(num_block):
            name = stack_name + "block{}_".format(block_id + 1)
            num_attn_low_head = stack_num_attn_low_head[block_id] if isinstance(stack_num_attn_low_head, (list, tuple)) else stack_num_attn_low_head
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            nn = attention_mlp_block(nn, num_head, num_attn_low_head, pool_size, mlp_ratio, layer_scale, block_drop_rate, activation=activation, name=name)
            global_block_id += 1
        nn = nn if image_data_format() == "channels_last" else layers.Permute([3, 1, 2])(nn)  # channels_last -> channels_first
    nn = layer_norm(nn, epsilon=LAYER_NORM_EPSILON, name="pre_output_")

    nn = output_block(nn, num_classes=num_classes, drop_rate=dropout, classifier_activation=classifier_activation)
    model = models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    reload_model_weights(model, PRETRAINED_DICT, "iformer", pretrained, PositionalEmbedding)
    return model


def IFormerSmall(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return InceptionTransformer(**locals(), model_name="iformer_small", **kwargs)


def IFormerBase(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [4, 6, 14, 6]
    embed_dims = [96, 192, 384, 512]
    num_heads = [3, 6, 12, 16]
    num_attn_low_heads = [1, 3, [8] * 7 + [10] * 7, 15]
    return InceptionTransformer(**locals(), model_name="iformer_base", **kwargs)


def IFormerLarge(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [4, 6, 18, 8]
    embed_dims = [96, 192, 448, 640]
    num_heads = [3, 6, 14, 20]
    num_attn_low_heads = [1, 3, [10] * 9 + [12] * 9, 19]
    return InceptionTransformer(**locals(), model_name="iformer_large", **kwargs)
