import tensorflow as tf
from tensorflow import keras
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
    "iformer_base": {"imagenet": {224: "90c8fe1307bc8cbd73f5a7358e65956d"}},
    "iformer_large": {"imagenet": {224: "2a116d5780a551846761aa43b1d74395"}},
    "iformer_small": {"imagenet": {224: "de2f8262da0cc1c6df43b32b827444f1"}},
}


def attention_low_frequency_mixer(inputs, num_heads=4, pool_size=1, dropout=0, name=""):
    if pool_size > 1:
        orign_height, orign_width = inputs.shape[1], inputs.shape[2]
        inputs = keras.layers.AvgPool2D(pool_size, strides=pool_size, name=name + "avg_down")(inputs)
    nn = multi_head_self_attention(inputs, num_heads=num_heads, qkv_bias=True, out_weight=False, attn_dropout=dropout, name=name + "attn_")
    if pool_size > 1:
        nn = keras.layers.UpSampling2D(size=pool_size, name=name + "up")(nn)
        if nn.shape[1] != orign_height or nn.shape[2] != orign_width:
            nn = nn[:, :orign_height, :orign_width]
    return nn


def conv_high_frequency_mixer(inputs, activation="gelu", name=""):
    input_channel = inputs.shape[-1]
    conv_branch, pool_branch = tf.split(inputs, 2, axis=-1)

    conv_branch = conv2d_no_bias(conv_branch, input_channel, kernel_size=1, name=name + "conv_branch_")
    conv_branch = depthwise_conv2d_no_bias(conv_branch, kernel_size=3, padding="same", name=name + "conv_branch_")
    conv_branch = activation_by_name(conv_branch, activation=activation, name=name + "conv_branch_")

    pool_branch = keras.layers.MaxPooling2D(3, strides=1, padding="SAME", name=name + "pool_branch_max")(pool_branch)
    pool_branch = conv2d_no_bias(pool_branch, input_channel, kernel_size=1, use_bias=True, name=name + "pool_branch_")
    pool_branch = activation_by_name(pool_branch, activation=activation, name=name + "pool_branch_")
    return tf.concat([conv_branch, pool_branch], axis=-1)


def conv_attention_mixer(inputs, num_heads=4, key_dim=0, num_attn_low_heads=1, pool_size=1, dropout=0, activation="gelu", name=""):
    _, hh, ww, input_channel = inputs.shape
    key_dim = key_dim if key_dim > 0 else input_channel // num_heads
    attention_low_channels = num_attn_low_heads * key_dim

    # print(f"{key_dim = }, {num_heads = }, {num_attn_low_heads = }, {attention_low_channels = }")
    conv_high, attention_low = tf.split(inputs, [-1, attention_low_channels], axis=-1)
    conv_high = conv_high_frequency_mixer(conv_high, activation="gelu", name=name + "high_")
    attention_low = attention_low_frequency_mixer(attention_low, num_heads=num_attn_low_heads, pool_size=pool_size, dropout=dropout, name=name + "low_")
    high_low = tf.concat([conv_high, attention_low], axis=-1)
    # print(f"{conv_high.shape = }, {attention_low.shape = }, {high_low.shape = }")
    high_low_fused = depthwise_conv2d_no_bias(high_low, kernel_size=3, padding="same", name=name + "fuse_")
    high_low_out = keras.layers.Add()([high_low, high_low_fused])

    out = conv2d_no_bias(high_low_out, input_channel, kernel_size=1, use_bias=True, name=name + "output_")
    if dropout > 0:
        out = keras.layers.Dropout(dropout)(out)
    return out


def attention_mlp_block(inputs, num_heads=8, num_attn_low_heads=1, pool_size=1, mlp_ratio=4, layer_scale=0.1, drop_rate=0, activation="gelu", name=""):
    input_channel = inputs.shape[-1]

    """ attention """
    nn = layer_norm(inputs, epsilon=LAYER_NORM_EPSILON, name=name + "attn_")
    nn = conv_attention_mixer(nn, num_heads=num_heads, num_attn_low_heads=num_attn_low_heads, pool_size=pool_size, activation=activation, name=name + "attn_")
    attn_out = add_with_layer_scale_and_drop_block(inputs, nn, layer_scale=layer_scale, drop_rate=drop_rate, name=name + "attn_")

    """ MLP """
    nn = layer_norm(attn_out, epsilon=LAYER_NORM_EPSILON, name=name + "mlp_")
    nn = mlp_block(nn, input_channel * mlp_ratio, activation=activation, name=name + "mlp_")
    nn = add_with_layer_scale_and_drop_block(attn_out, nn, layer_scale=layer_scale, drop_rate=drop_rate, name=name + "mlp_")
    return nn


def InceptionTransformer(
    num_blocks=[3, 3, 9, 3],
    embed_dims=[96, 192, 320, 384],
    num_heads=[3, 6, 10, 12],
    num_attn_low_heads=[1, 3, [7] * 4 + [9] * 5, 11],
    pool_sizes=[2, 2, 1, 1],
    layer_scales=[0, 0, 1e-6, 1e-6],
    mlp_ratios=4,
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="gelu",
    drop_connect_rate=0,
    dropout=0,
    classifier_activation="softmax",
    pretrained=None,
    model_name="iformer",
    kwargs=None,
):
    inputs = keras.layers.Input(input_shape)

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
    nn = layer_norm(nn, epsilon=LAYER_NORM_EPSILON, name="pre_output_")

    nn = output_block(nn, num_classes=num_classes, drop_rate=dropout, classifier_activation=classifier_activation)
    model = tf.keras.models.Model(inputs, nn, name=model_name)
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
