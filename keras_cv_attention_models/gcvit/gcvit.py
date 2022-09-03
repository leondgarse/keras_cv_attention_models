import tensorflow as tf
from tensorflow import keras
from keras_cv_attention_models.attention_layers import (
    ChannelAffine,
    MultiHeadRelativePositionalEmbedding,
    activation_by_name,
    conv2d_no_bias,
    depthwise_conv2d_no_bias,
    drop_block,
    layer_norm,
    mhsa_with_multi_head_relative_position_embedding,
    mlp_block,
    output_block,
    se_module,
    window_attention,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

PRETRAINED_DICT = {
    "gcvit_xx_tiny": {"imagenet": {224: "ff516a5a1d3dfdda0c0b2e0051206c00"}},
    "gcvit_x_tiny": {"imagenet": {224: "723155237e083716bb3df904c80711c4"}},
    "gcvit_tiny": {"imagenet": {224: "0e6ecf576b649f7077f4f2f8122b420e"}},
    "gcvit_small": {"imagenet": {224: "ac0cfb4240ae85a40a88691c2329edab"}},
    "gcvit_base": {"imagenet": {224: "ef6e4015239f68dcabbb8ae9cb799d76"}},
}

def gcvit_block(inputs, window_size, num_heads=4, global_query=None, mlp_ratio=4, layer_scale=0, drop_rate=0, activation="gelu", name=""):
    # print(global_query)
    input_channel = inputs.shape[-1]
    attn = layer_norm(inputs, name=name + "attn_")
    attention_block = lambda inputs, num_heads, name: mhsa_with_multi_head_relative_position_embedding(
        inputs, num_heads=num_heads, global_query=global_query, qkv_bias=True, out_bias=True, name=name
    )
    attn = window_attention(attn, window_size=window_size, num_heads=num_heads, attention_block=attention_block, name=name + "window_mhsa_")
    attn = ChannelAffine(use_bias=False, weight_init_value=layer_scale, name=name + "1_gamma")(attn) if layer_scale >= 0 else attn
    attn = drop_block(attn, drop_rate=drop_rate, name=name + "attn_")
    attn_out = keras.layers.Add(name=name + "attn_out")([inputs, attn])

    mlp = layer_norm(attn_out, name=name + "mlp_")
    mlp = mlp_block(mlp, int(input_channel * mlp_ratio), use_conv=False, activation=activation, name=name + "mlp_")
    mlp = ChannelAffine(use_bias=False, weight_init_value=layer_scale, name=name + "2_gamma")(mlp) if layer_scale >= 0 else mlp
    mlp = drop_block(mlp, drop_rate=drop_rate, name=name + "mlp_")
    return keras.layers.Add(name=name + "output")([attn_out, mlp])


def to_global_query(inputs, window_ratio, num_heads=4, activation="gelu", name=""):
    input_channel = inputs.shape[-1]
    query = inputs
    num_window = 1
    if window_ratio == 1:
        query = extract_feature(query, strides=1, activation=activation, name=name + "down1_")
    else:
        while num_window < window_ratio:
            num_window *= 2
            query = extract_feature(query, strides=2, activation=activation, name=name + "down{}_".format(num_window))

    # print(f"{inputs.shape = }, {query.shape = }, {num_window = }, {window_ratio = }")
    query = tf.reshape(query, [-1, query.shape[1] * query.shape[2], num_heads, input_channel // num_heads])
    query = tf.transpose(query, [0, 2, 1, 3])
    query = tf.repeat(query, num_window * num_window, axis=0)
    # print(f"{query.shape = }")
    return query


def down_sample(inputs, out_channels=-1, activation="gelu", name=""):
    out_channels = out_channels if out_channels > 0 else inputs.shape[-1]
    nn = layer_norm(inputs, name=name + "down_1_")
    nn = extract_feature(nn, strides=1, activation=activation, name=name + "down_")
    nn = conv2d_no_bias(nn, out_channels, kernel_size=3, strides=2, padding="same", name=name + "down_")
    nn = layer_norm(nn, name=name + "down_2_")
    return nn


def extract_feature(inputs, strides=2, activation="gelu", name=""):
    input_channel = inputs.shape[-1]
    nn = depthwise_conv2d_no_bias(inputs, kernel_size=3, padding="same", name=name + "extract_")
    nn = activation_by_name(nn, activation=activation, name=name + "extract_")
    nn = se_module(nn, divisor=1, use_bias=False, activation=activation, use_conv=False, name=name + "extract_se_")
    nn = conv2d_no_bias(nn, input_channel, kernel_size=1, name=name + "extract_")
    nn = inputs + nn
    return keras.layers.MaxPool2D(pool_size=3, strides=strides, padding="SAME", name=name + "extract_maxpool")(nn) if strides > 1 else nn
    # if strides > 1:
    #     nn = tf.pad(nn, [[0, 0], [1, 1], [1, 1], [0, 0]])
    #     nn = keras.layers.MaxPool2D(pool_size=3, strides=strides, padding="VALID", name=name + "extract_maxpool")(nn)
    # return nn


def GCViT(
    num_blocks=[2, 2, 6, 2],
    num_heads=[2, 4, 8, 16],
    # window_size=[7, 7, 14, 7],
    window_ratios=[8, 4, 1, 1],
    embed_dim=64,
    mlp_ratio=3,
    layer_scale=-1,
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="gelu",
    drop_connect_rate=0,
    classifier_activation="softmax",
    dropout=0,
    pretrained=None,
    model_name="gcvit",
    kwargs=None,
):
    """Patch stem"""
    inputs = keras.layers.Input(input_shape)
    nn = conv2d_no_bias(inputs, embed_dim, kernel_size=3, strides=2, use_bias=True, padding="SAME", name="stem_conv")
    nn = down_sample(nn, name="stem_")

    """ stages """
    total_blocks = sum(num_blocks)
    global_block_id = 0
    num_stacks = len(num_blocks)
    for stack_id, (num_block, num_head, window_ratio) in enumerate(zip(num_blocks, num_heads, window_ratios)):
        stack_name = "stack{}_".format(stack_id + 1)
        if stack_id > 0:
            nn = down_sample(nn, out_channels=nn.shape[-1] * 2, name=stack_name)

        window_size = (nn.shape[1] // window_ratio, nn.shape[2] // window_ratio)
        # window_size = (int(tf.math.ceil(nn.shape[1] / window_ratio), int(tf.math.ceil(nn.shape[2] / window_ratio))
        global_query = to_global_query(nn, window_ratio, num_head, activation=activation, name=stack_name + "q_global_")

        for block_id in range(num_block):
            block_name = stack_name + "block{}_".format(block_id + 1)
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            cur_global_query = None if block_id % 2 == 0 else global_query
            nn = gcvit_block(nn, window_size, num_head, cur_global_query, mlp_ratio, layer_scale, block_drop_rate, activation=activation, name=block_name)
            global_block_id += 1
    nn = layer_norm(nn, name="pre_output_")

    nn = output_block(nn, num_classes=num_classes, drop_rate=dropout, classifier_activation=classifier_activation)
    model = keras.models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    reload_model_weights(model, PRETRAINED_DICT, "gcvit", pretrained, MultiHeadRelativePositionalEmbedding)
    return model


def GCViT_XXTiny(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained=None, **kwargs):
    return GCViT(**locals(), model_name="gcvit_xx_tiny", **kwargs)


def GCViT_XTiny(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained=None, **kwargs):
    num_blocks = [3, 4, 6, 5]
    return GCViT(**locals(), model_name="gcvit_x_tiny", **kwargs)


def GCViT_Tiny(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 19, 5]
    return GCViT(**locals(), model_name="gcvit_tiny", **kwargs)


def GCViT_Small(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 19, 5]
    num_heads = [3, 6, 12, 24]
    embed_dim = 96
    mlp_ratio = 2
    layer_scale = 1e-5
    return GCViT(**locals(), model_name="gcvit_small", **kwargs)


def GCViT_Base(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 19, 5]
    num_heads = [4, 8, 16, 32]
    embed_dim = 128
    mlp_ratio = 2
    layer_scale = 1e-5
    return GCViT(**locals(), model_name="gcvit_base", **kwargs)
