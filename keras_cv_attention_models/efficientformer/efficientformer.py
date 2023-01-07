import tensorflow as tf
from tensorflow import keras
from keras_cv_attention_models.attention_layers import (
    ChannelAffine,
    MultiHeadPositionalEmbedding,
    batchnorm_with_activation,
    conv2d_no_bias,
    drop_block,
    layer_norm,
    mlp_block,
    mhsa_with_multi_head_position,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

PRETRAINED_DICT = {
    "efficientformer_l1": {"imagenet": {224: "7698d40d502ccc548a7e2890fb33db34"}},
    "efficientformer_l3": {"imagenet": {224: "ee3d11742d233bc2ec36648440cb5a0b"}},
    "efficientformer_l7": {"imagenet": {224: "66c26fc1e0bd39bbf6886d570956d178"}},
}


def meta_block(inputs, is_attn_block=False, num_heads=8, key_dim=32, attn_ratio=4, mlp_ratio=4, layer_scale=0, drop_rate=0, activation="gelu", name=""):
    input_channel = inputs.shape[-1]

    if is_attn_block:
        nn = layer_norm(inputs, name=name + "attn_")
        nn = mhsa_with_multi_head_position(nn, num_heads, key_dim=key_dim, attn_ratio=attn_ratio, use_bn=False, qkv_bias=True, out_bias=True, name=name)
    else:
        nn = keras.layers.AvgPool2D(pool_size=3, strides=1, padding="SAME")(inputs)  # count_include_pad=False [ ??? ]
        nn = nn - inputs
    nn = ChannelAffine(use_bias=False, weight_init_value=layer_scale, name=name + "attn_gamma")(nn) if layer_scale >= 0 else nn
    nn = drop_block(nn, drop_rate=drop_rate, name=name + "attn_")
    attn_out = keras.layers.Add(name=name + "attn_out")([inputs, nn])

    if is_attn_block:
        nn = layer_norm(attn_out, name=name + "mlp_")
        nn = mlp_block(nn, input_channel * mlp_ratio, activation=activation, name=name)
    else:
        nn = conv2d_no_bias(attn_out, input_channel * mlp_ratio, 1, strides=1, use_bias=True, name=name + "mlp_1_")
        nn = batchnorm_with_activation(nn, activation=activation, name=name + "mlp_1_")
        nn = conv2d_no_bias(nn, input_channel, 1, strides=1, use_bias=True, name=name + "mlp_2_")
        nn = batchnorm_with_activation(nn, activation=None, name=name + "mlp_2_")
    nn = ChannelAffine(use_bias=False, weight_init_value=layer_scale, name=name + "mlp_gamma")(nn) if layer_scale >= 0 else nn
    nn = drop_block(nn, drop_rate=drop_rate, name=name + "mlp_")
    return keras.layers.Add(name=name + "output")([attn_out, nn])


def EfficientFormer(
    num_blocks=[3, 2, 6, 4],
    out_channels=[48, 96, 224, 448],
    num_attn_blocks_in_last_stack=1,
    stem_width=-1,
    stem_activation="relu",
    mlp_ratio=4,
    layer_scale=1e-5,
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="gelu",
    drop_connect_rate=0,
    classifier_activation=None,
    use_distillation=True,
    dropout=0,
    pretrained=None,
    model_name="efficientformer",
    kwargs=None,
):
    inputs = keras.layers.Input(input_shape)
    stem_width = stem_width if stem_width > 0 else out_channels[0]
    stem_activation = stem_activation if stem_activation is not None else activation
    nn = conv2d_no_bias(inputs, stem_width // 2, 3, strides=2, use_bias=True, padding="same", name="stem_1_")
    nn = batchnorm_with_activation(nn, activation=stem_activation, name="stem_1_")
    nn = conv2d_no_bias(nn, stem_width, 3, strides=2, use_bias=True, padding="same", name="stem_2_")
    nn = batchnorm_with_activation(nn, activation=stem_activation, name="stem_2_")

    """ stages """
    total_blocks = sum(num_blocks)
    global_block_id = 0
    for stack_id, (num_block, out_channel) in enumerate(zip(num_blocks, out_channels)):
        stack_name = "stack{}_".format(stack_id + 1)
        if stack_id > 0:
            ds_name = stack_name + "downsample_"
            nn = conv2d_no_bias(nn, out_channel, kernel_size=3, strides=2, use_bias=True, padding="SAME", name=ds_name)
            nn = batchnorm_with_activation(nn, activation=None, name=ds_name)
        for block_id in range(num_block):
            block_name = stack_name + "block{}_".format(block_id + 1)
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            is_attn_block = True if stack_id == len(num_blocks) - 1 and block_id >= num_block - num_attn_blocks_in_last_stack else False
            nn = meta_block(nn, is_attn_block, mlp_ratio=mlp_ratio, layer_scale=layer_scale, drop_rate=block_drop_rate, activation=activation, name=block_name)
            global_block_id += 1

    """ output """
    if num_classes > 0:
        nn = layer_norm(nn, name="pre_output_")
        nn = keras.layers.GlobalAveragePooling2D()(nn)  # tf.reduce_mean(nn, axis=1)
        if dropout > 0 and dropout < 1:
            nn = keras.layers.Dropout(dropout)(nn)
        out = keras.layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="head")(nn)

        if use_distillation:
            distill = keras.layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="distill_head")(nn)
            out = [out, distill]
    else:
        out = nn

    model = keras.models.Model(inputs, out, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    reload_model_weights(model, PRETRAINED_DICT, "efficientformer", pretrained, MultiHeadPositionalEmbedding)
    return model


def EfficientFormerL1(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", use_distillation=True, pretrained="imagenet", **kwargs):
    return EfficientFormer(**locals(), model_name="efficientformer_l1", **kwargs)


def EfficientFormerL3(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", use_distillation=True, pretrained="imagenet", **kwargs):
    num_blocks = [4, 4, 12, 6]
    out_channels = [64, 128, 320, 512]
    num_attn_blocks_in_last_stack = 4
    return EfficientFormer(**locals(), model_name="efficientformer_l3", **kwargs)


def EfficientFormerL7(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", use_distillation=True, pretrained="imagenet", **kwargs):
    num_blocks = [6, 6, 18, 8]
    out_channels = [96, 192, 384, 768]
    num_attn_blocks_in_last_stack = 8
    return EfficientFormer(**locals(), model_name="efficientformer_l7", **kwargs)
