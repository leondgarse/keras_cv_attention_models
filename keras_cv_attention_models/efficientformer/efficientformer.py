from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models, functional, image_data_format
from keras_cv_attention_models.attention_layers import (
    add_with_layer_scale_and_drop_block,
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


def attn_block(inputs, num_heads=8, key_dim=32, attn_ratio=4, mlp_ratio=4, layer_scale=0, drop_rate=0, activation="gelu", name=""):
    input_channel = inputs.shape[-1]  # also using -1 for channels_first

    nn = layer_norm(inputs, axis=-1, name=name + "attn_")
    nn = mhsa_with_multi_head_position(nn, num_heads, key_dim=key_dim, attn_ratio=attn_ratio, qkv_bias=True, out_bias=True, name=name)
    attn_out = add_with_layer_scale_and_drop_block(inputs, nn, layer_scale=layer_scale, drop_rate=drop_rate, axis=-1, name=name + "attn_")

    nn = layer_norm(attn_out, axis=-1, name=name + "mlp_")
    nn = mlp_block(nn, input_channel * mlp_ratio, activation=activation, name=name)
    return add_with_layer_scale_and_drop_block(attn_out, nn, layer_scale=layer_scale, drop_rate=drop_rate, axis=-1, name=name + "mlp_")


def conv_block(inputs, mlp_ratio=4, layer_scale=0, drop_rate=0, activation="gelu", name=""):
    input_channel = inputs.shape[-1 if image_data_format() == "channels_last" else 1]

    nn = layers.AvgPool2D(pool_size=3, strides=1, padding="SAME")(inputs)  # count_include_pad=False [ ??? ]
    nn = nn - inputs
    attn_out = add_with_layer_scale_and_drop_block(inputs, nn, layer_scale=layer_scale, drop_rate=drop_rate, name=name + "attn_")

    nn = conv2d_no_bias(attn_out, input_channel * mlp_ratio, 1, strides=1, use_bias=True, name=name + "mlp_1_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "mlp_1_")
    nn = conv2d_no_bias(nn, input_channel, 1, strides=1, use_bias=True, name=name + "mlp_2_")
    nn = batchnorm_with_activation(nn, activation=None, name=name + "mlp_2_")
    return add_with_layer_scale_and_drop_block(attn_out, nn, layer_scale=layer_scale, drop_rate=drop_rate, name=name + "mlp_")


def EfficientFormer(
    num_blocks=[3, 2, 6, 4],
    out_channels=[48, 96, 224, 448],
    mlp_ratios=4,
    num_attn_blocks_each_stack=[0, 0, 0, 1],
    stem_width=-1,
    stem_activation="relu",
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
    # Regard input_shape as force using original shape if len(input_shape) == 4,
    # else assume channel dimention is the one with min value in input_shape, and put it first or last regarding image_data_format
    input_shape = backend.align_input_shape_by_image_data_format(input_shape)
    inputs = layers.Input(input_shape)
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

        cur_mlp_ratios = mlp_ratios[stack_id] if isinstance(mlp_ratios, (list, tuple)) else mlp_ratios
        cur_num_attn_blocks = num_attn_blocks_each_stack[stack_id] if isinstance(num_attn_blocks_each_stack, (list, tuple)) else num_attn_blocks_each_stack
        attn_block_start_id = num_block - cur_num_attn_blocks
        for block_id in range(num_block):
            block_name = stack_name + "block{}_".format(block_id + 1)
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            mlp_ratio = cur_mlp_ratios[block_id] if isinstance(cur_mlp_ratios, (list, tuple)) else cur_mlp_ratios
            if block_id >= attn_block_start_id:
                nn = layers.Permute([2, 3, 1])(nn) if block_id == attn_block_start_id and image_data_format() == "channels_first" else nn
                nn = attn_block(nn, mlp_ratio=mlp_ratio, layer_scale=layer_scale, drop_rate=block_drop_rate, activation=activation, name=block_name)
                nn = layers.Permute([3, 1, 2])(nn) if block_id == num_block - 1 and image_data_format() == "channels_first" else nn
            else:
                nn = conv_block(nn, mlp_ratio=mlp_ratio, layer_scale=layer_scale, drop_rate=block_drop_rate, activation=activation, name=block_name)

            global_block_id += 1

    """ output """
    if num_classes > 0:
        nn = layer_norm(nn, name="pre_output_")
        nn = layers.GlobalAveragePooling2D()(nn)  # tf.reduce_mean(nn, axis=1)
        if dropout > 0 and dropout < 1:
            nn = layers.Dropout(dropout)(nn)
        out = layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="head")(nn)

        if use_distillation:
            distill = layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="distill_head")(nn)
            out = [out, distill]
    else:
        out = nn

    model = models.Model(inputs, out, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    reload_model_weights(model, PRETRAINED_DICT, "efficientformer", pretrained, MultiHeadPositionalEmbedding)
    return model


def EfficientFormerL1(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", use_distillation=True, pretrained="imagenet", **kwargs):
    return EfficientFormer(**locals(), model_name="efficientformer_l1", **kwargs)


def EfficientFormerL3(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", use_distillation=True, pretrained="imagenet", **kwargs):
    num_blocks = [4, 4, 12, 6]
    out_channels = [64, 128, 320, 512]
    num_attn_blocks_each_stack = [0, 0, 0, 4]
    return EfficientFormer(**locals(), model_name="efficientformer_l3", **kwargs)


def EfficientFormerL7(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", use_distillation=True, pretrained="imagenet", **kwargs):
    num_blocks = [6, 6, 18, 8]
    out_channels = [96, 192, 384, 768]
    num_attn_blocks_each_stack = [0, 0, 0, 8]
    return EfficientFormer(**locals(), model_name="efficientformer_l7", **kwargs)
