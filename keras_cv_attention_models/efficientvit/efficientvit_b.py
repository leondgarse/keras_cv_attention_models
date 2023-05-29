from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models, functional, image_data_format
from keras_cv_attention_models.models import register_model
from keras_cv_attention_models.attention_layers import (
    activation_by_name,
    batchnorm_with_activation,
    conv2d_no_bias,
    depthwise_conv2d_no_bias,
    drop_block,
    layer_norm,
    qkv_to_multi_head_channels_last_format,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

PRETRAINED_DICT = {"": {"imagenet": {224: ""}}}


def MBConv(inputs, output_channel, shortcut=True, strides=1, expansion=4, use_bias=False, use_norm=True, drop_rate=0, activation="hard_swish", name=""):
    input_channel = inputs.shape[-1 if image_data_format() == "channels_last" else 1]
    if expansion > 1:
        nn = conv2d_no_bias(inputs, int(input_channel * expansion), 1, strides=1, use_bias=use_bias, name=name + "expand_")
        nn = batchnorm_with_activation(nn, activation=activation, name=name + "expand_") if use_norm else activation_by_name(nn, activation=activation)
    else:
        nn = inputs
    nn = depthwise_conv2d_no_bias(nn, 3, strides=strides, use_bias=use_bias, padding="same", name=name)
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "dw_") if use_norm else activation_by_name(nn, activation=activation)

    nn = conv2d_no_bias(nn, output_channel, 1, strides=1, use_bias=False, name=name + "pw_")
    nn = batchnorm_with_activation(nn, activation=None, zero_gamma=True, name=name + "pw_")
    nn = drop_block(nn, drop_rate=drop_rate, name=name)
    return layers.Add(name=name + "output")([inputs, nn]) if shortcut else nn

def lite_mhsa(inputs, num_heads=8, key_dim=16, sr_ratio=5, qkv_bias=False, out_shape=None, out_bias=False, dropout=0, activation="relu", name=None):
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    input_channel = inputs.shape[channel_axis]
    height, width = inputs.shape[1:-1] if image_data_format() == "channels_last" else inputs.shape[2:]
    key_dim = key_dim if key_dim > 0 else input_channel // num_heads
    out_shape = input_channel if out_shape is None else out_shape
    emb_dim = num_heads * key_dim

    # query = layers.Dense(emb_dim, use_bias=qkv_bias, name=name and name + "query")(inputs)
    qkv = conv2d_no_bias(inputs, emb_dim * 3, use_bias=qkv_bias, name=name and name + "qkv_")
    sr_qkv = depthwise_conv2d_no_bias(qkv, kernel_size=sr_ratio, use_bias=qkv_bias, padding="same", name=name and name + "qkv_")
    sr_qkv = conv2d_no_bias(sr_qkv, emb_dim * 3, use_bias=qkv_bias, groups=3 * num_heads, name=name and name + "qkv_pw_")
    qkv = functional.concat([qkv, sr_qkv], axis=channel_axis)

    query, key, value = functional.split(qkv, 3, axis=channel_axis)  # [TODO] <- num_heads * 3 * key_dim
    query = activation_by_name(query, activation=activation)
    key = activation_by_name(query, activation=activation)
    query, key, value = qkv_to_multi_head_channels_last_format(query, key, value, num_heads=num_heads)
    # print(f">>>> {inputs.shape = }, {query.shape = }, {sr_ratio = }")

    attn = query @ (key @ value)
    scale = query @ functional.reduce_sum(key, axis=-1, keepdims=True)  # [TODO] -> functional.reduce_sum(query @ key, axis=-1, keepdims=True) ???
    attention_output = attn / (scale + 1e-15)

    if image_data_format() == "channels_last":
        output = functional.transpose(attention_output, perm=[0, 2, 1, 3])  # [batch, q_blocks, num_heads, key_dim * attn_ratio]
        output = functional.reshape(output, [-1, height, width, output.shape[2] * output.shape[3]])
    else:
        output = functional.transpose(attention_output, perm=[0, 1, 3, 2])  # [batch, num_heads, key_dim * attn_ratio, q_blocks]
        output = functional.reshape(output, [-1, output.shape[1] * output.shape[2], height, width])
    output = conv2d_no_bias(output, out_shape, use_bias=out_bias, name=name and name + "out_")
    output = batchnorm_with_activation(output, activation=None, name=name and name + "output_")
    return output


def EfficientViT_B(
    num_blocks=[2, 3, 3, 4],
    out_channels=[32, 64, 128, 256],
    stem_width=16,
    block_types=["conv", "conv", "transform", "transform"],
    expansion=4,
    head_ratio=16,
    head_dimension=16,
    output_filters=[1536, 1600],
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="hard_swish",
    drop_connect_rate=0,
    classifier_activation="softmax",
    dropout=0,
    pretrained=None,
    model_name="efficientvit",
    kwargs=None,
):
    # Regard input_shape as force using original shape if len(input_shape) == 4,
    # else assume channel dimension is the one with min value in input_shape, and put it first or last regarding image_data_format
    input_shape = backend.align_input_shape_by_image_data_format(input_shape)
    inputs = layers.Input(input_shape)

    """ stage 0, Stem_stage """
    nn = conv2d_no_bias(inputs, stem_width, 3, strides=2, padding="same", name="stem_")
    nn = batchnorm_with_activation(nn, activation=activation, name="stem_")
    nn = MBConv(nn, stem_width, shortcut=True, expansion=1, activation=activation, name="stem_MB_")

    """ stage [1, 2, 3, 4] """
    total_blocks = sum(num_blocks)
    global_block_id = 0
    for stack_id, (num_block, out_channel, block_type) in enumerate(zip(num_blocks, out_channels, block_types)):
        is_conv_block = True if block_type[0].lower() == "c" else False
        for block_id in range(num_block):
            name = "stack_{}_block_{}_".format(stack_id + 1, block_id + 1)
            stride = 2 if block_id == 0 else 1
            shortcut = False if block_id == 0 else True
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks

            use_bias, use_norm = (False, True) if is_conv_block else (True, False)
            if is_conv_block or block_id == 0:
                cur_name = (name + "downsample_") if stride > 1 else name
                nn = MBConv(nn, out_channel, shortcut, stride, expansion, use_bias, use_norm, drop_rate=block_drop_rate, activation=activation, name=cur_name)
            else:
                num_heads = out_channel // head_ratio
                attn = lite_mhsa(nn, num_heads=num_heads, key_dim=head_dimension, sr_ratio=5, name=name)
                nn = nn + attn
                nn = MBConv(nn, out_channel, shortcut, stride, expansion, use_bias, use_norm, drop_rate=block_drop_rate, activation=activation, name=name)
            global_block_id += 1

    output_filters = output_filters if isinstance(output_filters, (list, tuple)) else (output_filters, 0)
    if output_filters[0] > 0:
        nn = conv2d_no_bias(nn, output_filters[0], name="features_")
        nn = batchnorm_with_activation(nn, activation=activation, name="features_")

    if num_classes > 0:
        nn = layers.GlobalAveragePooling2D(name="avg_pool")(nn) if len(nn.shape) == 4 else nn
        if dropout > 0:
            nn = layers.Dropout(dropout, name="head_drop")(nn)
        if output_filters[1] > 0:
            nn = layers.Dense(output_filters[1], use_bias=False, name="pre_predictions")(nn)
            nn = layer_norm(nn, name="pre_predictions_")
            nn = activation_by_name(nn, activation=activation, name="pre_predictions_")
        nn = layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="predictions")(nn)
    model = models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    reload_model_weights(model, PRETRAINED_DICT, "efficientvit", pretrained)
    return model


@register_model
def EfficientViT_B1(input_shape=(224, 224, 3), num_classes=1000, drop_connect_rate=0, classifier_activation="softmax", **kwargs):
    return EfficientViT_B(**locals(), model_name="efficientvit_b1", **kwargs)


@register_model
def EfficientViT_B2(input_shape=(224, 224, 3), num_classes=1000, drop_connect_rate=0, classifier_activation="softmax", **kwargs):
    out_channels = [48, 96, 192, 384]
    num_blocks = [3, 4, 4, 6]
    stem_width = 24
    head_dimension = 32
    output_filters = [2304, 2560]
    return EfficientViT_B(**locals(), model_name="efficientvit_b2", **kwargs)


@register_model
def EfficientViT_B3(input_shape=(224, 224, 3), num_classes=1000, drop_connect_rate=0, classifier_activation="softmax", **kwargs):
    out_channels = [64, 128, 256, 512]
    num_blocks = [4, 6, 6, 9]
    stem_width = 32
    head_dimension = 32
    output_filters = [2304, 2560]
    return EfficientViT_B(**locals(), model_name="efficientvit_b3", **kwargs)
