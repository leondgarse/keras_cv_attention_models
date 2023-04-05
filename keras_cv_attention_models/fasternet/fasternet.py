from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models, functional, image_data_format
from keras_cv_attention_models.attention_layers import (
    activation_by_name,
    add_with_layer_scale_and_drop_block,
    batchnorm_with_activation,
    conv2d_no_bias,
    HeadInitializer,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

LAYER_NORM_EPSILON = 1e-6
PRETRAINED_DICT = {
    "fasternet_l": {"imagenet": "42de42d7d4405716e575e25aa58e7445"},
    "fasternet_m": {"imagenet": "2de1a10a5aa1092f82cd6f0104b29c6d"},
    "fasternet_s": {"imagenet": "58f97e2986da1b6793ee4466b4421a16"},
    "fasternet_t0": {"imagenet": "c330a6fa902f17993eba5d734c822551"},
    "fasternet_t1": {"imagenet": "1eb9cb6c77542f5f485efc65c75fa780"},
    "fasternet_t2": {"imagenet": "b4edf4df9e261766fb2e17f0bb50651b"},
}


def block(inputs, mlp_ratio=2, partial_conv_ratio=0.25, layer_scale=1e-6, drop_rate=0, activation="gelu", name=""):
    channel_axis = -1 if backend.image_data_format() == "channels_last" else 1
    input_channel = inputs.shape[channel_axis]
    conv_branch_channels = int(input_channel * partial_conv_ratio)

    conv_branch, non_conv_branch = functional.split(inputs, [conv_branch_channels, -1], axis=channel_axis)
    conv_branch = conv2d_no_bias(conv_branch, conv_branch_channels, kernel_size=3, strides=1, padding="SAME", name=name + "partial")
    nn = functional.concat([conv_branch, non_conv_branch], axis=channel_axis)

    mlp = conv2d_no_bias(nn, int(input_channel * mlp_ratio), kernel_size=1, name=name + "mlp_1_")
    mlp = batchnorm_with_activation(mlp, activation=activation, name=name + "mlp_")
    mlp = conv2d_no_bias(mlp, input_channel, kernel_size=1, name=name + "mlp_2_")
    return add_with_layer_scale_and_drop_block(inputs, mlp, layer_scale=layer_scale, drop_rate=drop_rate, name=name)


def FasterNet(
    num_blocks=[1, 2, 8, 2],
    embed_dim=40,
    patch_size=4,
    mlp_ratio=2,
    partial_conv_ratio=0.25,
    output_conv_filter=1280,
    layer_scale=0,  # > 0 for applying layer_scale, 0 for not using
    head_init_scale=0.02,
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="gelu",
    drop_connect_rate=0,
    classifier_activation="softmax",
    dropout=0,
    pretrained=None,
    model_name="fasternet",
    kwargs=None,
):
    # Regard input_shape as force using original shape if len(input_shape) == 4,
    # else assume channel dimention is the one with min value in input_shape, and put it first or last regarding image_data_format
    input_shape = backend.align_input_shape_by_image_data_format(input_shape)
    inputs = layers.Input(input_shape)

    """ Stem """
    nn = conv2d_no_bias(inputs, embed_dim, kernel_size=patch_size, strides=patch_size, padding="VALID", name="stem_")
    nn = batchnorm_with_activation(nn, activation=None, name="stem_")

    """ Blocks """
    total_blocks = sum(num_blocks)
    global_block_id = 0
    for stack_id, num_block in enumerate(num_blocks):
        stack_name = "stack{}_".format(stack_id + 1)
        if stack_id > 0:
            input_channel = nn.shape[-1] if backend.image_data_format() == "channels_last" else nn.shape[1]
            nn = conv2d_no_bias(nn, input_channel * 2, kernel_size=2, strides=2, name=stack_name + "downsample_")
            nn = batchnorm_with_activation(nn, activation=None, name=stack_name + "downsample_")
        for block_id in range(num_block):
            block_name = stack_name + "block{}_".format(block_id + 1)
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            nn = block(nn, mlp_ratio, partial_conv_ratio, layer_scale, block_drop_rate, activation, name=block_name)
            global_block_id += 1

    """  Output head """
    if num_classes > 0:
        nn = layers.GlobalAveragePooling2D(name="avg_pool", keepdims=True)(nn)
        nn = conv2d_no_bias(nn, output_conv_filter, 1, strides=1, padding="valid", name="post_")
        nn = activation_by_name(nn, activation=activation, name="post_")
        nn = layers.Flatten()(nn)

        if dropout > 0:
            nn = layers.Dropout(dropout, name="head_drop")(nn)
        head_init = HeadInitializer(scale=head_init_scale)
        nn = layers.Dense(
            num_classes, dtype="float32", activation=classifier_activation, kernel_initializer=head_init, bias_initializer=head_init, name="predictions"
        )(nn)

    model = models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    reload_model_weights(model, pretrained_dict=PRETRAINED_DICT, sub_release="fasternet", pretrained=pretrained)
    return model


def FasterNetT0(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return FasterNet(**locals(), model_name="fasternet_t0", **kwargs)


def FasterNetT1(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    embed_dim = 64
    return FasterNet(**locals(), model_name="fasternet_t1", **kwargs)


def FasterNetT2(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    embed_dim = 96
    return FasterNet(**locals(), model_name="fasternet_t2", **kwargs)


def FasterNetS(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    embed_dim = 128
    num_blocks = [1, 2, 13, 2]
    return FasterNet(**locals(), model_name="fasternet_s", **kwargs)


def FasterNetM(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    embed_dim = 144
    num_blocks = [3, 4, 18, 3]
    return FasterNet(**locals(), model_name="fasternet_m", **kwargs)


def FasterNetL(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    embed_dim = 192
    num_blocks = [3, 4, 18, 3]
    return FasterNet(**locals(), model_name="fasternet_l", **kwargs)
