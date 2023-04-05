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
    HeadInitializer,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

LAYER_NORM_EPSILON = 1e-6

PRETRAINED_DICT = {
    "inceptionnext_base": {"imagenet": {224: "9b2bc220f5fc1cc9715daa9c899bff0b", 384: "c1740ea8e9115612eb12ce54c035ed32"}},
    "inceptionnext_small": {"imagenet": "046ec2140cce1620fd94cd1b208a9bb6"},
    "inceptionnext_tiny": {"imagenet": "1026a14bc0367f9d29a8d5bf952ac056"},
}


def inception_dwconv2d(inputs, square_kernel_size=3, band_kernel_size=11, branch_ratio=0.125, name=""):
    channel_axis = -1 if image_data_format() == "channels_last" else 1
    input_channel = inputs.shape[channel_axis]

    split_size = int(input_channel * branch_ratio)
    nn_short, nn_hw, nn_ww, nn_hh = functional.split(inputs, [-1, split_size, split_size, split_size], axis=channel_axis)

    nn_hw = depthwise_conv2d_no_bias(nn_hw, kernel_size=square_kernel_size, padding="SAME", use_bias=True, name=name + "hw_")
    nn_ww = depthwise_conv2d_no_bias(nn_ww, kernel_size=(1, band_kernel_size), padding="SAME", use_bias=True, name=name + "ww_")
    nn_hh = depthwise_conv2d_no_bias(nn_hh, kernel_size=(band_kernel_size, 1), padding="SAME", use_bias=True, name=name + "hh_")

    out = functional.concat([nn_short, nn_hw, nn_ww, nn_hh], axis=channel_axis)
    return out


def mixer_mlp_block(inputs, mlp_ratio=4, layer_scale=0.1, drop_rate=0, activation="gelu", name=""):
    mixer = inception_dwconv2d(inputs, name=name + "mixer_")
    mixer = batchnorm_with_activation(mixer, activation=None, name=name + "mixer_")

    input_channel = inputs.shape[-1 if image_data_format() == "channels_last" else 1]  # Channels_last only
    mlp = mlp_block(mixer, input_channel * mlp_ratio, use_conv=True, use_bias=True, activation=activation, name=name)
    return add_with_layer_scale_and_drop_block(inputs, mlp, layer_scale=layer_scale, drop_rate=drop_rate, name=name + "output_")


def InceptionNeXt(
    num_blocks=[3, 3, 9, 3],
    embed_dims=[96, 192, 384, 768],
    mlp_ratios=[4, 4, 4, 3],
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="gelu",
    drop_connect_rate=0,
    dropout=0,
    layer_scale=1e-6,
    classifier_activation="softmax",
    pretrained=None,
    model_name="inceptionnext",
    kwargs=None,
):
    # Regard input_shape as force using original shape if len(input_shape) == 4,
    # else assume channel dimention is the one with min value in input_shape, and put it first or last regarding image_data_format
    input_shape = backend.align_input_shape_by_image_data_format(input_shape)
    inputs = layers.Input(input_shape)

    """ Stem """
    nn = conv2d_no_bias(inputs, embed_dims[0], kernel_size=4, strides=4, padding="valid", use_bias=True, name="stem_")
    nn = batchnorm_with_activation(nn, activation=None, name="stem_")

    """ stacks """
    total_blocks = sum(num_blocks)
    global_block_id = 0
    for stack_id, (num_block, embed_dim) in enumerate(zip(num_blocks, embed_dims)):
        stack_name = "stack{}_".format(stack_id + 1)
        if stack_id > 0:
            nn = batchnorm_with_activation(nn, activation=None, name=stack_name + "downsample_")  # Using epsilon=1e-5
            nn = conv2d_no_bias(nn, embed_dim, 2, strides=2, padding="valid", use_bias=True, name=stack_name + "downsample_")

        mlp_ratio = mlp_ratios[stack_id] if isinstance(mlp_ratios, (list, tuple)) else mlp_ratios
        for block_id in range(num_block):
            name = stack_name + "block{}_".format(block_id + 1)
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            nn = mixer_mlp_block(nn, mlp_ratio, layer_scale, block_drop_rate, activation=activation, name=name)
            global_block_id += 1

    """  Output head """
    if num_classes > 0:
        nn = layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        head_init = HeadInitializer(scale=0.02)
        nn = layers.Dense(embed_dims[-1] * 3, use_bias=True, kernel_initializer=head_init, bias_initializer="zeros", name="pre_head_dense")(nn)
        nn = activation_by_name(nn, activation=activation, name="pre_head_")
        nn = layer_norm(nn, epsilon=LAYER_NORM_EPSILON, name="pre_head_")

        if dropout > 0:
            nn = layers.Dropout(dropout, name="head_drop")(nn)
        nn = layers.Dense(
            num_classes, dtype="float32", activation=classifier_activation, kernel_initializer=head_init, bias_initializer="zeros", name="predictions"
        )(nn)
    model = models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    reload_model_weights(model, PRETRAINED_DICT, "inceptionnext", pretrained)
    return model


def InceptionNeXtTiny(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return InceptionNeXt(**locals(), model_name="inceptionnext_tiny", **kwargs)


def InceptionNeXtSmall(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 3, 27, 3]
    return InceptionNeXt(**locals(), model_name="inceptionnext_small", **kwargs)


def InceptionNeXtBase(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 3, 27, 3]
    embed_dims = [128, 256, 512, 1024]
    return InceptionNeXt(**locals(), model_name="inceptionnext_base", **kwargs)
