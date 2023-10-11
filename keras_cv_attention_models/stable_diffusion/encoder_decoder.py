import numpy as np
from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, functional, models, initializers, image_data_format
from keras_cv_attention_models.models import register_model
from keras_cv_attention_models.attention_layers import (
    activation_by_name,
    conv2d_no_bias,
    group_norm,
    add_pre_post_process,
    qkv_to_multi_head_channels_last_format,
    scaled_dot_product_attention,
)
from keras_cv_attention_models.stable_diffusion.unet import res_block
from keras_cv_attention_models.download_and_load import reload_model_weights

GROUP_NORM_EPSILON = 1e-6

PRETRAINED_DICT = {"decoder": {"v1_5": "3b37d81cd07fbab875bdbb48fbb91e27"}, "encoder": {"v1_5": "d053d1d1d43cba0f357381cf2710836a"}}


def gaussian_distribution(inputs):
    # [TODO] keras.layers.GaussianNoise
    channel_axis = -1 if backend.image_data_format() == "channels_last" else 1
    mean, log_var = functional.split(inputs, 2, axis=channel_axis)
    log_var = functional.clip_by_value(log_var, -30.0, 20.0)
    std = functional.exp(log_var * 0.5)

    gaussian = initializers.RandomNormal()(std.shape)
    return mean + std * gaussian


def attention_block(inputs, num_attention_block=1, mlp_ratio=4, num_heads=4, head_dim=0, name=""):
    input_channels = inputs.shape[-1 if backend.image_data_format() == "channels_last" else 1]
    inputs_shape = functional.shape(inputs) if None in inputs.shape[1:] or -1 in inputs.shape[1:] else [-1, *inputs.shape[1:]]
    qk_scale = 1.0 / (float(input_channels) ** 0.5)

    nn = group_norm(inputs, epsilon=GROUP_NORM_EPSILON, name=name + "in_layers_")

    qq = conv2d_no_bias(nn, input_channels, use_bias=True, name=name and name + "query_")
    kk = conv2d_no_bias(nn, input_channels, use_bias=True, name=name and name + "key_")
    vv = conv2d_no_bias(nn, input_channels, use_bias=True, name=name and name + "value_")

    if image_data_format() == "channels_last":
        qq = layers.Reshape([-1, qq.shape[-1]])(qq)
        kk = functional.transpose(layers.Reshape([-1, kk.shape[-1]])(kk), [0, 2, 1])
        vv = layers.Reshape([-1, vv.shape[-1]])(vv)
    else:
        qq = functional.transpose(layers.Reshape([qq.shape[1], -1])(qq), [0, 2, 1])
        kk = layers.Reshape([kk.shape[1], -1])(kk)
        vv = layers.Reshape([vv.shape[1], -1])(vv)

    attention_scores = (qq @ kk) * qk_scale
    attention_scores = layers.Softmax(axis=-1, name=name and name + "attention_scores")(attention_scores)

    if image_data_format() == "channels_last":
        attention_output = attention_scores @ vv
        output = functional.reshape(attention_output, inputs_shape)
    else:
        attention_output = vv @ functional.transpose(attention_scores, [0, 2, 1])
        output = functional.reshape(attention_output, inputs_shape)
    output = conv2d_no_bias(output, input_channels, use_bias=True, name=name and name + "out")
    return layers.Add(name=name + "out")([output, inputs])


@register_model
def Encoder(
    input_shape=(512, 512, 3),  # input -> DownSample 3 times -> // 8, 512 -> 64
    num_blocks=[2, 2, 2, 2],
    hidden_channels=128,
    hidden_expands=[1, 2, 4, 4],
    output_channels=4,
    post_encoder_channels=4,  # > 0 value for an additional Conv2D layer after output layer
    pretrained="v1_5",
    model_name="encoder",
    kwargs=None,  # Not using, recieving parameter
):
    # Regard input_shape as force using original shape if len(input_shape) == 4,
    # else assume channel dimension is the one with min value in input_shape, and put it first or last regarding image_data_format
    inputs = layers.Input(backend.align_input_shape_by_image_data_format(input_shape))
    channel_axis = -1 if backend.image_data_format() == "channels_last" else 1
    nn = conv2d_no_bias(inputs, hidden_channels, kernel_size=3, use_bias=True, padding="SAME", name="stem_")

    """ Down blocks """
    for stack_id, (num_block, hidden_expand) in enumerate(zip(num_blocks, hidden_expands)):
        stack_name = "stack{}_".format(stack_id + 1)
        out_channels = hidden_expand * hidden_channels
        if stack_id > 0:
            nn = conv2d_no_bias(nn, nn.shape[channel_axis], 3, 2, use_bias=True, padding="same", use_torch_padding=False, name=stack_name + "downsample_")

        for block_id in range(num_block):
            block_name = stack_name + "down_block{}_".format(block_id + 1)
            nn = res_block(nn, time_embedding=None, out_channels=out_channels, name=block_name)
    # print(f">>>> {[ii.shape for ii in skip_connections] = }")

    """ Middle blocks """
    nn = res_block(nn, time_embedding=None, name="middle_block_1_")
    nn = attention_block(nn, name="middle_block_attn_")
    nn = res_block(nn, time_embedding=None, name="middle_block_2_")

    """ Output blocks """
    nn = group_norm(nn, epsilon=GROUP_NORM_EPSILON, name="output_")
    nn = activation_by_name(nn, activation="swish", name="output_")
    # * 2 means [mean, std]
    outputs = conv2d_no_bias(nn, output_channels * 2, kernel_size=3, use_bias=True, padding="SAME", name="output_")
    outputs = conv2d_no_bias(outputs, 2 * post_encoder_channels, use_bias=True, name="post_encoder_") if post_encoder_channels > 0 else outputs

    model = models.Model(inputs, outputs, name=model_name)
    reload_model_weights(model, PRETRAINED_DICT, "stable_diffusion", pretrained)
    return model


@register_model
def Decoder(
    input_shape=(64, 64, 4),
    num_blocks=[2, 2, 2, 2],
    hidden_channels=128,
    hidden_expands=[4, 4, 2, 1],
    output_channels=3,
    pre_decoder_channels=4,  # > 0 value for an additional Conv2D layer after input layer
    pretrained="v1_5",
    model_name="decoder",
    kwargs=None,  # Not using, recieving parameter
):
    # Regard input_shape as force using original shape if len(input_shape) == 4,
    # else assume channel dimension is the one with min value in input_shape, and put it first or last regarding image_data_format
    inputs = layers.Input(backend.align_input_shape_by_image_data_format(input_shape))
    channel_axis = -1 if backend.image_data_format() == "channels_last" else 1
    nn = conv2d_no_bias(inputs, pre_decoder_channels, use_bias=True, name="pre_decoder_") if pre_decoder_channels > 0 else inputs
    nn = conv2d_no_bias(nn, hidden_channels * hidden_expands[0], kernel_size=3, use_bias=True, padding="SAME", name="stem_")

    """ Middle blocks """
    nn = res_block(nn, time_embedding=None, epsilon=GROUP_NORM_EPSILON, name="middle_block_1_")
    nn = attention_block(nn, name="middle_block_attn_")
    nn = res_block(nn, time_embedding=None, epsilon=GROUP_NORM_EPSILON, name="middle_block_2_")

    """ Up blocks """
    for stack_id, (num_block, hidden_expand) in enumerate(zip(num_blocks, hidden_expands)):
        stack_name = "stack{}_".format(stack_id + 1)
        out_channels = hidden_expand * hidden_channels
        if stack_id > 0:
            nn = layers.UpSampling2D(size=2, name=stack_name + "upsample_")(nn)
            nn = conv2d_no_bias(nn, nn.shape[channel_axis], kernel_size=3, strides=1, use_bias=True, padding="same", name=stack_name + "upsample_")

        for block_id in range(num_block + 1):
            block_name = stack_name + "up_block{}_".format(block_id + 1)
            nn = res_block(nn, time_embedding=None, epsilon=GROUP_NORM_EPSILON, out_channels=out_channels, name=block_name)

    """ Output blocks """
    nn = group_norm(nn, epsilon=GROUP_NORM_EPSILON, name="output_")
    nn = activation_by_name(nn, activation="swish", name="output_")
    outputs = conv2d_no_bias(nn, output_channels, kernel_size=3, use_bias=True, padding="SAME", name="output_")

    model = models.Model(inputs, outputs, name=model_name)
    reload_model_weights(model, PRETRAINED_DICT, "stable_diffusion", pretrained)
    return model
