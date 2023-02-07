from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models, functional
from keras_cv_attention_models.attention_layers import (
    activation_by_name,
    batchnorm_with_activation,
    conv2d_no_bias,
    drop_block,
    group_norm,
    # mlp_block, # cannot import name 'mlp_block' due to circular import
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

PRETRAINED_DICT = {
    "wavemlp_t": {"imagenet": "c8fe3c22c129180c5cff7b734ede831c"},
    "wavemlp_s": {"imagenet": "b0375b030d1c3012232286cff76d301d"},
    "wavemlp_m": {"imagenet": "d2364156144ab259069026b417ca21da"},
}


def mlp_block(inputs, hidden_dim, out_channel, drop_rate=0, activation="gelu", name=""):
    nn = layers.Conv2D(hidden_dim, kernel_size=1, use_bias=True, name=name and name + "Conv_0")(inputs)
    nn = activation_by_name(nn, activation, name=name and name + activation)
    nn = layers.Dropout(drop_rate) if drop_rate > 0 else nn
    nn = layers.Conv2D(out_channel, kernel_size=1, use_bias=True, name=name and name + "Conv_1")(nn)
    nn = layers.Dropout(drop_rate) if drop_rate > 0 else nn
    return nn


def phase_aware_token_mixing(inputs, out_channel=-1, qkv_bias=False, output_dropout=0, activation="gelu", name=None):
    input_channel = inputs.shape[-1] if backend.image_data_format() == "channels_last" else inputs.shape[1]
    out_channel = out_channel if out_channel > 0 else input_channel

    theta_h = conv2d_no_bias(inputs, out_channel, kernel_size=1, use_bias=True, name=name and name + "theta_h_")
    theta_h = batchnorm_with_activation(theta_h, activation="relu", name=name and name + "theta_h_")  # Fixed as relu [ ??? ]
    height = conv2d_no_bias(inputs, out_channel, kernel_size=1, use_bias=qkv_bias, name=name and name + "height_")
    # height = layers.Concatenate(axis=-1)([height * functional.cos(theta_h), height * functional.sin(theta_h)])
    height_cos = layers.Multiply()([height, functional.cos(theta_h)])
    height_sin = layers.Multiply()([height, functional.sin(theta_h)])
    height = layers.Concatenate(axis=-1 if backend.image_data_format() == "channels_last" else 1)([height_cos, height_sin])
    height = conv2d_no_bias(height, out_channel, kernel_size=(1, 7), padding="SAME", groups=out_channel, use_bias=False, name=name and name + "height_down_")

    theta_w = conv2d_no_bias(inputs, out_channel, kernel_size=1, use_bias=True, name=name and name + "theta_w_")
    theta_w = batchnorm_with_activation(theta_w, activation="relu", name=name and name + "theta_w_")  # Fixed as relu [ ??? ]
    width = conv2d_no_bias(inputs, out_channel, kernel_size=1, use_bias=qkv_bias, name=name and name + "width_")
    # width = layers.Concatenate(axis=-1)([width * functional.cos(theta_w), width * functional.sin(theta_w)])
    width_cos = layers.Multiply()([width, functional.cos(theta_w)])
    width_sin = layers.Multiply()([width, functional.sin(theta_w)])
    width = layers.Concatenate(axis=-1 if backend.image_data_format() == "channels_last" else 1)([width_cos, width_sin])
    width = conv2d_no_bias(width, out_channel, kernel_size=(7, 1), padding="SAME", groups=out_channel, use_bias=False, name=name and name + "width_down_")

    channel = conv2d_no_bias(inputs, out_channel, kernel_size=1, use_bias=qkv_bias, name=name and name + "channel_")

    # print(f"{height.shape = }, {width.shape = }, {channel.shape = }, {out_channel = }")
    nn = layers.Add(name=name and name + "combine")([height, width, channel])
    nn = layers.GlobalAveragePooling2D(keepdims=True)(nn)
    nn = mlp_block(nn, out_channel // 4, out_channel=out_channel * 3, activation=activation, name=name and name + "reweight_")
    nn = layers.Reshape([1, 1, out_channel, 3] if backend.image_data_format() == "channels_last" else [out_channel, 1, 1, 3])(nn)
    nn = layers.Softmax(axis=-1, name=name and name + "attention_scores")(nn)
    attn_height, attn_width, attn_channel = functional.unstack(nn, axis=-1)
    # attn = layers.Add()([height * attn_height, width * attn_width, channel * attn_channel])
    attn_height = layers.Multiply()([height, attn_height])
    attn_width = layers.Multiply()([width, attn_width])
    attn_channel = layers.Multiply()([channel, attn_channel])
    # print(f"{attn_height.shape = }, {attn_width.shape = }, {attn_channel.shape = }")
    attn = layers.Add()([attn_height, attn_width, attn_channel])

    out = conv2d_no_bias(attn, out_channel, kernel_size=1, use_bias=True, name=name and name + "out_")
    out = layers.Dropout(output_dropout, name=name and name + "out_drop")(out) if output_dropout > 0 else out
    return out


def wave_block(inputs, qkv_bias=False, mlp_ratio=4, use_group_norm=False, drop_rate=0, activation="gelu", name=""):
    input_channel = inputs.shape[-1] if backend.image_data_format() == "channels_last" else inputs.shape[1]

    attn = group_norm(inputs, groups=1, name=name + "attn_") if use_group_norm else batchnorm_with_activation(inputs, activation=None, name=name + "attn_")
    attn = phase_aware_token_mixing(attn, out_channel=input_channel, qkv_bias=qkv_bias, activation=activation, name=name + "attn_")
    attn = drop_block(attn, drop_rate=drop_rate, name=name + "attn_")
    attn_out = layers.Add(name=name + "attn_out")([inputs, attn])

    mlp = group_norm(attn_out, groups=1, name=name + "mlp_") if use_group_norm else batchnorm_with_activation(attn_out, activation=None, name=name + "mlp_")
    mlp = mlp_block(mlp, int(input_channel * mlp_ratio), out_channel=input_channel, activation=activation, name=name + "mlp_")
    mlp = drop_block(mlp, name=name + "mlp_")
    mlp_out = layers.Add(name=name + "mlp_out")([attn_out, mlp])
    return mlp_out


def WaveMLP(
    num_blocks=[2, 2, 4, 2],
    out_channels=[64, 128, 320, 512],
    stem_width=-1,
    mlp_ratios=[4, 4, 4, 4],
    use_downsample_norm=True,
    use_group_norm=False,
    qkv_bias=False,
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="gelu",
    sam_rho=0,
    drop_connect_rate=0,
    classifier_activation="softmax",
    dropout=0,
    pretrained=None,
    model_name="wavemlp",
    kwargs=None,
):
    # Regard input_shape as force using original shape if len(input_shape) == 4,
    # else assume channel dimention is the one with min value in input_shape, and put it first or last regarding image_data_format
    input_shape = backend.align_input_shape_by_image_data_format(input_shape)
    inputs = layers.Input(input_shape)
    stem_width = stem_width if stem_width > 0 else out_channels[0]
    nn = layers.ZeroPadding2D(padding=2, name="stem_pad")(inputs)
    nn = conv2d_no_bias(nn, stem_width, kernel_size=7, strides=4, padding="valid", use_bias=True, name="stem_")
    if use_downsample_norm:
        nn = group_norm(nn, groups=1, name="stem_") if use_group_norm else batchnorm_with_activation(nn, activation=None, name="stem_")

    """ stage [1, 2, 3, 4] """
    total_blocks = sum(num_blocks)
    global_block_id = 0
    for stack_id, (num_block, out_channel, mlp_ratio) in enumerate(zip(num_blocks, out_channels, mlp_ratios)):
        stage_name = "stack{}_".format(stack_id + 1)
        if stack_id > 0:
            nn = conv2d_no_bias(nn, out_channel, kernel_size=3, strides=2, padding="same", use_bias=True, name=stage_name + "down_sample_")
            if use_downsample_norm:
                norm_name = stage_name + "down_sample_"
                nn = group_norm(nn, groups=1, name=norm_name) if use_group_norm else batchnorm_with_activation(nn, activation=None, name=norm_name)
        for block_id in range(num_block):
            name = stage_name + "block{}_".format(block_id + 1)
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            global_block_id += 1
            nn = wave_block(nn, qkv_bias, mlp_ratio, use_group_norm, block_drop_rate, activation=activation, name=name)

    if num_classes > 0:
        nn = group_norm(nn, groups=1, name="output_") if use_group_norm else batchnorm_with_activation(nn, activation=None, name="output_")
        nn = layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        if dropout > 0:
            nn = layers.Dropout(dropout, name="head_drop")(nn)
        nn = layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="predictions")(nn)

    if sam_rho != 0:
        from keras_cv_attention_models.model_surgery import SAMModel

        model = SAMModel(inputs, nn, name=model_name)
    else:
        model = models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    reload_model_weights(model, PRETRAINED_DICT, "mlp_family", pretrained)
    return model


def WaveMLP_T(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 2, 4, 2]
    out_channels = [64, 128, 320, 512]
    return WaveMLP(**locals(), model_name="wavemlp_t", **kwargs)


def WaveMLP_S(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 3, 10, 3]
    out_channels = [64, 128, 320, 512]
    use_group_norm = True
    return WaveMLP(**locals(), model_name="wavemlp_s", **kwargs)


def WaveMLP_M(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 18, 3]
    out_channels = [64, 128, 320, 512]
    mlp_ratios = [8, 8, 4, 4]
    use_group_norm = True
    use_downsample_norm = False
    return WaveMLP(**locals(), model_name="wavemlp_m", **kwargs)


def WaveMLP_B(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained=None, **kwargs):
    num_blocks = [2, 2, 18, 2]
    out_channels = [96, 192, 384, 768]
    use_group_norm = True
    use_downsample_norm = False
    return WaveMLP(**locals(), model_name="wavemlp_b", **kwargs)
