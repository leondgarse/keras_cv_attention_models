from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models, functional, image_data_format
from keras_cv_attention_models.attention_layers import (
    activation_by_name,
    add_with_layer_scale_and_drop_block,
    addaptive_pooling_2d,
    conv2d_no_bias,
    depthwise_conv2d_no_bias,
    layer_norm,
    output_block,
    scaled_dot_product_attention,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

LAYER_NORM_EPSILON = 1e-6

PRETRAINED_DICT = {
    "pvt_v2_b0": {"imagenet": "f7af8430bec6c6b7c71e160f295288ce"},
    "pvt_v2_b1": {"imagenet": "91d70e33baa69338c8c831dbf36caeee"},
    "pvt_v2_b2": {"imagenet": "4902059a9c61661466c2102e070973ff"},
    "pvt_v2_b2_linear": {"imagenet": "0bd4864b6fc63419cbde100e594f7180"},
    "pvt_v2_b3": {"imagenet": "97d0e12d53898b0f2c66efb30a4b9de8"},
    "pvt_v2_b4": {"imagenet": "166f6323c6fe7a578970efc9cc8e5b17"},
    "pvt_v2_b5": {"imagenet": "43cf0deec87e4d7c25f3f7d57d49b917"},
}


def attention_block_with_conv_down(
    inputs, num_heads=4, key_dim=0, sr_ratio=1, qkv_bias=True, out_bias=True, use_linear=False, linear_activation="gelu", dropout=0, name=""
):
    _, hh, ww, input_channel = inputs.shape
    key_dim = key_dim if key_dim > 0 else input_channel // num_heads
    # out_shape = input_channel if out_shape is None or not out_weight else out_shape
    emb_dim = num_heads * key_dim

    query = layers.Dense(emb_dim, use_bias=qkv_bias, name=name and name + "query")(inputs)
    # print(f">>>> {inputs.shape = }, {query.shape = }, {sr_ratio = }")
    # [batch, num_heads, hh * ww, key_dim]
    query = functional.transpose(functional.reshape(query, [-1, inputs.shape[1] * inputs.shape[2], num_heads, key_dim]), [0, 2, 1, 3])

    if use_linear:
        key_value = inputs if image_data_format() == "channels_last" else layers.Permute([3, 1, 2], name=name + "permute_pre")(inputs)
        key_value = addaptive_pooling_2d(key_value, output_size=7, reduce="mean")
        key_value = conv2d_no_bias(key_value, input_channel, kernel_size=1, use_bias=qkv_bias, name=name + "kv_sr_")
        key_value = key_value if image_data_format() == "channels_last" else layers.Permute([2, 3, 1], name=name + "permute_post")(key_value)
        key_value = layer_norm(key_value, axis=-1, name=name + "kv_sr_")  # Using epsilon=1e-5
        key_value = activation_by_name(key_value, activation=linear_activation, name=name + "kv_sr_")
    elif sr_ratio > 1:
        key_value = inputs if image_data_format() == "channels_last" else layers.Permute([3, 1, 2], name=name + "permute_pre")(inputs)
        key_value = conv2d_no_bias(key_value, input_channel, kernel_size=sr_ratio, strides=sr_ratio, use_bias=qkv_bias, name=name + "kv_sr_")
        key_value = key_value if image_data_format() == "channels_last" else layers.Permute([2, 3, 1], name=name + "permute_post")(key_value)
        key_value = layer_norm(key_value, axis=-1, name=name + "kv_sr_")  # Using epsilon=1e-5
        # key_value = layers.AvgPool2D(sr_ratio, strides=sr_ratio, name=name + "kv_sr_")(inputs)
    else:
        key_value = inputs
    _, kv_hh, kv_ww, _ = key_value.shape
    # key_value = [batch, num_heads, hh, ww, kv_kernel * kv_kernel, key_dim * 2]
    key_value = layers.Dense(emb_dim * 2, use_bias=qkv_bias, name=name and name + "key_value")(key_value)
    key, value = functional.split(key_value, 2, axis=-1)
    key = functional.transpose(functional.reshape(key, [-1, kv_hh * kv_ww, num_heads, key_dim]), [0, 2, 3, 1])  # [batch, num_heads, key_dim, hh * ww]
    value = functional.transpose(functional.reshape(value, [-1, kv_hh * kv_ww, num_heads, key_dim]), [0, 2, 1, 3])  # [batch, num_heads, hh * ww, key_dim]

    output_shape = (hh, ww, input_channel)
    output = scaled_dot_product_attention(query, key, value, output_shape=output_shape, out_weight=False, dropout=dropout, name=name)
    return layers.Dense(input_channel, use_bias=out_bias, name=name and name + "out")(output)


def mlp_block_with_depthwise_conv(inputs, hidden_dim, kernel_size=3, use_bias=True, drop_rate=0, activation="gelu", name=""):
    input_channel = inputs.shape[-1]
    first_activation, middle_activation = activation if isinstance(activation, (list, tuple)) else (activation, activation)
    nn = layers.Dense(hidden_dim, use_bias=use_bias, name=name and name + "1_dense")(inputs)
    nn = activation_by_name(nn, first_activation, name=name)

    nn = nn if image_data_format() == "channels_last" else layers.Permute([3, 1, 2], name=name + "permute_pre")(nn)
    nn = depthwise_conv2d_no_bias(nn, use_bias=use_bias, kernel_size=kernel_size, strides=1, padding="same", name=name and name + "mid_")
    nn = nn if image_data_format() == "channels_last" else layers.Permute([2, 3, 1], name=name + "permute_post")(nn)
    nn = activation_by_name(nn, middle_activation, name=name and name + "mid_")
    nn = layers.Dropout(drop_rate)(nn) if drop_rate > 0 else nn

    nn = layers.Dense(input_channel, use_bias=use_bias, name=name and name + "2_dense")(nn)
    nn = layers.Dropout(drop_rate)(nn) if drop_rate > 0 else nn
    return nn


def attention_mlp_block(inputs, embed_dim, num_heads=8, sr_ratio=1, mlp_ratio=4, use_linear=False, layer_scale=0.1, drop_rate=0, activation="gelu", name=""):
    channnel_axis = -1 if image_data_format() == "channels_last" else 1
    input_channel = inputs.shape[channnel_axis]

    """ attention """
    pre = inputs if image_data_format() == "channels_last" else layers.Permute([2, 3, 1], name=name + "permute_pre")(inputs)
    nn = layer_norm(pre, epsilon=LAYER_NORM_EPSILON, axis=-1, name=name + "attn_")
    nn = attention_block_with_conv_down(nn, num_heads=num_heads, sr_ratio=sr_ratio, use_linear=use_linear, linear_activation=activation, name=name + "attn_")
    attn_out = add_with_layer_scale_and_drop_block(pre, nn, layer_scale=layer_scale, drop_rate=drop_rate, axis=-1, name=name + "attn_")

    """ MLP """
    nn = layer_norm(attn_out, epsilon=LAYER_NORM_EPSILON, axis=-1, name=name + "mlp_")
    mlp_activation = ("relu", activation) if use_linear else (None, activation)
    nn = mlp_block_with_depthwise_conv(nn, input_channel * mlp_ratio, activation=mlp_activation, name=name + "mlp_")
    nn = add_with_layer_scale_and_drop_block(attn_out, nn, layer_scale=layer_scale, drop_rate=drop_rate, axis=-1, name=name + "mlp_")
    nn = nn if image_data_format() == "channels_last" else layers.Permute([3, 1, 2], name=name + "permute_post")(nn)  # channels_first -> channels_last
    return nn


def PyramidVisionTransformerV2(
    num_blocks=[2, 2, 2, 2],
    embed_dims=[64, 128, 320, 512],
    num_heads=[1, 2, 5, 8],
    mlp_ratios=[8, 8, 4, 4],
    sr_ratios=[8, 4, 2, 1],
    stem_patch_size=7,
    use_linear=False,
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="gelu",
    drop_connect_rate=0,
    dropout=0,
    layer_scale=0,
    classifier_activation="softmax",
    pretrained=None,
    model_name="pvt",
    kwargs=None,
):
    # Regard input_shape as force using original shape if len(input_shape) == 4,
    # else assume channel dimention is the one with min value in input_shape, and put it first or last regarding image_data_format
    input_shape = backend.align_input_shape_by_image_data_format(input_shape)
    inputs = layers.Input(input_shape)

    """ Stem """
    nn = conv2d_no_bias(inputs, embed_dims[0], stem_patch_size, strides=4, padding="same", use_bias=True, name="stem_")
    nn = layer_norm(nn, name="stem_")  # Using epsilon=1e-5

    """ stacks """
    total_blocks = sum(num_blocks)
    global_block_id = 0
    for stack_id, (num_block, embed_dim) in enumerate(zip(num_blocks, embed_dims)):
        stack_name = "stack{}_".format(stack_id + 1)
        if stack_id > 0:
            nn = conv2d_no_bias(nn, embed_dim, 3, strides=2, padding="same", use_bias=True, name=stack_name + "downsample_")
            nn = layer_norm(nn, name=stack_name + "downsample_")  # Using epsilon=1e-5

        stack_num_head = num_heads[stack_id] if isinstance(num_heads, (list, tuple)) else num_heads
        stack_mlp_ratio = mlp_ratios[stack_id] if isinstance(mlp_ratios, (list, tuple)) else mlp_ratios
        stack_sr_ratio = sr_ratios[stack_id] if isinstance(sr_ratios, (list, tuple)) else sr_ratios
        for block_id in range(num_block):
            name = stack_name + "block{}_".format(block_id + 1)
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            nn = attention_mlp_block(nn, embed_dim, stack_num_head, stack_sr_ratio, stack_mlp_ratio, use_linear, layer_scale, block_drop_rate, activation, name)
            global_block_id += 1
        nn = layer_norm(nn, epsilon=LAYER_NORM_EPSILON, name=name + "output_")

    nn = output_block(nn, num_classes=num_classes, drop_rate=dropout, classifier_activation=classifier_activation)
    model = models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    reload_model_weights(model, PRETRAINED_DICT, "pvt", pretrained)
    return model


def PVT_V2B0(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    embed_dims = [32, 64, 160, 256]
    return PyramidVisionTransformerV2(**locals(), model_name="pvt_v2_b0", **kwargs)


def PVT_V2B1(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return PyramidVisionTransformerV2(**locals(), model_name="pvt_v2_b1", **kwargs)


def PVT_V2B2(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 6, 3]
    return PyramidVisionTransformerV2(**locals(), model_name="pvt_v2_b2", **kwargs)


def PVT_V2B2_linear(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 6, 3]
    use_linear = True
    return PyramidVisionTransformerV2(**locals(), model_name="pvt_v2_b2_linear", **kwargs)


def PVT_V2B3(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 18, 3]
    return PyramidVisionTransformerV2(**locals(), model_name="pvt_v2_b3", **kwargs)


def PVT_V2B4(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 8, 27, 3]
    return PyramidVisionTransformerV2(**locals(), model_name="pvt_v2_b4", **kwargs)


def PVT_V2B5(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 6, 40, 3]
    mlp_ratios = 4
    return PyramidVisionTransformerV2(**locals(), model_name="pvt_v2_b5", **kwargs)
