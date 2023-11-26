import math
import numpy as np
from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models, functional, image_data_format, initializers
from keras_cv_attention_models.models import register_model
from keras_cv_attention_models.attention_layers import (
    activation_by_name,
    add_with_layer_scale_and_drop_block,
    batchnorm_with_activation,
    conv2d_no_bias,
    depthwise_conv2d_no_bias,
    inverted_residual_block,
    layer_norm,
    mlp_block,
    multi_head_self_attention,
    mhsa_with_multi_head_position,
    qkv_to_multi_head_channels_last_format,
    scaled_dot_product_attention,
    window_attention,
    ClassToken,
    MultiHeadPositionalEmbedding,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights


def attention(query, key=None, value=None, num_heads=8, head_dim=0, name=""):
    key = query if key is None else key
    value = query if value is None else value
    # print(f"{query.shape = }, {key.shape = }, {value.shape = }, {name = }")
    _, bb, cc = query.shape
    head_dim = head_dim if head_dim > 0 else cc // num_heads
    emded_dim = int(num_heads * head_dim)

    query = layers.Dense(emded_dim, name=name and name + "query")(query)
    key = layers.Dense(emded_dim, name=name and name + "key")(key)
    value = layers.Dense(emded_dim, name=name and name + "value")(value)

    query, key, value = qkv_to_multi_head_channels_last_format(query, key, value, num_heads, data_format="channels_last")
    output_shape = [-1, -1 if bb is None else bb, cc]
    return scaled_dot_product_attention(query, key, value, output_shape, out_weight=True, out_bias=True, name=name)


def two_way_attention_block(
    query, key, query_position, key_position, num_heads=8, downsample_rate=2, skip_first_layer_pe=False, mlp_ratio=8, activation="relu", name=""
):
    if skip_first_layer_pe:
        query = attention(query, name=name + "query_")
    else:
        attn_out = attention(query + query_position, value=query, name=name + "query_")
        query = query + attn_out
    query = layer_norm(query, axis=-1, name=name + "query_")

    # Cross attention block, tokens attending to image embedding
    query_with_pe = query + query_position
    key_with_pe = key + key_position
    head_dim = query.shape[-1] // num_heads // downsample_rate
    attn_out = attention(query_with_pe, key=key_with_pe, value=key, num_heads=num_heads, head_dim=head_dim, name=name + "cross_embedding_")
    query = query + attn_out
    query = layer_norm(query, axis=-1, name=name + "cross_embedding_")

    # MLP block
    mlp_out = mlp_block(query, hidden_dim=int(query.shape[-1] * mlp_ratio), activation=activation, name=name + "mlp_")
    query = query + mlp_out
    query = layer_norm(query, axis=-1, name=name + "mlp_")

    # Cross attention block, image embedding attending to tokens
    query_with_pe = query + query_position
    attn_out = attention(key_with_pe, key=query_with_pe, value=query, num_heads=num_heads, head_dim=head_dim, name=name + "cross_tokens_")
    key = key + attn_out
    key = layer_norm(key, axis=-1, name=name + "cross_tokens_")
    return query, key


def two_way_transformer(
    image_embedding, image_position, point_embedding, depth=2, num_heads=8, mlp_dim=2048, downsample_rate=2, mlp_ratio=8, activation="relu", name=""
):
    query, query_position, key, key_position = point_embedding, point_embedding, image_embedding, image_position

    # key = key if backend.image_data_format() == "channels_last" else layers.Permute([2, 3, 1])(key)
    pre_shape = functional.shape(key) if None in key.shape[1:] or -1 in key.shape[1:] else [-1, *key.shape[1:]]  # Could be dynamic shape, reshape back later
    key = layers.Reshape([-1, key.shape[-1]])(key)
    key_position = layers.Reshape([-1, key_position.shape[-1]])(key_position)

    for ii in range(depth):
        skip_first_layer_pe = True if ii == 0 else False
        query, key = two_way_attention_block(
            query, key, query_position, key_position, num_heads, downsample_rate, skip_first_layer_pe, mlp_ratio, activation, name=name + "{}_".format(ii)
        )

    query_with_pe = query + query_position
    key_with_pe = key + key_position
    head_dim = query.shape[-1] // num_heads // downsample_rate
    attn_out = attention(query_with_pe, key=key_with_pe, value=key, num_heads=num_heads, head_dim=head_dim, name=name + "token_to_image_")
    query = query + attn_out
    query = layer_norm(query, axis=-1, name=name + "token_to_image_")

    key = functional.reshape(key, pre_shape)
    return query, key


def mlp_block_3(inputs, hidden_dim, output_channel=-1, use_bias=True, activation="relu", name=None):
    output_channel = output_channel if output_channel > 0 else inputs.shape[-1]
    nn = layers.Dense(hidden_dim, use_bias=use_bias, name=name and name + "dense_1")(inputs)
    nn = activation_by_name(nn, activation, name=name + "1_")
    nn = layers.Dense(hidden_dim, use_bias=use_bias, name=name and name + "dense_2")(nn)
    nn = activation_by_name(nn, activation, name=name + "2_")
    nn = layers.Dense(output_channel, use_bias=use_bias, name=name and name + "dense_3")(nn)
    return nn


def MaskDecoder(embed_dims=256, num_mask_tokens=4, activation="relu", name="mask_decoder"):
    image_embedding = layers.Input([None, None, embed_dims], batch_size=1, name="image_embedding")  # Inputs is channels_last also for PyTorch backend
    point_embedding = layers.Input([None, embed_dims], batch_size=1, name="point_embedding")
    image_position = layers.Input([None, None, embed_dims], batch_size=1, name="image_position")

    point_embedding_with_tokens = ClassToken(num_tokens=num_mask_tokens + 1, name="cls_token")(point_embedding)
    iou_masks, encoded_image_embedding = two_way_transformer(image_embedding, image_position, point_embedding_with_tokens, name="attn_")
    # print(f"{iou_masks.shape = }, {encoded_image_embedding.shape = }")

    # output_upscaling
    nn = encoded_image_embedding if backend.image_data_format() == "channels_last" else layers.Permute([3, 1, 2])(encoded_image_embedding)
    nn = layers.Conv2DTranspose(embed_dims // 4, kernel_size=2, strides=2, name="up_1_conv_transpose")(nn)
    nn = layer_norm(nn, name="up_1_")
    nn = activation_by_name(nn, activation=activation, name="up_1_")
    nn = layers.Conv2DTranspose(embed_dims // 8, kernel_size=2, strides=2, name="up_2_conv_transpose")(nn)
    nn = activation_by_name(nn, activation=activation, name="up_2_")
    nn = layers.Permute([3, 1, 2])(nn) if backend.image_data_format() == "channels_last" else nn  # Put channels first
    pre_shape = functional.shape(nn)[2:] if None in nn.shape[2:] or -1 in nn.shape[2:] else nn.shape[2:]  # Could be dynamic shape, reshape back later
    upscaled_image_embedding = layers.Reshape([embed_dims // 8, -1])(nn)

    iou_masks = functional.split(iou_masks, [5, -1], axis=1)[0]
    iou_token_out, masks_top, masks_left, masks_bottom, masks_right = functional.unstack(iou_masks, axis=1)
    iou_pred = mlp_block_3(iou_token_out, hidden_dim=embed_dims, output_channel=num_mask_tokens, activation=activation, name="iou_pred_")

    hyper_in = []
    for id, (ii, name) in enumerate(zip([masks_top, masks_left, masks_bottom, masks_right], ["top", "left", "bottom", "right"])):
        hyper_in.append(mlp_block_3(ii, hidden_dim=embed_dims, output_channel=embed_dims // 8, activation=activation, name="masks_" + name + "_"))
    # print(f"{[ii.shape for ii in hyper_in] = }")
    hyper_in = functional.stack(hyper_in, axis=1)

    # print(f"{hyper_in.shape = }, {upscaled_image_embedding.shape = }")
    masks = hyper_in @ upscaled_image_embedding  # [batch, 4, 32] @ [batch, 32, height * width] -> [batch, 4, height * width]
    # print(f"{masks.shape = }")
    # [batch, 4, height * width] -> [batch, 4, height, width] -> [batch, height, width, 4], outputs channels_last also for PyTorch backend
    masks = layers.Permute([2, 3, 1])(functional.reshape(masks, [-1, masks.shape[1], *pre_shape]))
    return models.Model([image_embedding, point_embedding, image_position], [masks, iou_pred], name=name)
