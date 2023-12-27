from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models, functional, image_data_format
from keras_cv_attention_models.attention_layers import (
    activation_by_name,
    layer_norm,
    qkv_to_multi_head_channels_last_format,
    scaled_dot_product_attention,
    ClassToken,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

LAYER_NORM_EPSILON = 1e-5
PRETRAINED_DICT = {
    "sam_mask_decoder": {"sam": "86ccca20e41dd15578fbbd067035fa70"},
    "tiny_sam_mask_decoder": {"sam": "34f68eb047de515721f4658106e4ccb5"},
}


def mlp_block_multi(inputs, hidden_dim, output_channel=-1, num_blocks=2, use_bias=True, activation="gelu", name=None):
    output_channel = output_channel if output_channel > 0 else inputs.shape[-1]
    nn = inputs
    for id in range(num_blocks - 1):
        nn = layers.Dense(hidden_dim, use_bias=use_bias, name=name and name + "dense_{}".format(id + 1))(nn)
        nn = activation_by_name(nn, activation, name=name + "{}_".format(id + 1))
    nn = layers.Dense(output_channel, use_bias=use_bias, name=name and name + "dense_{}".format(num_blocks))(nn)
    return nn


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
    query, key, query_position, key_position, num_heads=8, downsample_rate=2, skip_first_layer_pe=False, mlp_ratio=8, activation="gelu", name=""
):
    if skip_first_layer_pe:
        query = attention(query, name=name + "query_")
    else:
        attn_out = attention(query + query_position, value=query, name=name + "query_")
        query = query + attn_out
    query = layer_norm(query, epsilon=LAYER_NORM_EPSILON, axis=-1, name=name + "query_")

    # Cross attention block, tokens attending to image embedding
    query_with_pe = query + query_position
    key_with_pe = key + key_position
    head_dim = query.shape[-1] // num_heads // downsample_rate
    attn_out = attention(query_with_pe, key=key_with_pe, value=key, num_heads=num_heads, head_dim=head_dim, name=name + "cross_embedding_")
    query = query + attn_out
    query = layer_norm(query, epsilon=LAYER_NORM_EPSILON, axis=-1, name=name + "cross_embedding_")

    # MLP block
    mlp_out = mlp_block_multi(query, hidden_dim=int(query.shape[-1] * mlp_ratio), num_blocks=2, activation=activation, name=name + "mlp_")
    query = query + mlp_out
    query = layer_norm(query, epsilon=LAYER_NORM_EPSILON, axis=-1, name=name + "mlp_")

    # Cross attention block, image embedding attending to tokens
    query_with_pe = query + query_position
    attn_out = attention(key_with_pe, key=query_with_pe, value=query, num_heads=num_heads, head_dim=head_dim, name=name + "cross_tokens_")
    key = key + attn_out
    key = layer_norm(key, epsilon=LAYER_NORM_EPSILON, axis=-1, name=name + "cross_tokens_")
    return query, key


def two_way_transformer(
    image_embedding, image_position, point_embedding, depth=2, num_heads=8, mlp_dim=2048, downsample_rate=2, mlp_ratio=8, activation="gelu", name=""
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
    query = layer_norm(query, epsilon=LAYER_NORM_EPSILON, axis=-1, name=name + "token_to_image_")

    key = functional.reshape(key, pre_shape)
    return query, key


def MaskDecoder(input_shape=(64, 64, 256), num_mask_tokens=4, activation="gelu", pretrained="sam", name="sam_mask_decoder"):
    image_embedding = layers.Input(input_shape, batch_size=1, name="image_embedding")  # Inputs is channels_last also for PyTorch backend
    point_embedding = layers.Input([None, input_shape[-1]], batch_size=1, name="point_embedding")
    image_position = layers.Input(input_shape, batch_size=1, name="image_position")
    embed_dims = input_shape[-1]

    point_embedding_with_tokens = ClassToken(num_tokens=num_mask_tokens + 1, name="cls_token")(point_embedding)
    iou_masks, encoded_image_embedding = two_way_transformer(image_embedding, image_position, point_embedding_with_tokens, activation="relu", name="attn_")
    # print(f"{iou_masks.shape = }, {encoded_image_embedding.shape = }")

    # output_upscaling
    nn = encoded_image_embedding if backend.image_data_format() == "channels_last" else layers.Permute([3, 1, 2])(encoded_image_embedding)
    nn = layers.Conv2DTranspose(embed_dims // 4, kernel_size=2, strides=2, name="up_1_conv_transpose")(nn)
    nn = layer_norm(nn, epsilon=1e-6, name="up_1_")  # epsilon is fixed using 1e-6
    nn = activation_by_name(nn, activation=activation, name="up_1_")
    nn = layers.Conv2DTranspose(embed_dims // 8, kernel_size=2, strides=2, name="up_2_conv_transpose")(nn)
    nn = activation_by_name(nn, activation=activation, name="up_2_")
    nn = layers.Permute([3, 1, 2])(nn) if backend.image_data_format() == "channels_last" else nn  # Put channels first
    pre_shape = functional.shape(nn)[2:] if None in nn.shape[2:] or -1 in nn.shape[2:] else nn.shape[2:]  # Could be dynamic shape, reshape back later
    upscaled_image_embedding = layers.Reshape([embed_dims // 8, -1])(nn)

    # iou_masks = functional.split(iou_masks, [num_mask_tokens + 1, -1], axis=1)[0]
    iou_masks = iou_masks[:, : num_mask_tokens + 1]
    iou_masks.set_shape([None, num_mask_tokens + 1, embed_dims])
    # iou_token_out, masks_top, masks_left, masks_bottom, masks_right = functional.unstack(iou_masks, axis=1)
    iou_pred = mlp_block_multi(iou_masks[:, 0], embed_dims, output_channel=num_mask_tokens, num_blocks=3, activation="relu", name="iou_pred_")

    hyper_in = []
    for id in range(num_mask_tokens):
        cur_mask, cur_name = iou_masks[:, id + 1], "masks_{}_".format(id + 1)
        hyper_in.append(mlp_block_multi(cur_mask, embed_dims, output_channel=embed_dims // 8, num_blocks=3, activation="relu", name=cur_name))
    # print(f"{[ii.shape for ii in hyper_in] = }")
    hyper_in = functional.stack(hyper_in, axis=1)

    # print(f"{hyper_in.shape = }, {upscaled_image_embedding.shape = }")
    masks = hyper_in @ upscaled_image_embedding  # [batch, 4, 32] @ [batch, 32, height * width] -> [batch, 4, height * width]
    # print(f"{masks.shape = }, {pre_shape = }")
    masks = functional.reshape(masks, [-1, num_mask_tokens, *pre_shape])  # [batch, 4, height * width] -> [batch, 4, height, width]

    model = models.Model([image_embedding, point_embedding, image_position], [masks, iou_pred], name=name)
    reload_model_weights(model, PRETRAINED_DICT, "segment_anything", pretrained)
    return model
