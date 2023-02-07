from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, functional, initializers
from keras_cv_attention_models.aotnet import AotNet
from keras_cv_attention_models.attention_layers import (
    RelativePositionalEmbedding,
    conv2d_no_bias,
    CompatibleExtractPatches,
    make_divisible,
    scaled_dot_product_attention,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

PRETRAINED_DICT = {
    "halonet26t": {"imagenet": {256: "6e8848ce6e98a13cd45f65dd68435d00"}},
    "halonet50t": {"imagenet": {256: "6c353144df942cb81a3eb7c140fc9791"}},
    "halonet_se33t": {"imagenet": {256: "7e0afb7f8fb6459491b8a46ad80bcd91"}},
    "halonext_eca26t": {"imagenet": {256: "630037a5c135bceacd0691a22855eb7e"}},
    "haloregnetz_b": {"imagenet": {224: "e889647682d1c554de032d376acf0c48"}},
    "halobotnet50t": {"imagenet": {256: "0af1faad1a81e468d6e670e9fc253edc"}},
}


def halo_attention(
    inputs, num_heads=8, key_dim=0, block_size=4, halo_size=1, strides=1, out_shape=None, out_weight=True, out_bias=False, attn_dropout=0, name=None
):
    _, hh, ww, cc = inputs.shape
    if key_dim > 1:
        key_dim = key_dim  # Specified one
    elif key_dim > 0:
        key_dim = make_divisible(cc * key_dim, divisor=8) // num_heads  # regard as key_dim_ratio
    else:
        key_dim = cc // num_heads  # Default value
    # qk_scale = float(1.0 / tf.math.sqrt(tf.cast(key_dim, "float32")))
    qk_scale = 1.0 / (float(key_dim) ** 0.5)
    out_shape = cc if out_shape is None else out_shape
    emb_dim = num_heads * key_dim
    kv_kernel = block_size + halo_size * 2
    if block_size % strides != 0:
        strides = 1
        avg_pool_down = True
    else:
        avg_pool_down = False
    query_block = block_size // strides

    query = conv2d_no_bias(inputs, emb_dim, kernel_size=1, strides=strides, name=name and name + "query_")
    _, hh, ww, cc = query.shape
    # print(f">>>> {inputs.shape = }, {query.shape = }, {block_size = }, {strides = }")
    # attn_query = rearrange(query, "B (h hb) (w wb) (hd c) -> B hd h w (hb wb) c", hb=query_block, wb=query_block, hd=num_heads)
    # pos_query = rearrange(attn_query, "B hd h w (hb wb) c -> B (hd h w) hb wb c", hb=query_block, wb=query_block)
    hh_qq, ww_qq, cc_qq = hh // query_block, ww // query_block, cc // num_heads
    query = functional.reshape(query, [-1, hh_qq, query_block, ww_qq, query_block, num_heads, cc_qq])
    query = functional.transpose(query, [0, 5, 1, 3, 2, 4, 6])  # [batch, num_heads, hh, ww, query_block, query_block, key_dim]
    # attn_query = [batch, num_heads, hh, ww, query_block * query_block, key_dim]
    attn_query = functional.reshape(query, [-1, num_heads, hh_qq, ww_qq, query_block * query_block, cc_qq]) * qk_scale  # qk_scale NOT multiplied with pos_query
    # pos_query = [batch, num_heads * hh * ww, query_block, query_block, key_dim]
    pos_query = functional.reshape(query, [-1, num_heads * hh_qq * ww_qq, query_block, query_block, cc_qq])

    # key_value = [batch, height, width, key_dim + out_shape]
    key_value = conv2d_no_bias(inputs, emb_dim + out_shape, kernel_size=1, use_bias=False, name=name and name + "key_value_")
    kv_padded = functional.pad(key_value, [[0, 0], [halo_size, halo_size], [halo_size, halo_size], [0, 0]])
    sizes, strides = [1, kv_kernel, kv_kernel, 1], [1, block_size, block_size, 1]
    # kv_inp = [batch, hh, ww, kv_kernel * kv_kernel * (key_dim + out_shape)]
    # kv_inp = tf.image.extract_patches(kv_padded, sizes=sizes, strides=strides, rates=[1, 1, 1, 1], padding="VALID")
    kv_inp = CompatibleExtractPatches(sizes=sizes, strides=strides, rates=[1, 1, 1, 1], padding="VALID")(kv_padded)
    # kv_inp = rearrange(kv_inp, "B h w (hb wb hd c) -> B hd h w (hb wb) c", hb=kv_kernel, wb=kv_kernel, hd=num_heads)
    _, hh_kk, ww_kk, cc = kv_inp.shape
    cc_kk = cc // num_heads // kv_kernel // kv_kernel
    kv_inp = functional.reshape(kv_inp, [-1, hh_kk, ww_kk, kv_kernel, kv_kernel, num_heads, cc_kk])
    kv_inp = functional.transpose(kv_inp, [0, 5, 1, 2, 3, 4, 6])
    kv_inp = functional.reshape(kv_inp, [-1, num_heads, hh_kk, ww_kk, kv_kernel * kv_kernel, cc_kk])

    # key = [batch, num_heads, hh, ww, kv_kernel * kv_kernel, key_dim]
    # value = [batch, num_heads, hh, ww, kv_kernel * kv_kernel, out_dim]
    key, value = functional.split(kv_inp, [emb_dim // num_heads, out_shape // num_heads], axis=-1)

    # scaled_dot_product_attention
    # print(f">>>> {attn_query.shape = }, {key.shape = }, {value.shape = }, {kv_inp.shape = }, {pos_query.shape = }, {num_heads = }")
    # attention_scores = [batch, num_heads, hh, ww, query_block * query_block, kv_kernel * kv_kernel]
    # attention_scores = layers.Lambda(lambda xx: functional.matmul(xx[0], xx[1], transpose_b=True))([attn_query, key])
    attention_scores = attn_query @ functional.transpose(key, [0, 1, 2, 3, 5, 4])
    # pos = [batch, num_heads * hh * ww, query_block, query_block, kv_kernel, kv_kernel]
    pos = RelativePositionalEmbedding(position_height=kv_kernel, name=name and name + "pos_emb")(pos_query)
    # print(f">>>> {pos.shape = }, {attention_scores.shape = }")
    pos = functional.reshape(pos, [-1, *attention_scores.shape[1:]])
    attention_scores = layers.Add()([attention_scores, pos])
    # attention_scores = tf.nn.softmax(attention_scores, axis=-1)
    attention_scores = layers.Softmax(axis=-1, name=name and name + "attention_scores")(attention_scores)

    if attn_dropout > 0:
        attention_scores = layers.Dropout(attn_dropout, name=name and name + "attn_drop")(attention_scores)

    # attention_output = [batch, num_heads, hh, ww, query_block * query_block, out_dim]
    # attention_output = layers.Lambda(lambda xx: functional.matmul(xx[0], xx[1]))([attention_scores, value])
    attention_output = attention_scores @ value
    # attention_output = rearrange(attention_output, "B hd h w (hb wb) c -> B (h hb) (w wb) (hd c)", hb=query_block, wb=query_block)
    _, heads, hh_aa, ww_aa, patch, cc_aa = attention_output.shape
    attention_output = functional.reshape(attention_output, [-1, heads, hh_aa, ww_aa, query_block, query_block, cc_aa])
    attention_output = functional.transpose(attention_output, [0, 2, 4, 3, 5, 1, 6])
    attention_output = functional.reshape(attention_output, [-1, hh_aa * query_block, ww_aa * query_block, heads * cc_aa])
    # print(f">>>> {attention_output.shape = }, {attention_scores.shape = }")

    if avg_pool_down:
        attention_output = layers.AvgPool2D(2, strides=2, name=name and name + "avg_pool")(attention_output)
    if out_weight:
        # [batch, hh, ww, num_heads * out_dim] * [out, out] --> [batch, hh, ww, out]
        attention_output = layers.Dense(out_shape, use_bias=out_bias, name=name and name + "output")(attention_output)
    return attention_output


BLOCK_CONFIGS = {
    "h0": {  # rv = 1, rb = 0.5
        "halo_block_size": 8,  # b
        "halo_halo_size": 3,  # h
        "halo_expansion": 1,  # rv
        "expansion": 0.5,  # rb
        "output_num_features": -1,  # df
        "num_blocks": [3, 3, 7, 3],
        "num_heads": [4, 8, 8, 8],
    },
    "h1": {  # rv = 1, rb = 1
        "halo_block_size": 8,
        "halo_halo_size": 3,
        "halo_expansion": 1,  # rv
        "expansion": 1,  # rb
        "output_num_features": -1,  # df
        "num_blocks": [3, 3, 10, 3],
        "num_heads": [4, 8, 8, 8],
    },
    "h2": {  # rv = 1, rb = 1.25
        "halo_block_size": 8,
        "halo_halo_size": 3,
        "halo_expansion": 1,  # rv
        "expansion": 1.25,  # rb
        "output_num_features": -1,  # df
        "num_blocks": [3, 3, 11, 3],
        "num_heads": [4, 8, 8, 8],
    },
    "h3": {  # rv = 1, rb = 1.5
        "halo_block_size": 10,
        "halo_halo_size": 3,
        "halo_expansion": 1,  # rv
        "expansion": 1.5,  # rb
        "output_num_features": 1024,  # df
        "num_blocks": [3, 3, 12, 3],
        "num_heads": [4, 8, 8, 8],
    },
    "h4": {  # rv = 1, rb = 3
        "halo_block_size": 12,
        "halo_halo_size": 2,
        "halo_expansion": 1,  # rv
        "expansion": 3,  # rb
        "output_num_features": 1280,  # df
        "num_blocks": [3, 3, 12, 3],
        "num_heads": [4, 8, 8, 8],
    },
    "h5": {  # rv = 2.5, rb = 2
        "halo_block_size": 14,
        "halo_halo_size": 2,
        "halo_expansion": 2.5,  # rv
        "expansion": 2,  # rb
        "output_num_features": 1536,  # df
        "num_blocks": [3, 3, 23, 3],
        "num_heads": [4, 8, 8, 8],
    },
    "h6": {  # rv = 3, rb = 2.75
        "halo_block_size": 8,
        "halo_halo_size": 4,
        "halo_expansion": 3,  # rv
        "expansion": 2.75,  # rb
        "output_num_features": 1536,  # df
        "num_blocks": [3, 3, 24, 3],
        "num_heads": [4, 8, 8, 8],
    },
    "h7": {  # rv = 4, rb = 3.5
        "halo_block_size": 10,
        "halo_halo_size": 3,
        "halo_expansion": 4,  # rv
        "num_heads": [4, 8, 8, 8],
        "expansion": 3.5,  # rb
        "output_num_features": 2048,  # df
        "num_blocks": [3, 3, 26, 3],
    },
}


def HaloNet(
    input_shape=(256, 256, 3),
    activation="swish",
    expansion=4,
    halo_block_size=4,
    halo_halo_size=1,
    halo_expansion=1,
    num_heads=8,
    pretrained=None,
    request_resolution=256,
    **kwargs
):
    attn_types = "halo"
    if isinstance(num_heads, (list, tuple)):
        attn_params = [
            {"block_size": halo_block_size, "halo_size": halo_halo_size, "halo_expansion": halo_expansion, "num_heads": hh, "out_weight": False}
            for hh in num_heads
        ]
    else:
        attn_params = {
            "block_size": halo_block_size,
            "halo_size": halo_halo_size,
            "halo_expansion": halo_expansion,
            "num_heads": num_heads,
            "out_weight": False,
        }
    out_channels = [ii * expansion for ii in [64, 128, 256, 512]]
    hidden_channel_ratio = 1 / expansion
    model = AotNet(
        input_shape=input_shape,
        out_channels=out_channels,
        hidden_channel_ratio=hidden_channel_ratio,
        activation=activation,
        attn_types=attn_types,
        attn_params=attn_params,
        **kwargs
    )
    reload_model_weights(model, PRETRAINED_DICT, "halonet", pretrained, RelativePositionalEmbedding)
    return model


def HaloNetH0(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return HaloNet(**BLOCK_CONFIGS["h0"], model_name="haloneth0", request_resolution=256, **locals(), **kwargs)


def HaloNetH1(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return HaloNet(**BLOCK_CONFIGS["h1"], model_name="haloneth1", request_resolution=256, **locals(), **kwargs)


def HaloNetH2(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return HaloNet(**BLOCK_CONFIGS["h2"], model_name="haloneth2", request_resolution=256, **locals(), **kwargs)


def HaloNetH3(input_shape=(320, 320, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return HaloNet(**BLOCK_CONFIGS["h3"], model_name="haloneth3", request_resolution=320, **locals(), **kwargs)


def HaloNetH4(input_shape=(384, 384, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return HaloNet(**BLOCK_CONFIGS["h4"], model_name="haloneth4", request_resolution=384, **locals(), **kwargs)


def HaloNetH5(input_shape=(448, 448, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return HaloNet(**BLOCK_CONFIGS["h5"], model_name="haloneth5", request_resolution=448, **locals(), **kwargs)


def HaloNetH6(input_shape=(512, 512, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return HaloNet(**BLOCK_CONFIGS["h6"], model_name="haloneth6", request_resolution=512, **locals(), **kwargs)


def HaloNetH7(input_shape=(600, 600, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    # input_shape should be divisible by `int(tf.reduce_prod(strides) * halo_block_size)`, may using 640 here
    return HaloNet(**BLOCK_CONFIGS["h7"], model_name="haloneth7", request_resolution=600, **locals(), **kwargs)


def HaloNet26T(input_shape=(256, 256, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 2, 2, 2]
    attn_types = [None, None, [None, "halo"], "halo"]
    attn_params = [
        None,
        None,
        [None, {"block_size": 8, "halo_size": 2, "num_heads": 8, "out_weight": False}],
        {"block_size": 8, "halo_size": 2, "num_heads": 8, "out_weight": False},
    ]
    # key_dim = 16
    stem_type = "tiered"
    model = AotNet(model_name="halonet26t", **locals(), **kwargs)
    reload_model_weights(model, PRETRAINED_DICT, "halonet", pretrained, RelativePositionalEmbedding)
    return model


def HaloNet50T(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 6, 3]
    attn_types = [None, [None, None, None, "halo"], [None, "halo"] * 3, [None, "halo", None]]
    attn_params = [
        None,
        [None, None, None, {"block_size": 8, "halo_size": 3, "num_heads": 4, "out_weight": False}],
        [None, {"block_size": 8, "halo_size": 3, "num_heads": 8, "out_weight": False}] * 3,
        [None, {"block_size": 8, "halo_size": 3, "num_heads": 8, "out_weight": False}, None],
    ]
    # key_dim = 16
    stem_type = "tiered"
    model = AotNet(model_name="halonet50t", **locals(), **kwargs)
    reload_model_weights(model, PRETRAINED_DICT, "halonet", pretrained, RelativePositionalEmbedding)
    return model


def HaloNetSE33T(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 3, 3, 2]
    attn_types = [None, [None, None, "halo"], [None, None, "halo"], "halo"]
    attn_params = [
        None,
        [None, None, {"block_size": 8, "halo_size": 3, "num_heads": 8, "out_weight": False}],
        [None, None, {"block_size": 8, "halo_size": 3, "num_heads": 8, "out_weight": False}],
        {"block_size": 8, "halo_size": 3, "num_heads": 8, "out_weight": False},
    ]
    se_ratio = 1 / 16
    out_channels = [256, 512, 1024, 1536]
    hidden_channel_ratio = [1 / 4, 1 / 4, 1 / 4, 1 / 3]
    # key_dim = 16
    stem_type = "tiered"
    stem_last_strides = 2
    stem_downsample = False
    output_num_features = 1280
    model = AotNet(model_name="halonet_se33t", **locals(), **kwargs)
    reload_model_weights(model, PRETRAINED_DICT, "halonet", pretrained, RelativePositionalEmbedding)
    return model


def HaloNextECA26T(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 2, 2, 2]
    attn_types = [None, None, [None, "halo"], "halo"]
    attn_params = [
        None,
        None,
        [None, {"block_size": 8, "halo_size": 2, "num_heads": 8, "key_dim": 16, "out_weight": False}],
        {"block_size": 8, "halo_size": 2, "num_heads": 8, "key_dim": 16, "out_weight": False},
    ]
    use_eca = True
    groups = [4, 8, 16, 32]
    stem_type = "tiered"
    model = AotNet(model_name="halonext_eca26t", **locals(), **kwargs)
    reload_model_weights(model, PRETRAINED_DICT, "halonet", pretrained, RelativePositionalEmbedding)
    return model


def HaloRegNetZB(input_shape=(224, 224, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [2, 6, 12, 2]
    strides = [2, 2, 2, 2]
    out_channels = [48, 96, 192, 288]
    # timm bottle_in=True mode, the first ratio in each stack is `ratio * previous_channel`
    hidden_channel_ratio = [[32 * 3 / 48, 3], [1.5] + [3] * 5, [1.5] + [3] * 11, [192 * 3 / 288, 3]]
    use_block_output_activation = False  # timm linear_out=True mode
    stem_type = "kernel_3x3"
    stem_width = 32
    stem_downsample = False
    attn_types = [None, None, [None, None, None, "halo"] * 3, "halo"]
    # "activation": "relu" is for se module.
    attn_params = {"block_size": 7, "halo_size": 2, "num_heads": 8, "key_dim": 0.33, "out_weight": False, "activation": "relu"}
    se_ratio = 0.25
    group_size = 16
    shortcut_type = None
    output_num_features = 1536
    model = AotNet(model_name="haloregnetz_b", **locals(), **kwargs)
    reload_model_weights(model, PRETRAINED_DICT, "halonet", pretrained, RelativePositionalEmbedding)
    return model


def HaloBotNet50T(input_shape=(256, 256, 3), num_classes=1000, activation="swish", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 6, 3]
    attn_types = [None, [None, "halo"] * 2, [None, "halo"] * 3, [None, "bot", None]]
    attn_params = [
        None,
        [None, {"block_size": 8, "halo_size": 3, "num_heads": 8, "out_weight": False}] * 2,
        [None, {"block_size": 8, "halo_size": 3, "num_heads": 8, "out_weight": False}] * 3,
        [None, {"num_heads": 4, "out_weight": False}, None],
    ]
    # key_dim = 16
    stem_type = "tiered"
    stem_last_strides = 2
    stem_downsample = False
    model = AotNet(model_name="halobotnet50t", **locals(), **kwargs)
    reload_model_weights(model, PRETRAINED_DICT, "halonet", pretrained, RelativePositionalEmbedding)
    return model
