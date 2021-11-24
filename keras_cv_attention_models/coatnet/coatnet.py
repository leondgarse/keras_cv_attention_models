import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from keras_cv_attention_models.attention_layers import (
    activation_by_name,
    batchnorm_with_activation,
    conv2d_no_bias,
    depthwise_conv2d_no_bias,
    drop_block,
    layer_norm,
    se_module,
    MultiHeadRelativePositionalEmbedding,
)


def mhsa_with_multi_head_relative_position_embedding(
    inputs, num_heads=4, key_dim=0, relative=True, out_shape=None, out_weight=True, out_bias=False, attn_dropout=0, name=None
):
    _, hh, ww, cc = inputs.shape
    key_dim = key_dim if key_dim > 0 else cc // num_heads
    qk_scale = 1.0 / tf.math.sqrt(tf.cast(key_dim, inputs.dtype))
    out_shape = cc if out_shape is None or not out_weight else out_shape
    qk_out = num_heads * key_dim
    vv_dim = out_shape // num_heads

    # qkv = keras.layers.Dense(emb_dim * 3, use_bias=False, name=name and name + "qkv")(inputs)
    qkv = conv2d_no_bias(inputs, qk_out * 2 + out_shape, kernel_size=1, name=name and name + "qkv_")
    if vv_dim == key_dim:
        qkv = tf.reshape(qkv, [-1, inputs.shape[1] * inputs.shape[2], 3, num_heads, key_dim])
        query, key, value = tf.transpose(qkv, [2, 0, 3, 1, 4])
        key = tf.transpose(key, [0, 1, 3, 2])
    else:
        qkv = tf.reshape(qkv, [-1, inputs.shape[1] * inputs.shape[2], qkv.shape[-1]])
        query, key, value = tf.split(qkv, [qk_out, qk_out, out_shape], axis=-1)
        # query = [batch, num_heads, hh * ww, key_dim]
        query = tf.transpose(tf.reshape(query, [-1, query.shape[1], num_heads, key_dim]), [0, 2, 1, 3])
        # key = [batch, num_heads, key_dim, hh * ww]
        key = tf.transpose(tf.reshape(key, [-1, key.shape[1], num_heads, key_dim]), [0, 2, 3, 1])
        # value = [batch, num_heads, hh * ww, vv_dim]
        value = tf.transpose(tf.reshape(value, [-1, value.shape[1], num_heads, vv_dim]), [0, 2, 1, 3])

    # query *= qk_scale
    # [batch, num_heads, hh * ww, hh * ww]
    attention_scores = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([query, key]) * qk_scale
    attention_scores = MultiHeadRelativePositionalEmbedding(with_cls_token=False, name=name and name + "pos_emb")(attention_scores)
    # attention_scores = tf.nn.softmax(attention_scores, axis=-1)
    attention_scores = keras.layers.Softmax(axis=-1, name=name and name + "attention_scores")(attention_scores)

    if attn_dropout > 0:
        attention_scores = keras.layers.Dropout(attn_dropout, name=name and name + "attn_drop")(attention_scores)
    # value = [batch, num_heads, hh * ww, vv_dim]
    # attention_output = tf.matmul(attention_scores, value)  # [batch, num_heads, hh * ww, vv_dim]
    attention_output = keras.layers.Lambda(lambda xx: tf.matmul(xx[0], xx[1]))([attention_scores, value])
    attention_output = tf.transpose(attention_output, perm=[0, 2, 1, 3])
    attention_output = tf.reshape(attention_output, [-1, inputs.shape[1], inputs.shape[2], num_heads * vv_dim])
    # print(f">>>> {attention_output.shape = }, {attention_scores.shape = }")

    if out_weight:
        # [batch, hh, ww, num_heads * vv_dim] * [num_heads * vv_dim, out] --> [batch, hh, ww, out]
        attention_output = keras.layers.Dense(out_shape, use_bias=out_bias, name=name and name + "output")(attention_output)
    return attention_output


def res_MBConv(inputs, output_channel, conv_short_cut=True, strides=1, expansion=4, se_ratio=0, drop_rate=0, activation="gelu", name=""):
    """ x ← Proj(Pool(x)) + Conv (DepthConv (Conv (Norm(x), stride = 2)))) """
    # preact
    preact = batchnorm_with_activation(inputs, activation=activation, zero_gamma=False, name=name + "preact_")

    if conv_short_cut:
        # Avg or Max pool
        # shortcut = keras.layers.AvgPool2D(strides, strides=strides, padding="SAME", name=name + "shortcut_pool")(inputs) if strides > 1 else inputs
        shortcut = keras.layers.AvgPool2D(strides, strides=strides, padding="SAME", name=name + "shortcut_pool")(preact) if strides > 1 else preact
        shortcut = conv2d_no_bias(shortcut, output_channel, 1, strides=1, name=name + "shortcut_")
    else:
        shortcut = inputs

    # MBConv
    input_channel = inputs.shape[-1]
    nn = conv2d_no_bias(preact, input_channel * expansion, 1, strides=1, padding="same", name=name + "expand_")  # May swap stirdes with DW
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "expand_")
    nn = depthwise_conv2d_no_bias(nn, 3, strides=strides, padding="same", name=name + "MB_")
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "MB_dw_")
    if se_ratio:
        nn = se_module(nn, se_ratio=se_ratio / expansion, activation=activation, name=name + "se_")
    nn = conv2d_no_bias(nn, output_channel, 1, strides=1, padding="same", name=name + "MB_pw_")
    nn = drop_block(nn, drop_rate=drop_rate, name=name)
    return keras.layers.Add()([shortcut, nn])


def res_ffn(inputs, expansion=4, kernel_size=1, drop_rate=0, activation="gelu", name=""):
    """ x ← x + Module (Norm(x)) """
    # preact
    preact = batchnorm_with_activation(inputs, activation=activation, zero_gamma=False, name=name + "preact_")
    # nn = layer_norm(inputs, name=name + "preact_")

    input_channel = inputs.shape[-1]
    nn = conv2d_no_bias(preact, input_channel * expansion, kernel_size, name=name + "1_")
    # nn = activation_by_name(nn, activation=activation, name=name)
    nn = batchnorm_with_activation(nn, activation=activation, name=name + "ffn_")
    nn = conv2d_no_bias(nn, input_channel, kernel_size, name=name + "2_")
    nn = drop_block(nn, drop_rate=drop_rate, name=name)
    return keras.layers.Add()([inputs, nn])


def res_mhsa(inputs, output_channel, conv_short_cut=True, strides=1, head_dimension=32, drop_rate=0, activation="gelu", name=""):
    """ x ← Proj(Pool(x)) + Attention (Pool(Norm(x))) """
    # preact
    preact = batchnorm_with_activation(inputs, activation=activation, zero_gamma=False, name=name + "preact_")
    # preact = layer_norm(inputs, name=name + "preact_")

    if conv_short_cut:
        # Avg or Max pool
        # shortcut = keras.layers.AvgPool2D(strides, strides=strides, padding="SAME", name=name + "shortcut_pool")(inputs) if strides > 1 else inputs
        shortcut = keras.layers.AvgPool2D(strides, strides=strides, padding="SAME", name=name + "shortcut_pool")(preact) if strides > 1 else preact
        shortcut = conv2d_no_bias(shortcut, output_channel, 1, strides=1, name=name + "shortcut_")
    else:
        shortcut = inputs

    nn = preact
    if strides != 1:  # Downsample
        # nn = keras.layers.ZeroPadding2D(padding=1, name=name + "pad")(nn)
        nn = keras.layers.MaxPool2D(pool_size=2, strides=strides, padding="SAME", name=name + "pool")(nn)
    num_heads = nn.shape[-1] // head_dimension
    nn = mhsa_with_multi_head_relative_position_embedding(nn, num_heads=num_heads, key_dim=head_dimension, out_shape=output_channel, name=name + "mhsa")
    nn = drop_block(nn, drop_rate=drop_rate, name=name)
    # print(f"{name = }, {inputs.shape = }, {shortcut.shape = }, {nn.shape = }")
    return keras.layers.Add()([shortcut, nn])


def CoAtNet(
    num_blocks,
    out_channels,
    stem_width=64,
    block_types=["conv", "conv", "transfrom", "transform"],
    strides=[2, 2, 2, 2],
    expansion=4,
    se_ratio=0.25,
    head_dimension=32,
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="gelu",
    drop_connect_rate=0,
    classifier_activation="softmax",
    drop_rate=0,
    pretrained=None,
    model_name="coatnet",
    kwargs=None,
):
    inputs = keras.layers.Input(input_shape)

    """ stage 0, Stem_stage """
    nn = conv2d_no_bias(inputs, stem_width, 3, strides=2, padding="same", name="stem_1_")
    nn = batchnorm_with_activation(nn, activation=activation, name="stem_1_")
    nn = conv2d_no_bias(nn, stem_width, 3, strides=1, padding="same", name="stem_2_")

    """ stage [1, 2, 3, 4] """
    total_blocks = sum(num_blocks)
    global_block_id = 0
    for stack_id, (num_block, out_channel, block_type) in enumerate(zip(num_blocks, out_channels, block_types)):
        is_conv_block = True if block_type[0].lower() == "c" else False
        stack_se_ratio = se_ratio[stack_id] if isinstance(se_ratio, (list, tuple)) else se_ratio
        stack_strides = strides[stack_id] if isinstance(strides, (list, tuple)) else strides
        for block_id in range(num_block):
            name = "stage_{}_block_{}_".format(stack_id + 1, block_id + 1)
            stride = stack_strides if block_id == 0 else 1
            conv_short_cut = True if block_id == 0 else False
            block_se_ratio = stack_se_ratio[block_id] if isinstance(stack_se_ratio, (list, tuple)) else stack_se_ratio
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            global_block_id += 1
            if is_conv_block:
                nn = res_MBConv(nn, out_channel, conv_short_cut, stride, expansion, block_se_ratio, block_drop_rate, activation=activation, name=name)
            else:
                nn = res_mhsa(nn, out_channel, conv_short_cut, stride, head_dimension, block_drop_rate, activation=activation, name=name)
                nn = res_ffn(nn, expansion=expansion, drop_rate=block_drop_rate, activation=activation, name=name + "ffn_")

    if num_classes > 0:
        nn = keras.layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        if drop_rate > 0:
            nn = keras.layers.Dropout(drop_rate)(nn)
        nn = keras.layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="predictions")(nn)

    model = keras.models.Model(inputs, nn, name=model_name)
    return model


def CoAtNetT(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", **kwargs):
    num_blocks = [2, 3, 5, 2]
    out_channels = [64, 128, 256, 512]
    stem_width = 64
    return CoAtNet(**locals(), model_name="coatnett", **kwargs)


def CoAtNet0(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", **kwargs):
    num_blocks = [2, 3, 5, 2]
    out_channels = [96, 192, 384, 768]
    stem_width = 64
    return CoAtNet(**locals(), model_name="coatnet0", **kwargs)


def CoAtNet1(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", **kwargs):
    num_blocks = [2, 6, 14, 2]
    out_channels = [96, 192, 384, 768]
    stem_width = 64
    return CoAtNet(**locals(), model_name="coatnet1", **kwargs)


def CoAtNet2(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", **kwargs):
    num_blocks = [2, 6, 14, 2]
    out_channels = [128, 256, 512, 1024]
    stem_width = 128
    return CoAtNet(**locals(), model_name="coatnet2", **kwargs)


def CoAtNet3(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", **kwargs):
    num_blocks = [2, 6, 14, 2]
    out_channels = [192, 384, 768, 1536]
    stem_width = 192
    return CoAtNet(**locals(), model_name="coatnet3", **kwargs)


def CoAtNet4(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", **kwargs):
    num_blocks = [2, 12, 28, 2]
    out_channels = [192, 384, 768, 1536]
    stem_width = 192
    return CoAtNet(**locals(), model_name="coatnet4", **kwargs)


def CoAtNet5(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", **kwargs):
    num_blocks = [2, 12, 28, 2]
    out_channels = [256, 512, 1280, 2048]
    stem_width = 192
    head_dimension = 64
    return CoAtNet(**locals(), model_name="coatnet5", **kwargs)


def CoAtNet6(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", **kwargs):
    num_blocks = [2, 4, 8, 42, 2]
    out_channels = [192, 384, 768, 1536, 2048]
    block_types = ["conv", "conv", "conv", "transfrom", "transform"]
    strides = [2, 2, 2, 1, 2]
    stem_width = 192
    head_dimension = 128
    return CoAtNet(**locals(), model_name="coatnet6", **kwargs)


def CoAtNet7(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", **kwargs):
    num_blocks = [2, 4, 8, 42, 2]
    out_channels = [256, 512, 1024, 2048, 3072]
    block_types = ["conv", "conv", "conv", "transfrom", "transform"]
    strides = [2, 2, 2, 1, 2]
    stem_width = 192
    head_dimension = 128
    return CoAtNet(**locals(), model_name="coatnet7", **kwargs)
