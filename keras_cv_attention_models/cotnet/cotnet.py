import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
from keras_cv_attention_models.download_and_load import reload_model_weights

BATCH_NORM_DECAY = 0.9
BATCH_NORM_EPSILON = 1e-5
CONV_KERNEL_INITIALIZER = tf.keras.initializers.VarianceScaling(scale=2.0, mode="fan_out", distribution="truncated_normal")
# CONV_KERNEL_INITIALIZER = 'glorot_uniform'


PRETRAINED_DICT = {
    "cotnet101": {"224": "6c65fceeae826a0659bf43f62b312441"},
    "cotnet50": {"224": "6a087a93b1669b7e4e2e8875f1f81b17"},
    "se_cotnetd101": {"224": "0b2f4b26c99d58de9043eb27cb1ad33d"},
    "se_cotnetd152": {"224": "9211c7166fe3d116fe4492b9a3069a21", "320": "a26c234902f64f24dcd716b6ad0da01d"},
    "se_cotnetd50": {"224": "cab719c9e54e4967f5a5aabe47863eaa"},
}


def batchnorm_with_activation(inputs, activation="relu", zero_gamma=False, name=""):
    """Performs a batch normalization followed by an activation. """
    bn_axis = 3 if K.image_data_format() == "channels_last" else 1
    gamma_initializer = tf.zeros_initializer() if zero_gamma else tf.ones_initializer()
    nn = keras.layers.BatchNormalization(
        axis=bn_axis,
        momentum=BATCH_NORM_DECAY,
        epsilon=BATCH_NORM_EPSILON,
        gamma_initializer=gamma_initializer,
        name=name + "bn",
    )(inputs)
    if activation:
        nn = keras.layers.Activation(activation=activation, name=name + activation)(nn)
    return nn


def conv2d_no_bias(inputs, filters, kernel_size, strides=1, padding="VALID", use_bias=False, name="", **kwargs):
    if padding.upper() == "SAME":
        inputs = keras.layers.ZeroPadding2D(padding=kernel_size // 2, name=name + "pad")(inputs)
    return keras.layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="VALID",
        use_bias=use_bias,
        kernel_initializer=CONV_KERNEL_INITIALIZER,
        name=name + "conv",
        **kwargs,
    )(inputs)


def group_conv(inputs, filters, kernel_size, groups=4, name="", **kwargs):
    # Using groups=num in `Conv2D` is slow with `mixed_float16` policy
    return conv2d_no_bias(inputs, filters, kernel_size, groups=groups, name=name)
    # splitted_inputs = tf.split(inputs, groups, axis=-1)
    # return tf.concat([conv2d_no_bias(splitted_inputs[ii], filters // groups, kernel_size, name=name + "g{}_".format(ii + 1), **kwargs) for ii in range(groups)], axis=-1)


def cot_attention(inputs, kernel_size=3, activation="relu", name=""):
    from tensorflow_addons.layers import GroupNormalization

    # inputs, kernel_size, strides, activation, name = tf.ones([1, 7, 7, 512]), 3, 1, "relu", ""
    filters = inputs.shape[-1]
    randix = 2

    # key_embed
    if kernel_size // 2 != 0:
        key_input = keras.layers.ZeroPadding2D(padding=kernel_size // 2, name=name + "conv_pad")(inputs)
    else:
        key_input = inputs
    key = group_conv(key_input, filters, kernel_size, groups=4, name=name + "key_")
    key = batchnorm_with_activation(key, activation=activation, zero_gamma=False, name=name + "key_")

    # query key
    qk = keras.layers.Concatenate(axis=-1)([inputs, key])
    _, height, width, _ = qk.shape

    # embed weights from query and key, ignore `num_heads`, as it's set as `1`
    reduction = 8
    embed_ww = conv2d_no_bias(qk, filters // randix, 1, name=name + "embed_ww_1_")
    embed_ww = batchnorm_with_activation(embed_ww, activation=activation, zero_gamma=False, name=name + "embed_ww_1_")
    embed_ww = conv2d_no_bias(embed_ww, kernel_size * kernel_size * filters // reduction, 1, use_bias=True, name=name + "embed_ww_2_")
    embed_ww = GroupNormalization(groups=filters // reduction, epsilon=BATCH_NORM_EPSILON, name=name + "embed_ww_group_norm")(embed_ww)
    embed_ww = tf.reshape(embed_ww, (-1, height, width, filters // reduction, kernel_size * kernel_size))
    embed_ww = tf.expand_dims(tf.transpose(embed_ww, [0, 1, 2, 4, 3]), axis=-2)  # expand dim on `reduction` axis

    # matmul, local_conv
    embed = conv2d_no_bias(inputs, filters, 1, name=name + "embed_1_")
    embed = batchnorm_with_activation(embed, activation=None, zero_gamma=False, name=name + "embed_1_")

    # unfold_j = torch.nn.Unfold(kernel_size=kernel_size, dilation=1, padding=1, stride=1)
    # x2 = unfold_j(bb).view(-1, reduction, filters // reduction, kernel_size * kernel_size, height, width)
    # y2 = (ww.unsqueeze(2) * x2.unsqueeze(1)).sum(-3).view(-1, filters, height, width)
    sizes, patch_strides = [1, kernel_size, kernel_size, 1], [1, 1, 1, 1]
    embed = keras.layers.ZeroPadding2D(padding=kernel_size // 2, name=name + "embed_pad")(embed)
    embed = tf.image.extract_patches(embed, sizes=sizes, strides=patch_strides, rates=(1, 1, 1, 1), padding="VALID")
    embed = tf.reshape(embed, [-1, height, width, kernel_size * kernel_size, reduction, filters // reduction])

    embed_out = keras.layers.Multiply(name=name + "local_conv_mul")([embed, embed_ww])
    embed_out = tf.reduce_sum(embed_out, axis=-3)  # reduce on `kernel_size * kernel_size` axis
    embed_out = tf.reshape(embed_out, [-1, height, width, filters])
    embed_out = batchnorm_with_activation(embed_out, activation="swish", zero_gamma=False, name=name + "embed_2_")

    # attention
    attn = keras.layers.Add()([embed_out, key])
    attn = tf.reduce_mean(attn, axis=[1, 2], keepdims=True)
    # attn se module
    attn_se_filters = max(filters * randix // 4, 32)
    # attn = keras.layers.Dense(attn_se_filters, use_bias=True, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + "attn_se_dense_1")(attn)
    attn = conv2d_no_bias(attn, attn_se_filters, 1, use_bias=True, name=name + "attn_se_1_")
    attn = batchnorm_with_activation(attn, activation=activation, zero_gamma=False, name=name + "attn_se_")
    # attn = keras.layers.Dense(filters * randix, use_bias=True, kernel_initializer=CONV_KERNEL_INITIALIZER, name=name + "attn_se_dense_2")(attn)
    attn = conv2d_no_bias(attn, filters * randix, 1, use_bias=True, name=name + "attn_se_2_")
    attn = tf.reshape(attn, [-1, 1, 1, filters, randix])
    attn = tf.nn.softmax(attn, axis=-1)

    # value and output
    value = keras.layers.Concatenate(axis=-1)([tf.expand_dims(embed_out, -1), tf.expand_dims(key, -1)])
    output = keras.layers.Multiply()([value, attn])
    output = tf.reduce_sum(output, axis=-1, name=name + "out")
    return output


def cot_block(
    inputs, filters, strides=1, shortcut=False, expansion=4, cardinality=1, attn_type="cot", avd_first=True, drop_rate=0, activation="relu", name=""
):
    # target_dimension = round(planes * block.expansion * self.rb)
    expanded_filter = round(filters * expansion)
    if shortcut:
        # print(">>>> Downsample")
        # shortcut = conv2d_no_bias(inputs, expanded_filter, 1, strides=strides, name=name + "shorcut_")
        short_cut = keras.layers.AveragePooling2D(pool_size=strides, strides=strides, padding="SAME", name=name + "shorcut_pool")(inputs)
        shortcut = conv2d_no_bias(short_cut, expanded_filter, 1, strides=1, name=name + "shorcut_")
        shortcut = batchnorm_with_activation(shortcut, activation=None, zero_gamma=False, name=name + "shorcut_")
    else:
        shortcut = inputs

    # width = planes
    nn = conv2d_no_bias(inputs, filters, 1, name=name + "1_")
    nn = batchnorm_with_activation(nn, activation=activation, zero_gamma=False, name=name + "1_")

    if avd_first and strides > 1 and attn_type == "cot":
        nn = keras.layers.ZeroPadding2D(padding=1, name=name + "pool_pad")(nn)
        nn = keras.layers.AveragePooling2D(3, strides=2, name=name + "pool")(nn)

    if attn_type == "cot":
        nn = cot_attention(nn, 3, activation=activation, name=name + "cot_")
    elif attn_type == "cotx":
        nn = coxt_attention(nn, 3)  # Not implemented
    elif attn_type == "sa":
        from keras_cv_attention_models.attention_layers import split_attention_conv2d

        nn = split_attention_conv2d(nn, filters, kernel_size=3, groups=1, strides=strides, activation="swish", name=name + "sa_")
    else:
        nn = conv2d_no_bias(nn, filters, 3, strides=strides, padding="SAME", name=name + "conv_")

    if not avd_first and strides > 1 and attn_type == "cot":
        nn = keras.layers.ZeroPadding2D(padding=1, name=name + "pool_pad")(nn)
        nn = keras.layers.AveragePooling2D(3, strides=2, name=name + "pool")(nn)

    nn = conv2d_no_bias(nn, expanded_filter, 1, name=name + "2_")
    nn = batchnorm_with_activation(nn, activation=None, zero_gamma=True, name=name + "2_")

    # print(">>>>", nn.shape, shortcut.shape)
    if drop_rate > 0:
        nn = keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1, 1), name=name + "drop")(nn)
    nn = keras.layers.Add()([shortcut, nn])
    return keras.layers.Activation(activation, name=name + "out")(nn)


def cot_stack(inputs, blocks, filter, strides=2, expansion=4, cardinality=1, attn_types="cot", avd_first=True, stack_drop=0, activation="relu", name=""):
    shortcut = True if strides != 1 or inputs.shape[-1] != filter * expansion else False
    attn_type = attn_types[0] if isinstance(attn_types, (list, tuple)) else attn_types
    stack_drop_s, stack_drop_e = stack_drop if isinstance(stack_drop, (list, tuple)) else [stack_drop, stack_drop]
    nn = cot_block(inputs, filter, strides, shortcut, expansion, cardinality, attn_type, avd_first, stack_drop_s, activation, name=name + "block1_")
    shortcut = False
    # print(">>>> attn_types:", attn_types)
    for ii in range(1, blocks):
        block_name = name + "block{}_".format(ii + 1)
        block_drop_rate = stack_drop_s + (stack_drop_e - stack_drop_s) * ii / blocks
        attn_type = attn_types[ii] if isinstance(attn_types, (list, tuple)) else attn_types
        nn = cot_block(nn, filter, 1, shortcut, expansion, cardinality, attn_type, avd_first, block_drop_rate, activation, name=block_name)
    return nn


def stem(inputs, stem_width, activation="relu", deep_stem=False, name=""):
    if deep_stem:
        nn = conv2d_no_bias(inputs, stem_width, 3, strides=2, padding="same", name=name + "1_")
        nn = batchnorm_with_activation(nn, activation=activation, name=name + "1_")
        nn = conv2d_no_bias(nn, stem_width, 3, strides=1, padding="same", name=name + "2_")
        nn = batchnorm_with_activation(nn, activation=activation, name=name + "2_")
        nn = conv2d_no_bias(nn, stem_width * 2, 3, strides=1, padding="same", name=name + "3_")

        nn = batchnorm_with_activation(nn, activation=activation, name=name)
    else:
        nn = conv2d_no_bias(inputs, stem_width, 7, strides=2, padding="same", name=name)

        nn = batchnorm_with_activation(nn, activation=activation, name=name)
        nn = keras.layers.ZeroPadding2D(padding=1, name=name + "pool_pad")(nn)
        nn = keras.layers.MaxPool2D(pool_size=3, strides=2, name=name + "pool")(nn)
    return nn


def CotNet(
    num_blocks,
    stem_width=64,
    deep_stem=False,
    attn_types="cot",
    avd_first=True,
    strides=[1, 2, 2, 2],
    expansion=4,
    cardinality=1,
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="relu",
    drop_connect_rate=0,
    classifier_activation="softmax",
    pretrained="imagenet",
    model_name="cotnet",
    **kwargs
):
    inputs = keras.layers.Input(input_shape)
    nn = stem(inputs, stem_width, activation=activation, deep_stem=deep_stem, name="stem_")

    out_channels = [64, 128, 256, 512]
    total_blocks = sum(num_blocks)
    global_block_id = 0
    drop_connect_s, drop_connect_e = 0, drop_connect_rate
    for id, (num_block, out_channel, stride) in enumerate(zip(num_blocks, out_channels, strides)):
        name = "stack{}_".format(id + 1)
        stack_drop_s = drop_connect_rate * global_block_id / total_blocks
        stack_drop_e = drop_connect_rate * (global_block_id + num_block) / total_blocks
        stack_drop = (stack_drop_s, stack_drop_e)
        attn_type = attn_types[id] if isinstance(attn_types, (list, tuple)) else attn_types
        nn = cot_stack(nn, num_block, out_channel, stride, expansion, cardinality, attn_type, avd_first, stack_drop, activation=activation, name=name)
        global_block_id += num_block

    if num_classes > 0:
        nn = keras.layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        nn = keras.layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="predictions")(nn)

    model = keras.models.Model(inputs, nn, name=model_name)
    request_resolution = "320" if input_shape[0] == 320 and model.name == "se_cotnetd152" else "224"
    pretrained = request_resolution if pretrained is not None else None
    reload_model_weights(model, pretrained_dict=PRETRAINED_DICT, sub_release="cotnet", input_shape=input_shape, pretrained=pretrained)
    return model


def CotNet50(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 6, 3]
    strides = [1, 2, 2, 2]
    avd_first = True
    return CotNet(**locals(), **kwargs, model_name="cotnet50")


def CotNet101(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 23, 3]
    strides = [1, 2, 2, 2]
    avd_first = True
    return CotNet(**locals(), **kwargs, model_name="cotnet101")


def SECotNetD50(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 6, 3]
    strides = [2, 2, 2, 2]
    attn_types = [
        "sa",  # stack 1
        "sa",  # stack 2
        ["cot", "sa"] * (num_blocks[2] // 2 + 1),  # stack 3
        "cot",  # stack 4
    ]
    avd_first = True
    return CotNet(deep_stem=True, stem_width=32, **locals(), **kwargs, model_name="se_cotnetd50")


def SECotNetD101(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 4, 23, 3]
    strides = [2, 2, 2, 2]
    # [stack 1, stack 2, stack 3, stack 4], 50 means just as long as larger than num_blocks[2]
    attn_types = ["sa", "sa", ["cot", "sa"] * 50, "cot"]
    avd_first = True
    return CotNet(deep_stem=True, stem_width=64, **locals(), **kwargs, model_name="se_cotnetd101")


def SECotNetD152(input_shape=(224, 224, 3), num_classes=1000, activation="relu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    num_blocks = [3, 8, 36, 3]
    strides = [2, 2, 2, 2]
    # [stack 1, stack 2, stack 3, stack 4], 50 means just as long as larger than num_blocks[2]
    attn_types = ["sa", "sa", ["cot", "sa"] * 50, "cot"]
    avd_first = False
    return CotNet(deep_stem=True, stem_width=64, **locals(), **kwargs, model_name="se_cotnetd152")
