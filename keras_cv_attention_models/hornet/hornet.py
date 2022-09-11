import tensorflow as tf
from tensorflow import keras
from keras_cv_attention_models.attention_layers import (
    ChannelAffine,
    conv2d_no_bias,
    depthwise_conv2d_no_bias,
    drop_block,
    layer_norm,
    mlp_block,
    add_pre_post_process,
)
from keras_cv_attention_models.download_and_load import reload_model_weights

PRETRAINED_DICT = {
    "hornet_base": {"imagenet": {224: "b5e808fe5996a8cc980552994f5ca875"}},
    "hornet_base_gf": {"imagenet": {224: "c08c8dc1666c4afbc98e856fa7e53c99"}},
    "hornet_large": {"imagenet22k": {224: "27e7a8f18657c82cf9ad07dc21df9a35"}},
    "hornet_large_gf": {"imagenet22k": {224: "ffd3e68dae2a365d0ecb5df2f0ea881b"}},
    "hornet_large_gf": {"imagenet22k": {384: "1e3bd0f4b63f65ff7aa3397129ff7a23"}},
    "hornet_small": {"imagenet": {224: "4f29c19491c7e51faa4161fcf3888531"}},
    "hornet_small_gf": {"imagenet": {224: "96c7f2c640b27b967a962377e71217ec"}},
    "hornet_tiny": {"imagenet": {224: "b1de78ead2cbc7649c6fbf3452ac4401"}},
    "hornet_tiny_gf": {"imagenet": {224: "358ab295b616cb4b0910aef4afbbf1d9"}},
}


@tf.keras.utils.register_keras_serializable(package="kecam/hornet")
class ComplexDense(keras.layers.Layer):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def build(self, input_shape):
        _, input_height, input_width, channel = input_shape

        param_shape = (2, input_height, input_width, channel)  # 2 means `real, img` for converting to complex
        initializer = tf.initializers.RandomNormal(stddev=0.02)
        self.complex_weight = self.add_weight(name="complex_weight", shape=param_shape, initializer=initializer, trainable=True)
        self.input_height, self.input_width = input_height, input_width

    def call(self, inputs):
        complex_weight = tf.complex(self.complex_weight[0], self.complex_weight[1])
        complex_weight = tf.cast(complex_weight, inputs.dtype)
        return inputs * complex_weight

    def load_resized_pos_emb(self, source_layer, method="bilinear"):
        if isinstance(source_layer, dict):
            source_tt = source_layer["complex_weight:0"]  # weights
        else:
            source_tt = source_layer.complex_weight  # layer
        tt = tf.image.resize(source_tt, (self.input_height, self.input_width), method=method, antialias=True)
        self.complex_weight.assign(tt)


def global_local_filter(inputs, name=None):
    _, height, width, channel = inputs.shape
    nn = layer_norm(inputs, name=name and name + "pre_")
    dw, fft = tf.split(nn, 2, axis=-1)
    dw = depthwise_conv2d_no_bias(dw, 3, padding="SAME", use_bias=False, name=name)

    # fft = tf.py_function(lambda xx: np.fft.rfft2(xx, axes=(1, 2), norm='ortho'), [fft], Tout=tf.complex128)
    ortho_norm = float(tf.sqrt(float(height * width)))
    # np.fft.rfft2(aa, axes=[1, 2]) ==> tf.transpose(tf.signal.rfft2d(tf.transpose(aa, [0, 3, 1, 2])), [0, 2, 3, 1])
    fft = tf.transpose(fft, [0, 3, 1, 2])
    fft = keras.layers.Lambda(tf.signal.rfft2d)(fft)
    fft = tf.transpose(fft, [0, 2, 3, 1])
    # fft /= ortho_norm  # Means `norm='ortho'`, but will multiply back for `irfft2d`, not affecting results.
    # fft.set_shape([None, height, (width + 2) // 2, channel // 2])
    # print(f">>>> {inputs.shape = }, {fft.shape = }, {fft.dtype = }")
    fft = ComplexDense(name=name and name + "complex_dense")(fft)
    # fft = tf.py_function(lambda xx: np.fft.irfft2(xx, s=[height, width], axes=(1, 2), norm='ortho'), [fft], Tout=inputs.dtype)
    # np.fft.irfft2(bb, s=[13, 14], axes=(1, 2)) ==> tf.transpose(tf.signal.irfft2d(tf.transpose(bb, [0, 3, 1, 2]), fft_length=[13, 14]), [0, 2, 3, 1])
    fft = tf.transpose(fft, [0, 3, 1, 2])
    fft = keras.layers.Lambda(lambda xx: tf.signal.irfft2d(xx, fft_length=[height, width]))(fft)
    fft = tf.transpose(fft, [0, 2, 3, 1])
    # fft *= ortho_norm
    # fft = tf.cast(fft, inputs.dtype)
    # fft.set_shape([None, height, width, channel // 2])

    out = tf.concat([tf.expand_dims(dw, -1), tf.expand_dims(fft, -1)], axis=-1)
    out = tf.reshape(out, [-1, height, width, channel])
    out = layer_norm(out, name=name and name + "post_")
    return out


def gnconv(inputs, use_global_local_filter=False, gn_split=5, scale=1.0, name=None):
    input_channel = inputs.shape[-1]
    nn = conv2d_no_bias(inputs, input_channel * 2, kernel_size=1, use_bias=True, name=name and name + "pre_")
    split_dims = [input_channel // (2 ** ii) for ii in range(gn_split)][::-1]
    # print(f">>>> {nn.shape = }, {split_dims = }")
    pw_first, dw_list = tf.split(nn, [split_dims[0], sum(split_dims)], axis=-1)

    if use_global_local_filter:
        dw_list = global_local_filter(dw_list, name=name and name + "gf_")
    else:
        dw_list = depthwise_conv2d_no_bias(dw_list, kernel_size=7, padding="SAME", use_bias=True, name=name and name + "list_")
    dw_list *= scale

    dw_list = tf.split(dw_list, split_dims, axis=-1)
    nn = pw_first * dw_list[0]
    for id, dw in enumerate(dw_list[1:], start=1):
        pw = conv2d_no_bias(nn, dw.shape[-1], kernel_size=1, use_bias=True, name=name and name + "pw{}_".format(id))
        nn = pw * dw

    nn = conv2d_no_bias(nn, input_channel, kernel_size=1, use_bias=True, name=name and name + "output_")
    return nn


def block(inputs, mlp_ratio=4, use_global_local_filter=False, gn_split=5, scale=1.0, layer_scale=0, drop_rate=0, activation="gelu", name=""):
    # print(global_query)
    input_channel = inputs.shape[-1]
    attn = layer_norm(inputs, name=name + "attn_")
    attn = gnconv(attn, use_global_local_filter, gn_split=gn_split, scale=scale, name=name + "gnconv_")
    attn = ChannelAffine(use_bias=False, weight_init_value=layer_scale, name=name + "1_gamma")(attn) if layer_scale >= 0 else attn
    attn = drop_block(attn, drop_rate=drop_rate, name=name + "attn_")
    attn_out = keras.layers.Add(name=name + "attn_out")([inputs, attn])

    mlp = layer_norm(attn_out, name=name + "mlp_")
    mlp = mlp_block(mlp, int(input_channel * mlp_ratio), use_conv=False, activation=activation, name=name + "mlp_")
    mlp = ChannelAffine(use_bias=False, weight_init_value=layer_scale, name=name + "2_gamma")(mlp) if layer_scale >= 0 else mlp
    mlp = drop_block(mlp, drop_rate=drop_rate, name=name + "mlp_")
    return keras.layers.Add(name=name + "output")([attn_out, mlp])


def HorNet(
    num_blocks=[2, 3, 18, 2],
    embed_dim=64,
    mlp_ratio=4,
    gn_split=[2, 3, 4, 5],
    use_global_local_filter=False,
    scale=0.3333333,
    layer_scale=1e-6,
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="gelu",
    drop_connect_rate=0,
    classifier_activation="softmax",
    dropout=0,
    pretrained=None,
    model_name="hornet",
    kwargs=None,
):
    """Patch stem"""
    inputs = keras.layers.Input(input_shape)
    nn = conv2d_no_bias(inputs, embed_dim, kernel_size=4, strides=4, use_bias=True, name="stem_")
    nn = layer_norm(nn, name="stem_")

    """ stages """
    total_blocks = sum(num_blocks)
    global_block_id = 0
    for stack_id, num_block in enumerate(num_blocks):
        stack_name = "stack{}_".format(stack_id + 1)
        if stack_id > 0:
            nn = layer_norm(nn, name=stack_name)
            nn = conv2d_no_bias(nn, nn.shape[-1] * 2, kernel_size=2, strides=2, use_bias=True, name=stack_name)

        cur_use_global_local_filter = use_global_local_filter[stack_id] if isinstance(use_global_local_filter, (list, tuple)) else use_global_local_filter
        cur_gn_split = gn_split[stack_id] if isinstance(gn_split, (list, tuple)) else gn_split
        cur_scale = scale[stack_id] if isinstance(scale, (list, tuple)) else scale

        for block_id in range(num_block):
            block_name = stack_name + "block{}_".format(block_id + 1)
            block_drop_rate = drop_connect_rate * global_block_id / total_blocks
            nn = block(nn, mlp_ratio, cur_use_global_local_filter, cur_gn_split, cur_scale, layer_scale, block_drop_rate, activation, name=block_name)
            global_block_id += 1

    if num_classes > 0:
        nn = keras.layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        if dropout > 0:
            nn = keras.layers.Dropout(dropout, name="head_drop")(nn)
        nn = layer_norm(nn, name="pre_output_")
        nn = keras.layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="predictions")(nn)

    model = keras.models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    reload_model_weights(model, PRETRAINED_DICT, "hornet", pretrained, mismatch_class=ComplexDense)
    return model


def HorNetTiny(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained=None, **kwargs):
    return HorNet(**locals(), model_name="hornet_tiny", **kwargs)


def HorNetTinyGF(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained=None, **kwargs):
    use_global_local_filter = [False, False, True, True]
    return HorNet(**locals(), model_name="hornet_tiny_gf", **kwargs)


def HorNetSmall(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained=None, **kwargs):
    embed_dim = 96
    return HorNet(**locals(), model_name="hornet_small", **kwargs)

def HorNetSmallGF(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained=None, **kwargs):
    embed_dim = 96
    use_global_local_filter = [False, False, True, True]
    return HorNet(**locals(), model_name="hornet_small_gf", **kwargs)


def HorNetBase(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained=None, **kwargs):
    embed_dim = 128
    return HorNet(**locals(), model_name="hornet_base", **kwargs)


def HorNetBaseGF(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained=None, **kwargs):
    embed_dim = 128
    use_global_local_filter = [False, False, True, True]
    return HorNet(**locals(), model_name="hornet_base_gf", **kwargs)


def HorNetLarge(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained=None, **kwargs):
    embed_dim = 192
    return HorNet(**locals(), model_name="hornet_large", **kwargs)


def HorNetLargeGF(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained=None, **kwargs):
    embed_dim = 192
    use_global_local_filter = [False, False, True, True]
    return HorNet(**locals(), model_name="hornet_large_gf", **kwargs)
