from tensorflow import keras
from keras_cv_attention_models.attention_layers import activation_by_name, add_pre_post_process
from keras_cv_attention_models.download_and_load import reload_model_weights

PRETRAINED_DICT = {
    "resmlp12": {"imagenet": "de6531fb461bcf52c25d3c36aa515583"},
    "resmlp24": {"imagenet": "f8127be7f8ba564fc59552c0cf6f3401"},
    "resmlp36": {"imagenet": "d0d3e6b09d7e975aaf46ff777c1fd73e"},
    "resmlp_b24": {"imagenet": "d7808ef59c06d2f1975ffddd28be82de", "imagenet22k": "8d3ae1abdac60b21ed1f2840b656b6bf"},
}


@keras.utils.register_keras_serializable(package="resmlp")
class ChannelAffine(keras.layers.Layer):
    def __init__(self, use_bias=True, weight_init_value=1, **kwargs):
        super(ChannelAffine, self).__init__(**kwargs)
        self.use_bias, self.weight_init_value = use_bias, weight_init_value
        self.ww_init = keras.initializers.Constant(weight_init_value) if weight_init_value != 1 else "ones"
        self.bb_init = "zeros"
        self.supports_masking = False

    def build(self, input_shape):
        self.ww = self.add_weight(name="weight", shape=(input_shape[-1],), initializer=self.ww_init, trainable=True)
        if self.use_bias:
            self.bb = self.add_weight(name="bias", shape=(input_shape[-1],), initializer=self.bb_init, trainable=True)
        super(ChannelAffine, self).build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs * self.ww + self.bb if self.use_bias else inputs * self.ww

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        config = super(ChannelAffine, self).get_config()
        config.update({"use_bias": self.use_bias, "weight_init_value": self.weight_init_value})
        return config


# NOT using
def channel_affine(inputs, use_bias=True, weight_init_value=1, name=""):
    ww_init = keras.initializers.Constant(weight_init_value) if weight_init_value != 1 else "ones"
    nn = keras.backend.expand_dims(inputs, 1)
    nn = keras.layers.DepthwiseConv2D(1, depthwise_initializer=ww_init, use_bias=use_bias, name=name)(nn)
    return keras.backend.squeeze(nn, 1)


def res_mlp_block(inputs, channels_mlp_dim, drop_rate=0, activation="gelu", name=None):
    nn = ChannelAffine(use_bias=True, name=name + "norm_1")(inputs)
    nn = keras.layers.Permute((2, 1), name=name + "permute_1")(nn)
    nn = keras.layers.Dense(nn.shape[-1], name=name + "token_mixing")(nn)
    nn = keras.layers.Permute((2, 1), name=name + "permute_2")(nn)
    nn = ChannelAffine(use_bias=False, name=name + "gamma_1")(nn)
    if drop_rate > 0:
        nn = keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1), name=name + "token_drop")(nn)
    token_out = keras.layers.Add(name=name + "add_1")([inputs, nn])

    nn = ChannelAffine(use_bias=True, name=name + "norm_2")(token_out)
    nn = keras.layers.Dense(channels_mlp_dim, name=name + "channel_mixing_1")(nn)
    nn = activation_by_name(nn, activation, name=name + activation)
    nn = keras.layers.Dense(inputs.shape[-1], name=name + "channel_mixing_2")(nn)
    channel_out = ChannelAffine(use_bias=False, name=name + "gamma_2")(nn)
    if drop_rate > 0:
        channel_out = keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1), name=name + "channel_drop")(channel_out)
    nn = keras.layers.Add(name=name + "output")([channel_out, token_out])
    return nn


def ResMLP(
    num_blocks,
    patch_size,
    stem_width,
    channels_mlp_dim,
    input_shape=(224, 224, 3),
    num_classes=0,
    activation="gelu",
    sam_rho=0,
    dropout=0,
    drop_connect_rate=0,
    classifier_activation="softmax",
    pretrained="imagenet",
    model_name="resmlp",
    kwargs=None,
):
    inputs = keras.Input(input_shape)
    nn = keras.layers.Conv2D(stem_width, kernel_size=patch_size, strides=patch_size, padding="valid", name="stem")(inputs)
    nn = keras.layers.Reshape([nn.shape[1] * nn.shape[2], stem_width])(nn)

    drop_connect_s, drop_connect_e = drop_connect_rate if isinstance(drop_connect_rate, (list, tuple)) else [drop_connect_rate, drop_connect_rate]
    for ii in range(num_blocks):
        name = "{}_{}_".format("ResMlpBlock", str(ii + 1))
        block_drop_rate = drop_connect_s + (drop_connect_e - drop_connect_s) * ii / num_blocks
        nn = res_mlp_block(nn, channels_mlp_dim=channels_mlp_dim, drop_rate=block_drop_rate, activation=activation, name=name)
    nn = ChannelAffine(name="pre_head_norm")(nn)

    if num_classes > 0:
        # nn = tf.reduce_mean(nn, axis=1)
        nn = keras.layers.GlobalAveragePooling1D()(nn)
        if dropout > 0 and dropout < 1:
            nn = keras.layers.Dropout(dropout)(nn)
        nn = keras.layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="predictions")(nn)

    if sam_rho != 0:
        from keras_cv_attention_models.model_surgery import SAMModel

        model = SAMModel(inputs, nn, name=model_name)
    else:
        model = keras.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    reload_model_weights(model, pretrained_dict=PRETRAINED_DICT, sub_release="mlp_family", pretrained=pretrained)
    return model


BLOCK_CONFIGS = {
    "12": {
        "num_blocks": 12,
        "patch_size": 16,
        "stem_width": 384,
        "channels_mlp_dim": 384 * 4,
    },
    "24": {
        "num_blocks": 24,
        "patch_size": 16,
        "stem_width": 384,
        "channels_mlp_dim": 384 * 4,
    },
    "36": {
        "num_blocks": 36,
        "patch_size": 16,
        "stem_width": 384,
        "channels_mlp_dim": 384 * 4,
    },
    "b24": {
        "num_blocks": 24,
        "patch_size": 8,
        "stem_width": 768,
        "channels_mlp_dim": 768 * 4,
    },
}


def ResMLP12(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return ResMLP(**BLOCK_CONFIGS["12"], **locals(), model_name="resmlp12", **kwargs)


def ResMLP24(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return ResMLP(**BLOCK_CONFIGS["24"], **locals(), model_name="resmlp24", **kwargs)


def ResMLP36(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return ResMLP(**BLOCK_CONFIGS["36"], **locals(), model_name="resmlp36", **kwargs)


def ResMLP_B24(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return ResMLP(**BLOCK_CONFIGS["b24"], **locals(), model_name="resmlp_b24", **kwargs)
