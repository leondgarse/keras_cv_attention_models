import numpy as np
from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models, functional, initializers
from keras_cv_attention_models.attention_layers import activation_by_name, add_pre_post_process
from keras_cv_attention_models.download_and_load import reload_model_weights

PRETRAINED_DICT = {
    "resmlp12": {"imagenet": "de6531fb461bcf52c25d3c36aa515583"},
    "resmlp24": {"imagenet": "f8127be7f8ba564fc59552c0cf6f3401"},
    "resmlp36": {"imagenet": "d0d3e6b09d7e975aaf46ff777c1fd73e"},
    "resmlp_b24": {"imagenet": "d7808ef59c06d2f1975ffddd28be82de", "imagenet22k": "8d3ae1abdac60b21ed1f2840b656b6bf"},
}


@backend.register_keras_serializable(package="resmlp")
class ChannelAffine(layers.Layer):
    def __init__(self, use_bias=True, weight_init_value=1, axis=-1, **kwargs):
        super().__init__(**kwargs)
        self.use_bias, self.weight_init_value, self.axis = use_bias, weight_init_value, axis
        self.ww_init = initializers.Constant(weight_init_value) if weight_init_value != 1 else "ones"
        self.bb_init = "zeros"
        self.supports_masking = False

    def build(self, input_shape):
        if self.axis == -1 or self.axis == len(input_shape) - 1:
            ww_shape = (input_shape[-1],)
        else:
            ww_shape = [1] * len(input_shape)
            axis = self.axis if isinstance(self.axis, (list, tuple)) else [self.axis]
            for ii in axis:
                ww_shape[ii] = input_shape[ii]
            ww_shape = ww_shape[1:]  # Exclude batch dimension

        self.ww = self.add_weight(name="weight", shape=ww_shape, initializer=self.ww_init, trainable=True)
        if self.use_bias:
            self.bb = self.add_weight(name="bias", shape=ww_shape, initializer=self.bb_init, trainable=True)
        super().build(input_shape)

    def call(self, inputs, **kwargs):
        return inputs * self.ww + self.bb if self.use_bias else inputs * self.ww

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_weights_channels_last(self):
        # channel_first -> channel_last
        weights = self.get_weights()
        if backend.image_data_format() != "channels_last" and self.axis == 1:
            weights = [np.squeeze(ii) for ii in weights]
        return weights

    def set_weights_channels_last(self, weights):
        # channel_last -> channel_first
        if backend.image_data_format() != "channels_last" and self.axis == 1:
            weights = [np.reshape(ii, self.ww.shape) for ii in weights]
        return self.set_weights(weights)

    def get_config(self):
        config = super().get_config()
        config.update({"use_bias": self.use_bias, "weight_init_value": self.weight_init_value, "axis": self.axis})
        return config


# NOT using
def channel_affine(inputs, use_bias=True, weight_init_value=1, name=""):
    ww_init = initializers.Constant(weight_init_value) if weight_init_value != 1 else "ones"
    nn = functional.expand_dims(inputs, 1)
    nn = layers.DepthwiseConv2D(1, depthwise_initializer=ww_init, use_bias=use_bias, name=name)(nn)
    return functional.squeeze(nn, 1)


def res_mlp_block(inputs, channels_mlp_dim, drop_rate=0, activation="gelu", name=None):
    channel_axis = -1 if backend.image_data_format() == "channels_last" else 1
    input_channel = inputs.shape[-1] if backend.image_data_format() == "channels_last" else inputs.shape[1]

    nn = ChannelAffine(use_bias=True, axis=channel_axis, name=name + "norm_1")(inputs)
    nn = layers.Permute((2, 1), name=name + "permute_1")(nn) if backend.image_data_format() == "channels_last" else nn
    nn = layers.Dense(nn.shape[-1], name=name + "token_mixing")(nn)
    nn = layers.Permute((2, 1), name=name + "permute_2")(nn) if backend.image_data_format() == "channels_last" else nn
    nn = ChannelAffine(use_bias=False, axis=channel_axis, name=name + "gamma_1")(nn)
    if drop_rate > 0:
        nn = layers.Dropout(drop_rate, noise_shape=(None, 1, 1), name=name + "token_drop")(nn)
    token_out = layers.Add(name=name + "add_1")([inputs, nn])

    nn = ChannelAffine(use_bias=True, axis=channel_axis, name=name + "norm_2")(token_out)
    nn = nn if backend.image_data_format() == "channels_last" else layers.Permute((2, 1), name=name and name + "permute_3")(nn)
    nn = layers.Dense(channels_mlp_dim, name=name + "channel_mixing_1")(nn)
    nn = activation_by_name(nn, activation, name=name + activation)
    nn = layers.Dense(input_channel, name=name + "channel_mixing_2")(nn)
    nn = nn if backend.image_data_format() == "channels_last" else layers.Permute((2, 1), name=name and name + "permute_4")(nn)
    channel_out = ChannelAffine(use_bias=False, axis=channel_axis, name=name + "gamma_2")(nn)
    if drop_rate > 0:
        channel_out = layers.Dropout(drop_rate, noise_shape=(None, 1, 1), name=name + "channel_drop")(channel_out)
    # print(f">>>> {inputs.shape = }, {channel_out.shape = }, {token_out.shape = }")
    nn = layers.Add(name=name + "output")([channel_out, token_out])
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
    # Regard input_shape as force using original shape if len(input_shape) == 4,
    # else assume channel dimention is the one with min value in input_shape, and put it first or last regarding image_data_format
    input_shape = backend.align_input_shape_by_image_data_format(input_shape)
    inputs = layers.Input(input_shape)
    nn = layers.Conv2D(stem_width, kernel_size=patch_size, strides=patch_size, padding="valid", name="stem")(inputs)
    new_shape = [nn.shape[1] * nn.shape[2], stem_width] if backend.image_data_format() == "channels_last" else [stem_width, nn.shape[2] * nn.shape[3]]
    nn = layers.Reshape(new_shape)(nn)

    drop_connect_s, drop_connect_e = drop_connect_rate if isinstance(drop_connect_rate, (list, tuple)) else [drop_connect_rate, drop_connect_rate]
    for ii in range(num_blocks):
        name = "{}_{}_".format("ResMlpBlock", str(ii + 1))
        block_drop_rate = drop_connect_s + (drop_connect_e - drop_connect_s) * ii / num_blocks
        nn = res_mlp_block(nn, channels_mlp_dim=channels_mlp_dim, drop_rate=block_drop_rate, activation=activation, name=name)
    nn = ChannelAffine(axis=-1 if backend.image_data_format() == "channels_last" else 1, name="pre_head_norm")(nn)

    if num_classes > 0:
        # nn = tf.reduce_mean(nn, axis=1)
        nn = layers.GlobalAveragePooling1D()(nn)
        if dropout > 0 and dropout < 1:
            nn = layers.Dropout(dropout)(nn)
        nn = layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="predictions")(nn)

    if sam_rho != 0:
        from keras_cv_attention_models.model_surgery import SAMModel

        model = SAMModel(inputs, nn, name=model_name)
    else:
        model = models.Model(inputs, nn, name=model_name)
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
