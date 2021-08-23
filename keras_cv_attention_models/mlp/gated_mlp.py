from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow as tf
import os

BATCH_NORM_EPSILON = 1e-5


def layer_norm(inputs, name=None):
    norm_axis = -1 if K.image_data_format() == "channels_last" else 1
    return keras.layers.LayerNormalization(axis=norm_axis, epsilon=BATCH_NORM_EPSILON, name=name)(inputs)


def res_gated_mlp_block(inputs, channels_mlp_dim, drop_rate=0, activation="gelu", name=None):
    nn = layer_norm(inputs, name=name + "pre_ln")
    nn = keras.layers.Dense(channels_mlp_dim, name=name + "pre_dense")(nn)
    nn = keras.layers.Activation(activation, name=name + "gelu")(nn)
    # Drop

    # SpatialGatingUnit
    uu, vv = tf.split(nn, 2, axis=-1)
    # print(f">>>> {uu.shape = }, {vv.shape = }")
    vv = layer_norm(vv, name=name + "vv_ln")
    vv = keras.layers.Permute((2, 1), name=name + "permute_1")(vv)
    ww_init = keras.initializers.truncated_normal(stddev=1e-6)
    vv = keras.layers.Dense(vv.shape[-1], kernel_initializer=ww_init, bias_initializer="ones", name=name + "vv_dense")(vv)
    vv = keras.layers.Permute((2, 1), name=name + "permute_2")(vv)
    # print(f">>>> {uu.shape = }, {vv.shape = }")
    gated_out = keras.layers.Multiply()([uu, vv])

    nn = keras.layers.Dense(inputs.shape[-1], name=name + "gated_dense")(gated_out)
    # Drop

    # Drop path
    if drop_rate > 0:
        nn = keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1), name=name + "drop")(nn)
    return keras.layers.Add(name=name + "out")([nn, inputs])


def GMLP(
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
    model_name="gmlp",
    kwargs=None,
):
    inputs = keras.Input(input_shape)
    nn = keras.layers.Conv2D(stem_width, kernel_size=patch_size, strides=patch_size, padding="valid", name="stem")(inputs)
    nn = keras.layers.Reshape([nn.shape[1] * nn.shape[2], stem_width])(nn)

    drop_connect_s, drop_connect_e = drop_connect_rate if isinstance(drop_connect_rate, (list, tuple)) else [drop_connect_rate, drop_connect_rate]
    for ii in range(num_blocks):
        name = "{}_{}_".format("gmlp", str(ii + 1))
        block_drop_rate = drop_connect_s + (drop_connect_e - drop_connect_s) * ii / num_blocks
        nn = res_gated_mlp_block(nn, channels_mlp_dim=channels_mlp_dim, drop_rate=block_drop_rate, activation=activation, name=name)
    nn = layer_norm(nn, name="pre_head_norm")

    if num_classes > 0:
        # nn = tf.reduce_mean(nn, axis=1)
        nn = keras.layers.GlobalAveragePooling1D()(nn)
        if dropout > 0 and dropout < 1:
            nn = keras.layers.Dropout(dropout)(nn)
        nn = keras.layers.Dense(num_classes, activation=classifier_activation, name="predictions")(nn)

    if sam_rho != 0:
        from keras_cv_attention_models.model_surgery import SAMModel

        model = SAMModel(inputs, nn, name=model_name)
    else:
        model = keras.Model(inputs, nn, name=model_name)
    reload_model_weights(model, input_shape, pretrained)
    return model


def reload_model_weights(model, input_shape=(224, 224, 3), pretrained="imagenet"):
    pretrained_dd = {
        "gmlp_s16": ["imagenet"],
    }
    if model.name not in pretrained_dd or pretrained not in pretrained_dd[model.name]:
        print(">>>> No pretraind available, model will be randomly initialized")
        return

    pre_url = "https://github.com/leondgarse/keras_cv_attention_models/releases/download/mlp/{}_{}.h5"
    url = pre_url.format(model.name, pretrained)
    file_name = os.path.basename(url)
    try:
        pretrained_model = keras.utils.get_file(file_name, url, cache_subdir="models")
    except:
        print("[Error] will not load weights, url not found or download failed:", url)
        return
    else:
        print(">>>> Load pretraind from:", pretrained_model)
        model.load_weights(pretrained_model, by_name=True, skip_mismatch=True)


BLOCK_CONFIGS = {
    "tiny16": {
        "num_blocks": 30,
        "patch_size": 16,
        "stem_width": 128,
        "channels_mlp_dim": 128 * 6,
    },
    "s16": {
        "num_blocks": 30,
        "patch_size": 16,
        "stem_width": 256,
        "channels_mlp_dim": 256 * 6,
    },
    "b16": {
        "num_blocks": 30,
        "patch_size": 16,
        "stem_width": 512,
        "channels_mlp_dim": 512 * 6,
    },
}


def GMLPTiny16(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return GMLP(**BLOCK_CONFIGS["tiny16"], **locals(), model_name="gmlp_tiny16", **kwargs)


def GMLPS16(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return GMLP(**BLOCK_CONFIGS["s16"], **locals(), model_name="gmlp_s16", **kwargs)


def GMLPB16(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return GMLP(**BLOCK_CONFIGS["b16"], **locals(), model_name="gmlp_b16", **kwargs)
