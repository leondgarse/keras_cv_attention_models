from tensorflow import keras
from tensorflow.keras import backend as K
import os

BATCH_NORM_EPSILON = 1e-5


def layer_norm(inputs, name=None):
    norm_axis = -1 if K.image_data_format() == "channels_last" else 1
    return keras.layers.LayerNormalization(axis=norm_axis, epsilon=BATCH_NORM_EPSILON, name=name)(inputs)


def mlp_block(inputs, hidden_dim, activation="gelu", name=None):
    nn = keras.layers.Dense(hidden_dim, name=name + "Dense_0")(inputs)
    nn = keras.layers.Activation(activation, name=name + "gelu")(nn)
    nn = keras.layers.Dense(inputs.shape[-1], name=name + "Dense_1")(nn)
    return nn


def mixer_block(inputs, tokens_mlp_dim, channels_mlp_dim, drop_rate=0, activation="gelu", name=None):
    nn = layer_norm(inputs, name=name + "LayerNorm_0")
    nn = keras.layers.Permute((2, 1), name=name + "permute_0")(nn)
    nn = mlp_block(nn, tokens_mlp_dim, activation, name=name + "token_mixing/")
    nn = keras.layers.Permute((2, 1), name=name + "permute_1")(nn)
    if drop_rate > 0:
        nn = keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1), name=name + "token_drop")(nn)
    token_out = keras.layers.Add(name=name + "add_0")([nn, inputs])

    nn = layer_norm(token_out, name=name + "LayerNorm_1")
    channel_out = mlp_block(nn, channels_mlp_dim, activation, name=name + "channel_mixing/")
    if drop_rate > 0:
        channel_out = keras.layers.Dropout(drop_rate, noise_shape=(None, 1, 1), name=name + "channel_drop")(channel_out)
    return keras.layers.Add(name=name + "add_1")([channel_out, token_out])


def MLPMixer(
    num_blocks,
    patch_size,
    stem_width,
    tokens_mlp_dim,
    channels_mlp_dim,
    input_shape=(224, 224, 3),
    num_classes=0,
    activation="gelu",
    sam_rho=0,
    dropout=0,
    drop_connect_rate=0,
    classifier_activation="softmax",
    pretrained="imagenet",
    model_name="mlp_mixer",
    kwargs=None,
):
    inputs = keras.Input(input_shape)
    nn = keras.layers.Conv2D(stem_width, kernel_size=patch_size, strides=patch_size, padding="same", name="stem")(inputs)
    nn = keras.layers.Reshape([nn.shape[1] * nn.shape[2], stem_width])(nn)

    drop_connect_s, drop_connect_e = drop_connect_rate if isinstance(drop_connect_rate, (list, tuple)) else [drop_connect_rate, drop_connect_rate]
    for ii in range(num_blocks):
        name = "{}_{}/".format("MixerBlock", str(ii))
        block_drop_rate = drop_connect_s + (drop_connect_e - drop_connect_s) * ii / num_blocks
        nn = mixer_block(nn, tokens_mlp_dim, channels_mlp_dim, drop_rate=block_drop_rate, activation=activation, name=name)
    nn = layer_norm(nn, name="pre_head_layer_norm")

    if num_classes > 0:
        nn = keras.layers.GlobalAveragePooling1D()(nn)  # tf.reduce_mean(nn, axis=1)
        if dropout > 0 and dropout < 1:
            nn = keras.layers.Dropout(dropout)(nn)
        nn = keras.layers.Dense(num_classes, activation=classifier_activation, name="head")(nn)

    if sam_rho != 0:
        from keras_cv_attention_models.model_surgery import SAMModel

        model = SAMModel(inputs, nn, name=model_name)
    else:
        model = keras.Model(inputs, nn, name=model_name)
    reload_model_weights(model, input_shape, pretrained)
    return model


def reload_model_weights(model, input_shape=(224, 224, 3), pretrained="imagenet"):
    pretrained_dd = {
        "mlp_mixer_b16": ["imagenet", "imagenet_sam", "imagenet21k"],
        "mlp_mixer_l16": ["imagenet", "imagenet21k"],
        "mlp_mixer_b32": ["imagenet_sam"],
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
    "s32": {
        "num_blocks": 8,
        "patch_size": 32,
        "stem_width": 512,
        "tokens_mlp_dim": 256,
        "channels_mlp_dim": 2048,
    },
    "s16": {
        "num_blocks": 8,
        "patch_size": 16,
        "stem_width": 512,
        "tokens_mlp_dim": 256,
        "channels_mlp_dim": 2048,
    },
    "b32": {
        "num_blocks": 12,
        "patch_size": 32,
        "stem_width": 768,
        "tokens_mlp_dim": 384,
        "channels_mlp_dim": 3072,
    },
    "b16": {
        "num_blocks": 12,
        "patch_size": 16,
        "stem_width": 768,
        "tokens_mlp_dim": 384,
        "channels_mlp_dim": 3072,
    },
    "l32": {
        "num_blocks": 24,
        "patch_size": 32,
        "stem_width": 1024,
        "tokens_mlp_dim": 512,
        "channels_mlp_dim": 4096,
    },
    "l16": {
        "num_blocks": 24,
        "patch_size": 16,
        "stem_width": 1024,
        "tokens_mlp_dim": 512,
        "channels_mlp_dim": 4096,
    },
    "h14": {
        "num_blocks": 32,
        "patch_size": 14,
        "stem_width": 1280,
        "tokens_mlp_dim": 640,
        "channels_mlp_dim": 5120,
    },
}


def MLPMixerS32(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return MLPMixer(**BLOCK_CONFIGS["s32"], **locals(), model_name="mlp_mixer_s32", **kwargs)


def MLPMixerS16(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return MLPMixer(**BLOCK_CONFIGS["s16"], **locals(), model_name="mlp_mixer_s16", **kwargs)


def MLPMixerB32(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return MLPMixer(**BLOCK_CONFIGS["b32"], **locals(), model_name="mlp_mixer_b32", **kwargs)


def MLPMixerB16(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return MLPMixer(**BLOCK_CONFIGS["b16"], **locals(), model_name="mlp_mixer_b16", **kwargs)


def MLPMixerL32(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return MLPMixer(**BLOCK_CONFIGS["l32"], **locals(), model_name="mlp_mixer_l32", **kwargs)


def MLPMixerL16(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return MLPMixer(**BLOCK_CONFIGS["l16"], **locals(), model_name="mlp_mixer_l16", **kwargs)


def MLPMixerH14(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return MLPMixer(**BLOCK_CONFIGS["h14"], **locals(), model_name="mlp_mixer_h14", **kwargs)


if __name__ == "__convert__":
    aa = np.load("../models/imagenet1k_Mixer-B_16.npz")
    bb = {kk: vv for kk, vv in aa.items()}
    # cc = {kk: vv.shape for kk, vv in bb.items()}

    import mlp_mixer

    mm = mlp_mixer.MLPMixerB16(num_classes=1000, pretrained=None)
    # dd = {ii.name: ii.shape for ii in mm.weights}

    target_weights_dict = {"kernel": 0, "bias": 1, "scale": 0, "running_var": 3}
    for kk, vv in bb.items():
        split_name = kk.split("/")
        source_name = "/".join(split_name[:-1])
        source_weight_type = split_name[-1]
        target_layer = mm.get_layer(source_name)

        target_weights = target_layer.get_weights()
        target_weight_pos = target_weights_dict[source_weight_type]
        print("[{}] source: {}, target: {}".format(kk, vv.shape, target_weights[target_weight_pos].shape))

        target_weights[target_weight_pos] = vv
        target_layer.set_weights(target_weights)
