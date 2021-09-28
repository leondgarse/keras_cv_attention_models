import os
from tensorflow import keras


def reload_model_weights(model, pretrained_dict, sub_release, input_shape=(224, 224, 3), pretrained="imagenet"):
    if model.name not in pretrained_dict or pretrained not in pretrained_dict[model.name]:
        print(">>>> No pretrained available, model will be randomly initialized")
        return None

    pre_url = "https://github.com/leondgarse/keras_cv_attention_models/releases/download/{}/{}_{}.h5"
    url = pre_url.format(sub_release, model.name, pretrained)
    file_name = os.path.basename(url)
    file_hash = pretrained_dict[model.name][pretrained]
    try:
        pretrained_model = keras.utils.get_file(file_name, url, cache_subdir="models", file_hash=file_hash)
    except:
        print("[Error] will not load weights, url not found or download failed:", url)
        return None
    else:
        print(">>>> Load pretrained from:", pretrained_model)
        model.load_weights(pretrained_model, by_name=True, skip_mismatch=True)
        return pretrained_model


def reload_model_weights_with_mismatch(
    model, pretrained_dict, sub_release, mismatch_class, request_resolution=224, input_shape=(224, 224, 3), pretrained="imagenet"
):
    pretrained_model = reload_model_weights(model, pretrained_dict, sub_release, input_shape=input_shape, pretrained=pretrained)
    if pretrained_model is None:
        return

    if input_shape[0] != request_resolution:
        try:
            print(">>>> Reload mismatched PositionalEmbedding weights: {} -> {}".format(request_resolution, input_shape[0]))
            bb = keras.models.load_model(pretrained_model)
            for ii in model.layers:
                if isinstance(ii, mismatch_class):
                    print(">>>> Reload layer:", ii.name)
                    model.get_layer(ii.name).load_resized_pos_emb(bb.get_layer(ii.name))
        except:
            pass


def load_weights_with_mismatch(model, weight_file, mismatch_class=None, custom_objects={}):
    model.load_weights(weight_file, by_name=True, skip_mismatch=True)
    if mismatch_class is not None:
        print(">>>> Reload mismatched weights.")
        bb = keras.models.load_model(weight_file, custom_objects=custom_objects)
        for ii in model.layers:
            if isinstance(ii, mismatch_class):
                print(">>>> Reload layer:", ii.name)
                model.get_layer(ii.name).load_resized_pos_emb(bb.get_layer(ii.name))


def state_dict_stack_by_layer(state_dict):
    stacked_state_dict = {}
    for kk, vv in state_dict.items():
        split_kk = kk.split(".")
        vv = vv.numpy()
        if split_kk[-1] in ["num_batches_tracked", "attention_bias_idxs", "attention_biases"]:
            continue
        if split_kk[-1] in ["weight", "bias", "running_mean", "running_var", "gain"]:
            layer_name = ".".join(split_kk[:-1])
            stacked_state_dict.setdefault(layer_name, []).append(vv)
        else:
            stacked_state_dict[kk] = [vv]
    return stacked_state_dict


def keras_reload_stacked_state_dict(model, stacked_state_dict, layer_names_matched_torch, save_name=None):
    import numpy as np

    for kk, tf_layer_name in zip(stacked_state_dict.keys(), layer_names_matched_torch):
        print("torch layer name: {}, tf layer name: {}".format(kk, tf_layer_name))
        tf_layer = model.get_layer(tf_layer_name)
        tf_weights = tf_layer.get_weights()
        torch_weight = stacked_state_dict[kk]
        # print("[{}] torch: {}, tf: {}".format(kk, [ii.shape for ii in torch_weight], [ii.shape for ii in tf_weights]))

        if isinstance(tf_layer, keras.layers.Conv2D):
            torch_weight[0] = np.transpose(torch_weight[0], (2, 3, 1, 0))
            if len(torch_weight) > 2: # gain
                torch_weight[2] = np.squeeze(torch_weight[2])
        elif isinstance(tf_layer, keras.layers.PReLU):
            torch_weight[0] = np.expand_dims(np.expand_dims(torch_weight[0], 0), 0)
        elif isinstance(tf_layer, keras.layers.Conv1D):
            torch_weight[0] = np.transpose(torch_weight[0], (2, 1, 0))
        elif isinstance(tf_layer, keras.layers.Dense):
            # fc layer after flatten, weights need to reshape according to NCHW --> NHWC
            torch_weight[0] = torch_weight[0].T
        print("[{}] torch: {}, tf: {}".format(kk, [ii.shape for ii in torch_weight], [ii.shape for ii in tf_weights]))

        tf_layer.set_weights(torch_weight)

    if save_name is None:
        save_name = model.name + ".h5"
    if len(save_name) != 0:
        print(">>>> Save model to:", save_name)
        model.save(save_name)
