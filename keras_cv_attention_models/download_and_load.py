import os
import tensorflow as tf
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

    if input_shape[0] != request_resolution or input_shape[1] != request_resolution:  # May be non-square
        try:
            import h5py

            print(">>>> Reload mismatched PositionalEmbedding weights: {} -> {}".format(request_resolution, input_shape[0]))
            ff = h5py.File(pretrained_model, mode="r")
            weights = ff["model_weights"]
            for ii in model.layers:
                if isinstance(ii, mismatch_class) and ii.name in weights:
                    print(">>>> Reload layer:", ii.name)
                    ss = weights[ii.name]
                    # ss = {ww.decode().split("/")[-1] : tf.convert_to_tensor(ss[ww]) for ww in ss.attrs['weight_names']}
                    ss = {ww.decode("utf8") if hasattr(ww, "decode") else ww: tf.convert_to_tensor(ss[ww]) for ww in ss.attrs["weight_names"]}
                    ss = {kk.split("/")[-1]: vv for kk, vv in ss.items()}
                    model.get_layer(ii.name).load_resized_pos_emb(ss)
            ff.close()

        # print(">>>> Reload mismatched PositionalEmbedding weights: {} -> {}".format(request_resolution, input_shape[0]))
        # bb = keras.models.load_model(pretrained_model)
        # for ii in model.layers:
        #     if isinstance(ii, mismatch_class):
        #         print(">>>> Reload layer:", ii.name)
        #         model.get_layer(ii.name).load_resized_pos_emb(bb.get_layer(ii.name))
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


def state_dict_stack_by_layer(state_dict, skip_weights=["num_batches_tracked"], unstack_weights=[]):
    stacked_state_dict = {}
    for kk, vv in state_dict.items():
        split_kk = kk.split(".")
        vv = vv.numpy()
        if split_kk[-1] in skip_weights:
            continue

        if split_kk[-1] in unstack_weights:
            stacked_state_dict[kk] = [vv]
        else:
            # split_kk[-1] in ["weight", "bias", "running_mean", "running_var", "gain"]
            layer_name = ".".join(split_kk[:-1])
            stacked_state_dict.setdefault(layer_name, []).append(vv)
    return stacked_state_dict


def match_layer_names_with_torch(target_names, tail_align_dict={}, full_name_align_dict={}, tail_split_position=2):
    layer_names_matched_torch = [""] * len(target_names)
    # is_tail_align_dict_split_by_stack = len(tail_align_dict) > 0 and isinstance(list(tail_align_dict.values())[0], dict)
    for id, ii in enumerate(target_names):
        name_split = ii.split("_")
        stack_name = name_split[0]
        tail_name = "_".join(name_split[tail_split_position:])
        # cur_tail_align_dict = tail_align_dict[stack_name] if is_tail_align_dict_split_by_stack else tail_align_dict
        cur_tail_align_dict = tail_align_dict.get(stack_name, tail_align_dict)
        # print("id = {}, ii = {}, stack_name = {}, tail_name = {}".format(id, ii, stack_name, tail_name))
        if ii in full_name_align_dict:
            align = full_name_align_dict[ii]
            layer_names_matched_torch.insert(id + align, ii)
            layer_names_matched_torch.pop(-1)
        elif tail_name in cur_tail_align_dict:
            align = cur_tail_align_dict[tail_name]
            layer_names_matched_torch.insert(id + align, ii)
            layer_names_matched_torch.pop(-1)
        else:
            layer_names_matched_torch[id] = ii
    return layer_names_matched_torch


def keras_reload_stacked_state_dict(model, stacked_state_dict, layer_names_matched_torch, additional_transfer={}, save_name=None):
    import numpy as np

    for kk, tf_layer_name in zip(stacked_state_dict.keys(), layer_names_matched_torch):
        print("  torch layer name: {}, tf layer name: {}".format(kk, tf_layer_name))
        tf_layer = model.get_layer(tf_layer_name)
        tf_weights = tf_layer.get_weights()
        torch_weight = stacked_state_dict[kk]
        print("    Before: [{}] torch: {}, tf: {}".format(kk, [ii.shape for ii in torch_weight], [ii.shape for ii in tf_weights]))

        if isinstance(tf_layer, keras.layers.DepthwiseConv2D):  # DepthwiseConv2D is instance of Conv2D, put it first
            torch_weight[0] = np.transpose(torch_weight[0], (2, 3, 0, 1))
        elif isinstance(tf_layer, keras.layers.Conv2D):
            torch_weight[0] = np.transpose(torch_weight[0], (2, 3, 1, 0))
            if len(torch_weight) > 2:  # gain
                torch_weight[2] = np.squeeze(torch_weight[2])
        elif isinstance(tf_layer, keras.layers.PReLU):
            torch_weight[0] = np.expand_dims(np.expand_dims(torch_weight[0], 0), 0)
        elif isinstance(tf_layer, keras.layers.Conv1D):
            torch_weight[0] = np.transpose(torch_weight[0], (2, 1, 0))
        elif isinstance(tf_layer, keras.layers.Dense):
            # [Note] if it's fc layer after flatten, weights need to reshape according to NCHW --> NHWC
            torch_weight[0] = torch_weight[0].T

        for add_layer, add_transfer in additional_transfer.items():
            if isinstance(add_layer, str):
                if tf_layer.name == add_layer:
                    torch_weight = add_transfer(torch_weight)
            elif isinstance(tf_layer, add_layer):
                torch_weight = add_transfer(torch_weight)
        print("    After: [{}] torch: {}, tf: {}".format(kk, [ii.shape for ii in torch_weight], [ii.shape for ii in tf_weights]))

        tf_layer.set_weights(torch_weight)

    if save_name is None:
        save_name = model.name + ".h5"
    if len(save_name) != 0:
        print()
        print(">>>> Save model to:", save_name)
        model.save(save_name)


def keras_reload_from_torch_model(
    torch_model,  # Torch model, Torch state_dict wights or Torch model weights file
    keras_model=None,
    input_shape=(224, 224),
    skip_weights=["num_batches_tracked"],
    unstack_weights=[],
    tail_align_dict={},
    full_name_align_dict={},
    tail_split_position=2,
    additional_transfer={},
    save_name=None,
    do_convert=True,
):
    import torch
    import numpy as np
    import tensorflow as tf
    from skimage.data import chelsea

    if isinstance(torch_model, str):
        print(">>>> Reload Torch weight file:", torch_model)
        torch_model = torch.load(torch_model, map_location=torch.device("cpu"))
        torch_model = torch_model.get("state_dict", torch_model)
    is_state_dict = isinstance(torch_model, dict)

    """ Chelsea the cat  """
    img = chelsea()
    img = keras.applications.imagenet_utils.preprocess_input(tf.image.resize(img, input_shape), mode="torch").numpy()

    if not is_state_dict:
        from torchsummary import summary

        _ = torch_model.eval()
        summary(torch_model, (3, *input_shape))
        state_dict = torch_model.state_dict()

        """ Torch Run predict """
        out = torch_model(torch.from_numpy(np.expand_dims(img.transpose(2, 0, 1), 0).astype("float32")))
        out = out.detach().cpu().numpy()
        # out = tf.nn.softmax(out).numpy()  # If classifier activation is not softmax
        torch_out = keras.applications.imagenet_utils.decode_predictions(out)
    else:
        state_dict = torch_model

    """ Convert torch weights """
    torch_params = {kk: (np.cumproduct(vv.shape)[-1] if len(vv.shape) != 0 else 1) for kk, vv in state_dict.items() if ".num_batches_tracked" not in kk}
    print(">>>> torch_model total_parameters :", np.sum(list(torch_params.values())))
    stacked_state_dict = state_dict_stack_by_layer(state_dict, skip_weights=skip_weights, unstack_weights=unstack_weights)
    aa = {kk: [1 if isinstance(jj, float) else jj.shape for jj in vv] for kk, vv in stacked_state_dict.items()}
    print(">>>> Torch weights:")
    _ = [print("  '{}': {}".format(kk, vv)) for kk, vv in aa.items()]
    print()

    if keras_model is None:
        return

    """ Keras model weights """
    target_names = [ii.name for ii in keras_model.layers if len(ii.weights) != 0]
    aa = {keras_model.get_layer(ii).name: [jj.shape.as_list() for jj in keras_model.get_layer(ii).weights] for ii in target_names}
    print(">>>> Keras weights:")
    _ = [print("  '{}': {}".format(kk, vv)) for kk, vv in aa.items()]
    print()

    if not do_convert:
        return

    """ Load torch weights and save h5 """
    layer_names_matched_torch = match_layer_names_with_torch(target_names, tail_align_dict, full_name_align_dict, tail_split_position)
    aa = {keras_model.get_layer(ii).name: [jj.shape.as_list() for jj in keras_model.get_layer(ii).weights] for ii in layer_names_matched_torch}
    print(">>>> Keras weights matched torch:")
    _ = [print("  '{}': {}".format(kk, vv)) for kk, vv in aa.items()]
    print()

    save_name = save_name if save_name is not None else keras_model.name + "_imagenet.h5"
    print(">>>> Keras reload torch weights:")
    keras_reload_stacked_state_dict(keras_model, stacked_state_dict, layer_names_matched_torch, additional_transfer, save_name=save_name)
    print()

    """ Keras run predict """
    try:
        pred = keras_model(tf.expand_dims(img, 0)).numpy()
        # pred = tf.nn.softmax(pred).numpy()  # If classifier activation is not softmax
        print(">>>> Keras model prediction:", keras.applications.imagenet_utils.decode_predictions(pred)[0])
        print()
        if not is_state_dict:
            print(">>>> Torch model prediction:", torch_out)
    except:
        pass
