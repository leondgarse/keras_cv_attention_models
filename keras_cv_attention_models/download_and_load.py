import os
import numpy as np
from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers

HDF5_OBJECT_HEADER_LIMIT = 64512


def reload_model_weights(
    model, pretrained_dict, sub_release, pretrained="imagenet", mismatch_class=None, force_reload_mismatch=False, request_resolution=-1, method=None
):
    if not isinstance(pretrained, (str, list, tuple)):
        return
    if isinstance(pretrained, (list, tuple)) or pretrained.endswith(".h5") or pretrained.endswith(".keras") or pretrained.endswith(".pt"):
        pretraineds = pretrained if isinstance(pretrained, (list, tuple)) else [pretrained]
        for pretrained in pretraineds:
            print(">>>> Load pretrained from:", pretrained)
            # model.load_weights(pretrained, by_name=True, skip_mismatch=True)
            load_weights_with_mismatch(model, pretrained, mismatch_class, request_resolution=request_resolution, method=method)
        return pretraineds

    file_hash = pretrained_dict.get(model.name, {}).get(pretrained, None)
    if file_hash is None:
        print(">>>> No pretrained available, model will be randomly initialized")
        return None

    if isinstance(file_hash, dict):
        # file_hash is a dict like {224: "aa", 384: "bb", 480: "cc"}
        if request_resolution == -1:
            model_input_shape = model.input_shape[0] if isinstance(model.input_shape, list) else model.input_shape  # In case multi inputs, just take the 1st
            input_height = model_input_shape[2]  # Either channels_last or channels_first, 2 aixs has to be a shape one
            if input_height is None:  # input_shape is (None, None, 3)
                request_resolution = max(file_hash.keys())
            else:
                request_resolution = min(file_hash.keys(), key=lambda ii: abs(ii - input_height))
        pretrained = "{}_".format(request_resolution) + pretrained
        file_hash = file_hash[request_resolution]
        # print(f"{request_resolution = }, {pretrained = }, {file_hash = }")
    elif request_resolution == -1:
        request_resolution = 224  # Default is 224

    if isinstance(file_hash, (list, tuple)):
        # For large weight file split in several pieces. In format aa.1.h5, aa.2.h5, ...
        pre_url = "https://github.com/leondgarse/keras_cv_attention_models/releases/download/{}/{}_{}.{}.h5"
        is_multi_files = True
    else:
        pre_url = "https://github.com/leondgarse/keras_cv_attention_models/releases/download/{}/{}_{}.h5"
        is_multi_files = False
        file_hash = [file_hash]

    pretrained_models = []
    for id, cur_file_hash in enumerate(file_hash):
        url = pre_url.format(sub_release, model.name, pretrained, id + 1) if is_multi_files else pre_url.format(sub_release, model.name, pretrained)
        file_name = os.path.basename(url)
        try:
            pretrained_model = backend.get_file(file_name, url, cache_subdir="models", file_hash=cur_file_hash)
        except Exception as ee:
            print(f"[Error] will not load weights, url not found or download failed. Error message: {ee}, url: {url}")
            return None
        else:
            print(">>>> Load pretrained from:", pretrained_model)
            # model.load_weights(pretrained_model, by_name=True, skip_mismatch=True)
            load_weights_with_mismatch(model, pretrained_model, mismatch_class, force_reload_mismatch, request_resolution, method)
            pretrained_models.append(pretrained_model)
    return pretrained_model[0] if len(pretrained_models) == 1 else pretrained_models


def load_weights_with_mismatch(model, weight_file, mismatch_class=None, force_reload_mismatch=False, request_resolution=-1, method=None):
    model.load_weights(weight_file, by_name=True, skip_mismatch=True)
    model_input_shape = model.input_shape[0] if isinstance(model.input_shape, list) else model.input_shape  # In case multi inputs, just take the 1st
    if len(model_input_shape) == 4 and backend.image_data_format() == "channels_first":
        input_height, input_width = model_input_shape[2], model_input_shape[3]
    elif len(model_input_shape) == 4:
        input_height, input_width = model_input_shape[1], model_input_shape[2]
    else:
        input_height, input_width = -1, -1

    # if mismatch_class is not None:
    if mismatch_class is not None and (force_reload_mismatch or request_resolution != input_height or request_resolution != input_width):
        try:
            import h5py

            print(">>>> Reload mismatched weights: {} -> {}".format(request_resolution, (input_height, input_width)))
            with h5py.File(weight_file, mode="r") as h5_file:
                weights = h5_file["model_weights"] if "model_weights" in h5_file else h5_file  # full model or weights only

                if isinstance(mismatch_class, (list, tuple)):
                    is_mismatch_class = lambda xx: any([isinstance(ii, mm) for mm in mismatch_class])
                else:
                    is_mismatch_class = lambda xx: isinstance(ii, mismatch_class)

                method_kwarg = {} if method is None else {"method": method}  # None for using class default one
                for ii in model.layers:
                    if is_mismatch_class(ii) and ii.name in weights:
                        print(">>>> Reload layer:", ii.name)
                        ss = weights[ii.name]
                        # ss = {ww.decode().split("/")[-1] : tf.convert_to_tensor(ss[ww]) for ww in ss.attrs['weight_names']}
                        ss = {ww.decode("utf8") if hasattr(ww, "decode") else ww: np.array(ss[ww]) for ww in ss.attrs["weight_names"]}
                        ss = {kk.split("/")[-1]: vv for kk, vv in ss.items()}
                        model.get_layer(ii.name).load_resized_weights(ss, **method_kwarg)
        except Exception as error:
            print("[Error] something went wrong in load_weights_with_mismatch:", error)
            pass


def download_buildin_dataset(data_name, buildin_datasets, cache_subdir="datasets"):
    url, dataset_file = buildin_datasets[data_name]["url"], buildin_datasets[data_name]["dataset_file"]
    file_path = os.path.join(os.path.expanduser("~"), ".keras", cache_subdir, data_name)
    if not os.path.exists(file_path):
        file_path = backend.get_file(origin=url, cache_subdir=cache_subdir, extract=True)  # returned tar file path
    data_name = os.path.join(os.path.dirname(file_path), data_name, dataset_file)
    print(">>>> Buildin dataset, path:", data_name)
    return data_name


""" Save / load h5 weights from keras.saving.legacy.hdf5_format, supports both TF and PyTorch model, and convert_torch_weights_to_h5 """


class H5orKerasFileReader:
    """Read `.h5` or `.keras` format weight file as a dict, `.keras` is the new saving format since TF 2.13
    >>> import kecam
    >>> mm = kecam.models.LLaMA2_15M()
    >>> mm.save('aa.keras')  # Or 'aa.h5'
    >>>
    >>> with kecam.download_and_load.H5orKerasFileReader('aa.keras') as ff:
    >>>     print({kk: [ii.shape for ii in vv] for kk, vv in ff.items()})
    >>>     kecam.download_and_load._load_layer_weights_nested_(mm, ff, debug=True)
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.is_keras_format = self.filepath.endswith(".keras")
        self.read = self.__read_keras_file__ if self.is_keras_format else self.__read_h5_file__

    def __enter__(self):
        return self.read()

    def __exit__(self, type, value, trace):
        self.close()

    def __read_h5_file__(self):
        import h5py

        self.h5_file = h5py.File(self.filepath, "r")
        weights = self.h5_file["model_weights"] if "model_weights" in self.h5_file else self.h5_file  # full model or weights only
        # return {kk: [vv[ww] for ww in vv.attrs["weight_names"]] for kk, vv in weights.items() if len(vv) > 0 and "weight_names" in vv.attrs}
        weights = {kk: weights[kk] for kk in weights.attrs["layer_names"]}
        return {kk: [vv[ww] for ww in vv.attrs["weight_names"]] for kk, vv in weights.items() if len(vv) > 0}

    def __read_keras_file__(self):
        import re, json, zipfile, h5py

        archive = zipfile.ZipFile(self.filepath, "r")
        with archive.open("config.json", "r") as ff:
            model_config = json.loads(ff.read())
        model_weight_file = archive.open("model.weights.h5", "r")
        h5_file = h5py.File(model_weight_file, mode="r")
        self.archive, self.model_weight_file, self.h5_file = archive, model_weight_file, h5_file

        if "_layer_checkpoint_dependencies" not in h5_file:
            print(">>>> [ERROR] weights key '_layer_checkpoint_dependencies' not exists in provided keras file {}".format(self.filepath))
            return {}

        dd = {}
        used_names = {}
        weights = h5_file["_layer_checkpoint_dependencies"]
        for layer in model_config["config"]["layers"]:
            # from keras.src.utils import generic_utils; generic_utils.to_snake_case
            map_name = layer["class_name"]
            map_name = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", map_name)
            map_name = re.sub("([a-z])([A-Z])", r"\1_\2", map_name).lower()
            map_name = ("private" + map_name) if map_name[0] == "_" else map_name

            # _load_container_state
            if map_name in used_names:
                used_names[map_name] += 1
                map_name = f"{map_name}_{used_names[map_name]}"
            else:
                used_names[map_name] = 0

            # print(f"{map_name = }, {layer['name'] = }")
            if map_name in weights and "vars" in weights[map_name]:
                wws = weights[map_name]["vars"]  # This is rather slow on the first time reading [???]
                if len(wws) > 0:
                    dd[layer["name"]] = list(wws.values())
                    # print("  shape:", [ii.shape for ii in dd[layer['name']]])
        return dd

    def close(self):
        self.h5_file.close()
        if self.is_keras_format:
            self.model_weight_file.close()
            self.archive.close()


def _load_layer_weights_nested_(model, weights, skip_mismatch=False, debug=False, prefix=""):
    for layer in model.layers:
        if len(layer.weights) == 0:
            continue
        if debug:
            print(">>>> Load layer weights:", "/".join([prefix, layer.name]) if prefix else layer.name, [ii.name for ii in layer.weights])
        if layer.name not in weights:
            if debug:
                print("Warning: {} not exists in provided h5 weights".format(layer.name))
            continue

        ss = weights[layer.name]
        if hasattr(layer, "layers"):  # nested
            dd = {}
            for weight_name in ss.attrs["weight_names"]:
                dd.setdefault("/".join(weight_name.split("/")[:-1]), []).append(ss[weight_name])
            _load_layer_weights_nested_(layer, dd, skip_mismatch=skip_mismatch, debug=debug, prefix=layer.name)
            continue

        if hasattr(ss, "attrs"):
            ss = [np.array(ss[ww]) for ww in ss.attrs["weight_names"]]
        elif hasattr(ss, "values"):
            ss = [np.array(ww) for ww in ss.values()]
        else:
            ss = [np.array(ww) for ww in ss]

        source_shape, target_shape = [ii.shape for ii in ss], [list(ii.shape) for ii in layer.weights]
        if debug:
            print("     [Before transpose] required weights: {}, provided: {}".format(target_shape, source_shape))
        if skip_mismatch and (len(source_shape) != len(target_shape) or any([np.prod(ii) != np.prod(jj) for ii, jj in zip(source_shape, target_shape)])):
            print("Warning: skip loading weights for layer: {}, required weights: {}, provided: {}".format(layer.name, target_shape, source_shape))
            continue

        if hasattr(layer, "set_weights_channels_last"):
            layer.set_weights_channels_last(ss)
        else:
            layer.set_weights(ss)


def load_weights_from_hdf5_file(filepath, model, skip_mismatch=False, debug=False):
    if not isinstance(filepath, str):  # Regard as dict
        _load_layer_weights_nested_(model, filepath, skip_mismatch=skip_mismatch, debug=debug)
    else:
        with H5orKerasFileReader(filepath) as weights:
            _load_layer_weights_nested_(model, weights, skip_mismatch=skip_mismatch, debug=debug)


def _save_attributes_to_hdf5_group_(group, name, data):
    """Saves attributes (data) of the specified name into the HDF5 group.

    This method deals with an inherent problem of HDF5 file which is not
    able to store data larger than HDF5_OBJECT_HEADER_LIMIT bytes.
    """
    # Check that no item in `data` is larger than `HDF5_OBJECT_HEADER_LIMIT`
    # because in that case even chunking the array would not make the saving possible.
    bad_attributes = [x for x in data if len(x) > HDF5_OBJECT_HEADER_LIMIT]

    # Expecting this to never be true.
    if bad_attributes:
        raise RuntimeError(
            "The following attributes cannot be saved to HDF5 file because " f"they are larger than {HDF5_OBJECT_HEADER_LIMIT} " f"bytes: {bad_attributes}"
        )

    data_npy = np.asarray(data)

    num_chunks = 1
    chunked_data = np.array_split(data_npy, num_chunks)

    # This will never loop forever thanks to the test above.
    while any([x.nbytes > HDF5_OBJECT_HEADER_LIMIT for x in chunked_data]):
        num_chunks += 1
        chunked_data = np.array_split(data_npy, num_chunks)

    if num_chunks > 1:
        for chunk_id, chunk_data in enumerate(chunked_data):
            group.attrs["%s%d" % (name, chunk_id)] = chunk_data
    else:
        group.attrs[name] = data


def _save_to_layer_group_nested_(layer_weights, layer_group, compression=None):
    if hasattr(layer_weights, "layers"):  # nested
        sub_model = layer_weights
        weight_names, weight_values = [], []
        for layer in sub_model.layers:
            weight_names += [ww.name.encode("utf8") for ww in layer.weights]
            weight_values += layer.get_weights_channels_last() if hasattr(layer, "get_weights_channels_last") else layer.get_weights()
        # for sub_layer in layer_weights.layers:
        #     sub_layer_group = layer_group.create_group(sub_layer.name)
        #     _save_to_layer_group_nested_(sub_layer, sub_layer_group)
        # return
    elif isinstance(layer_weights, layers.Layer) or (hasattr(layers, "Shape") and isinstance(layer_weights, layers.Shape)):
        layer = layer_weights
        weight_names = [ww.name.encode("utf8") for ww in layer.weights]
        weight_values = layer.get_weights_channels_last() if hasattr(layer, "get_weights_channels_last") else layer.get_weights()
        # print("{}:".format(layer.name), [ii.shape for ii in weight_values])
    else:
        weight_names = [ww.encode("utf8") for ww in layer_weights]
        weight_values = list(layer_weights.values())

    _save_attributes_to_hdf5_group_(layer_group, "weight_names", weight_names)
    for name, val in zip(weight_names, weight_values):
        # print(f"{name = }, {val.shape = }")
        param_dset = layer_group.create_dataset(name, val.shape, dtype=val.dtype, compression=compression, chunks=True)
        if not val.shape:
            # scalar
            param_dset[()] = val
        else:
            param_dset[:] = val


def save_weights_to_hdf5_file(filepath, model, compression=None, layer_start=None, layer_end=None):
    """Saves the weights of a list of layers to a HDF5 file.
    - compression: refer `import h5py; help(h5py.File.create_dataset)`
          and `from h5py._hl import dataset; help(dataset.make_new_dset)`.
    """
    import h5py

    if isinstance(model, dict):
        weights_dict = model
    else:
        weights_dict = {layer.name: layer for layer in model.layers[layer_start:layer_end]}

    with h5py.File(filepath, "w") as h5_file:
        _save_attributes_to_hdf5_group_(h5_file, "layer_names", [layer_name.encode("utf8") for layer_name in weights_dict])
        h5_file.attrs["backend"] = backend.backend().encode("utf8")
        h5_file.attrs["keras_version"] = "2.15.0".encode("utf8")  # Crucial for some weight shape transpose while using in TF >= 2.15.0
        for layer_name, layer_weights in weights_dict.items():
            layer_group = h5_file.create_group(layer_name)
            # print(layer_name, layer_weights)
            _save_to_layer_group_nested_(layer_weights, layer_group, compression=compression)


def convert_torch_weights_to_h5(
    source_pt_path,
    save_path="AUTO",
    skip_weights=["num_batches_tracked"],
    name_convert_funcs=None,
    name_convert_map=None,
    weight_convert_funcs=None,
    to_fp16=False,
):
    """
    Examples:
    # Save weights is torch pt -> convert to h5 -> load back
    >>> os.environ['KECAM_BACKEND'] = 'torch'
    >>> from keras_cv_attention_models import download_and_load, models
    >>> mm = models.AotNet50()
    >>> mm.save('test.pt')
    >>> conv_func = lambda target_name, ww: ww.transpose([2, 3, 1, 0]) if target_name.endswith("conv") and len(ww.shape) == 4 else ww
    >>> _ = download_and_load.convert_torch_weights_to_h5('test.pt', 'test.h5', weight_convert_funcs=[conv_func])
    >>> mm.load("test.h5")
    """
    import os
    import torch

    source_state_dict = torch.load(source_pt_path, map_location=torch.device("cpu")) if isinstance(source_pt_path, str) else source_pt_path
    source_state_dict = source_state_dict.get("state_dict", source_state_dict.get("model", source_state_dict))
    name_convert_funcs = [] if name_convert_funcs is None else name_convert_funcs
    name_convert_map = {} if name_convert_map is None else name_convert_map
    weight_convert_funcs = [] if weight_convert_funcs is None else weight_convert_funcs
    save_path = (os.path.splitext(source_pt_path)[0] + ".h5" if isinstance(source_pt_path, str) else "converted.h5") if save_path == "AUTO" else save_path

    """ keras_reload_stacked_state_dict """
    stacked_state_dict = {}
    for source_name in list(source_state_dict.keys()):
        source_weight = source_state_dict.pop(source_name)  # pop out in case taking too much memory
        # target_name = source_name.replace("layers.", "blocks.")
        # target_name = target_name[len("model."):] if target_name.startswith("model.") else target_name
        if any([source_name.endswith(ii) for ii in skip_weights]):
            continue
        source_name_split = source_name.split(".")
        source_layer_name, source_weight_name = ".".join(source_name_split[:-1]), source_name_split[-1]

        target_layer_name = source_layer_name
        for additional_name_func in name_convert_funcs:
            target_layer_name = additional_name_func(target_layer_name)
        for kk, vv in name_convert_map.items():
            if kk in target_layer_name:
                target_layer_name = target_layer_name.replace(kk, vv)
        print(">>>> layer {} -> {}, {}: {}".format(source_layer_name, target_layer_name, source_weight_name, source_weight.shape))

        source_weight = source_weight.numpy() if hasattr(source_weight, "numpy") else source_weight
        source_weight = source_weight.astype("float16") if to_fp16 else source_weight
        for additional_weight_func in weight_convert_funcs:
            # source_weight = [ii.T if len(ii) == 2 else ii for ii in source_weight]
            source_weight = additional_weight_func(target_layer_name, source_weight)
        print("    weight: {}, shape: {}".format(source_weight_name, source_weight.shape))
        stacked_state_dict.setdefault(target_layer_name, {}).update({source_layer_name + "/" + source_weight_name: source_weight})

    if save_path:
        print(">>>> Save to:", save_path)
        save_weights_to_hdf5_file(save_path, stacked_state_dict)
    return stacked_state_dict


""" Convert PyTorch weights """


def state_dict_stack_by_layer(state_dict, skip_weights=["num_batches_tracked"], unstack_weights=[]):
    stacked_state_dict = {}
    for kk, vv in state_dict.items():
        if kk[-2] == ":":
            kk = kk[:-2]  # Keras weight name like "..../weight:0"
        split_token = "/" if "/" in kk else "."
        split_kk = kk.split(split_token)
        vv = (vv.float().numpy() if hasattr(vv, "float") else vv.numpy()) if hasattr(vv, "numpy") else vv
        if split_kk[-1] in skip_weights:
            continue

        if split_kk[-1] in unstack_weights:
            stacked_state_dict[kk] = [vv]
        else:
            # split_kk[-1] in ["weight", "bias", "running_mean", "running_var", "gain"]
            layer_name = split_token.join(split_kk[:-1])
            stacked_state_dict.setdefault(layer_name, []).append(vv)
    return stacked_state_dict


def match_layer_names_with_torch(target_names, tail_align_dict={}, full_name_align_dict={}, tail_split_position=2):
    layer_names_matched_torch = [""] * len(target_names)
    raw_id_dict = {ii: id for id, ii in enumerate(target_names)}

    # is_tail_align_dict_split_by_stack = len(tail_align_dict) > 0 and isinstance(list(tail_align_dict.values())[0], dict)
    for id, ii in enumerate(target_names):
        name_split = ii.split("_")
        stack_name = name_split[0]
        head_name = "_".join(name_split[:tail_split_position])
        tail_name = "_".join(name_split[tail_split_position:])
        # cur_tail_align_dict = tail_align_dict[stack_name] if is_tail_align_dict_split_by_stack else tail_align_dict
        cur_tail_align_dict = tail_align_dict.get(stack_name, tail_align_dict)
        # print("id = {}, ii = {}, stack_name = {}, tail_name = {}".format(id, ii, stack_name, tail_name))
        if ii in full_name_align_dict:
            align = full_name_align_dict[ii]
            if isinstance(align, str):
                align = raw_id_dict[align]
            else:
                align = id + align if align < 0 else align
            layer_names_matched_torch.insert(align, ii)
            layer_names_matched_torch.pop(-1)
        elif tail_name in cur_tail_align_dict:
            align = cur_tail_align_dict[tail_name]
            if isinstance(align, str):
                align = raw_id_dict[head_name + "_" + align]
            else:
                align = id + align if align < 0 else align
            layer_names_matched_torch.insert(align, ii)
            layer_names_matched_torch.pop(-1)
        else:
            layer_names_matched_torch[id] = ii
    return layer_names_matched_torch


def align_layer_names_multi_stage(target_names, tail_align_dict={}, full_name_align_dict={}, tail_split_position=2, specific_match_func=None, verbose=1):
    if isinstance(tail_split_position, int):
        tail_align_dict, full_name_align_dict, tail_split_position = [tail_align_dict], [full_name_align_dict], [tail_split_position]
    full_name_align_dict = full_name_align_dict if isinstance(full_name_align_dict, (list, tuple)) else [full_name_align_dict]
    # for ii, jj, kk in zip(tail_align_dict, full_name_align_dict, tail_split_position):
    for idx in range(max(len(tail_split_position), len(full_name_align_dict))):
        cur_tail = tail_align_dict[idx] if idx < len(tail_align_dict) else {}
        cur_full = full_name_align_dict[idx] if idx < len(full_name_align_dict) else {}
        cur_split = tail_split_position[idx] if idx < len(tail_split_position) else tail_split_position[-1]
        if verbose > 0:
            print(">>>> tail_align_dict:", cur_tail)
            print(">>>> full_name_align_dict:", cur_full)
            print(">>>> tail_split_position:", cur_split)
        target_names = match_layer_names_with_torch(target_names, cur_tail, cur_full, cur_split)

    if specific_match_func is not None:
        target_names = specific_match_func(target_names)
    return target_names


def keras_reload_stacked_state_dict(model, stacked_state_dict, layer_names_matched_torch, additional_transfer={}, save_name=None, verbose=1):
    import numpy as np

    for kk, tf_layer_name in zip(stacked_state_dict.keys(), layer_names_matched_torch):
        if verbose > 0:
            print("  torch layer name: {}, tf layer name: {}".format(kk, tf_layer_name))
        tf_layer = model.get_layer(tf_layer_name)
        tf_weights = tf_layer.get_weights()
        torch_weight = stacked_state_dict[kk]
        if verbose > 0:
            print("    Before: [{}] torch: {}, tf: {}".format(kk, [ii.shape for ii in torch_weight], [ii.shape for ii in tf_weights]))

        if isinstance(tf_layer, layers.DepthwiseConv2D):  # DepthwiseConv2D is instance of Conv2D, put it first
            torch_weight[0] = np.transpose(torch_weight[0], (2, 3, 0, 1))
        elif isinstance(tf_layer, layers.Conv2D):
            torch_weight[0] = np.transpose(torch_weight[0], (2, 3, 1, 0))
            if len(torch_weight) > 2:  # gain
                torch_weight[2] = np.squeeze(torch_weight[2])
        elif isinstance(tf_layer, layers.PReLU):
            torch_weight[0] = np.expand_dims(np.expand_dims(torch_weight[0], 0), 0)
        elif isinstance(tf_layer, layers.Conv1D):
            torch_weight[0] = np.transpose(torch_weight[0], (2, 1, 0))
        elif isinstance(tf_layer, layers.Dense):
            # [Note] if it's fc layer after flatten, weights need to reshape according to NCHW --> NHWC
            torch_weight[0] = torch_weight[0].T

        for add_layer, add_transfer in additional_transfer.items():
            if isinstance(add_layer, str):
                if tf_layer.name.endswith(add_layer):
                    torch_weight = add_transfer(torch_weight)
            elif isinstance(tf_layer, add_layer):
                torch_weight = add_transfer(torch_weight)
        if verbose > 0:
            print("    After: [{}] torch: {}, tf: {}".format(kk, [ii.shape for ii in torch_weight], [ii.shape for ii in tf_weights]))

        tf_layer.set_weights(torch_weight)

    if save_name is None:
        save_name = model.name + ".h5"
    if len(save_name) != 0:
        if verbose > 0:
            print()
            print(">>>> Save model to:", save_name)
        model.save(save_name)


def try_save_pth_and_onnx(torch_model, save_pth=True, save_onnx=True, input_shape=(10, 3, 224, 224), dtype="float32", save_name=None):
    import torch
    import numpy as np

    save_name = torch_model.__class__.__name__ if save_name is None else save_name
    dummy_inputs = torch.from_numpy(np.random.uniform(size=input_shape).astype(dtype))
    if save_pth:
        output_name = save_name + ".pth"
        traced_cell = torch.jit.trace(torch_model, (dummy_inputs), strict=False)
        torch.jit.save(traced_cell, output_name)
        print(">>>> Saved to:", output_name)

    if save_onnx:
        output_name = save_name + ".onnx"
        torch.onnx.export(
            model=torch_model,
            args=dummy_inputs,
            f=output_name,
            verbose=False,
            keep_initializers_as_inputs=True,
            training=torch.onnx.TrainingMode.PRESERVE,
            do_constant_folding=False,
            opset_version=13,
        )
        print(">>>> Saved to:", output_name)


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
    specific_match_func=None,
    save_name=None,
    do_convert=True,
    do_predict=True,
    verbose=1,
):
    """Convert and load weights from a PyTorch model into a Keras model, with optional prediction verification.

    The main work is aligning the weight name order between torch and keras models. The conversion has 3 stages:
      1. **Weight collection** (``state_dict_stack_by_layer``): Groups torch state_dict entries by layer name,
         filtering with ``skip_weights`` and separating entries listed in ``unstack_weights``.
      2. **Name alignment** (``align_layer_names_multi_stage``): Reorders keras layer names to match the torch
         stacked weight order, using ``tail_align_dict``, ``full_name_align_dict``, and ``specific_match_func``.
      3. **Weight transfer** (``keras_reload_stacked_state_dict``): Applies standard shape transforms (Conv2D
         transpose, Dense transpose, etc.) plus ``additional_transfer`` overrides, sets weights, and saves.

    Args:
        torch_model: PyTorch model instance, state_dict (dict), or path to a ``.pth``/``.pt`` weights file.
        keras_model: Target Keras model. If ``None``, only prints the torch weight summary without converting.
        input_shape: Input spatial dimensions for prediction test, default ``(224, 224)``.

        skip_weights: Weight name suffixes to drop entirely during collection.
            Example: ``["num_batches_tracked", "relative_position_index"]``.
        unstack_weights: Weight name suffixes to keep as individual entries (not grouped with their parent
            layer). Example: ``["cls_token", "pos_embed", "gamma_1", "gamma_2"]``.

        tail_align_dict: Reposition keras layers by tail name match. Format: ``{tail_name: offset}``.
            Negative offset moves the layer earlier in the ordered list. Can be scoped by stack name using
            nested dicts: ``{"stack3": {"attn_gamma": -6}}``. The "tail" is determined by ``tail_split_position``.
        full_name_align_dict: Reposition keras layers by exact full name. Value can be a negative int (relative
            offset), positive int (absolute position), or a string (position of another named layer).
        tail_split_position: Where to split keras layer name into head/tail for ``tail_align_dict`` matching.
            Default ``2``, meaning ``stack1_block1`` is head and ``attn_gamma`` is tail. Set to ``1`` for
            head=``stack1`` and tail=``block1_attn_gamma``.
        specific_match_func: A function ``f(names) -> reordered_names`` for full control over name ordering,
            bypassing ``tail_align_dict`` and ``full_name_align_dict``. Useful when dicts can't express the mapping.

        additional_transfer: Custom weight transforms applied after standard Conv2D/Dense transposes.
            Keys can be layer classes (matched via ``isinstance``) or name suffix strings (matched via
            ``endswith``). Values are ``lambda ww: [transformed_weights]``.
        save_name: Output ``.h5`` file path. Defaults to ``{keras_model.name}_imagenet.h5``.
        do_convert: If ``True``, perform the actual weight loading and saving. If ``False``, only print
            weight lists for inspection (useful for debugging alignment).
        do_predict: If ``True`` (and ``do_convert=True``), run prediction on a test image and print results
            for both torch and keras models to verify correctness.
        verbose: Verbosity level. ``1`` prints weight lists and progress, ``0`` is silent.
    """
    import torch
    import numpy as np

    from keras_cv_attention_models import test_images
    from keras_cv_attention_models.common_layers import PreprocessInput
    from keras_cv_attention_models.imagenet.eval_func import decode_predictions

    input_shape = input_shape[:2] if keras_model is None or None in keras_model.input_shape[1:] else keras_model.input_shape[1:-1]
    if isinstance(torch_model, str):
        print(">>>> Reload Torch weight file:", torch_model)
        torch_model = torch.load(torch_model, map_location=torch.device("cpu"))
        torch_model = torch_model.get("model", torch_model.get("state_dict", torch_model))
    is_state_dict = isinstance(torch_model, dict)

    """ Chelsea the cat  """
    do_predict = do_predict and do_convert
    if do_predict:
        pp = PreprocessInput(input_shape=input_shape, rescale_mode="torch")
        img = pp(test_images.cat()).numpy()

    if not is_state_dict:
        _ = torch_model.eval()
        state_dict = torch_model.state_dict()

        if do_predict:
            try:
                # from torchsummary import summary
                # summary(torch_model, (3, *input_shape), device="cpu")
                """Torch Run predict"""
                out = torch_model(torch.from_numpy(img.copy().transpose(0, 3, 1, 2).astype("float32")))
                out = out.detach().cpu().numpy()
                out = out[None] if len(out.shape) == 1 else out
                # out = tf.nn.softmax(out).numpy()  # If classifier activation is not softmax
                torch_out = decode_predictions(out)
            except Exception as error:
                print("[Error] something went wrong in running PyTorch model prediction:", error)
                pass
    else:
        state_dict = torch_model
    do_convert = do_convert and len(state_dict) > 0

    """ Convert torch weights """
    # torch_params = {kk: (np.cumproduct(vv.shape)[-1] if len(vv.shape) != 0 else 1) for kk, vv in state_dict.items() if ".num_batches_tracked" not in kk}
    stacked_state_dict = state_dict_stack_by_layer(state_dict, skip_weights=skip_weights, unstack_weights=unstack_weights)
    if verbose > 0:
        print(">>>> torch_model total_parameters :", np.sum([np.sum([np.prod(jj.shape) for jj in ii]) for ii in stacked_state_dict.values()]))
    aa = {kk: [1 if isinstance(jj, float) else jj.shape for jj in vv] for kk, vv in stacked_state_dict.items()}
    if verbose > 0:
        print(">>>> Torch weights:")
        _ = [print("  '{}': {}".format(kk, vv)) for kk, vv in aa.items()]
        print()

    if keras_model is None:
        return

    """ Keras model weights """
    target_names = [ii.name for ii in keras_model.layers if len(ii.weights) != 0]
    aa = {keras_model.get_layer(ii).name: [jj.shape.as_list() for jj in keras_model.get_layer(ii).weights] for ii in target_names}
    if verbose > 0:
        print(">>>> keras_model total_parameters :", np.sum([np.sum([int(np.prod(jj)) for jj in ii]) for ii in aa.values()]))
        print(">>>> Keras weights:")
        _ = [print("  '{}': {}".format(kk, vv)) for kk, vv in aa.items()]
        print()

    """ Load torch weights and save h5 """
    if len(tail_align_dict) != 0 or len(full_name_align_dict) != 0 or specific_match_func is not None:
        aligned_names = align_layer_names_multi_stage(target_names, tail_align_dict, full_name_align_dict, tail_split_position, specific_match_func, verbose)
        aa = {keras_model.get_layer(ii).name: [jj.shape.as_list() for jj in keras_model.get_layer(ii).weights] for ii in aligned_names}
        if verbose > 0:
            print(">>>> Keras weights matched torch:")
            _ = [print("  '{}': {}".format(kk, vv)) for kk, vv in aa.items()]
            print()
    else:
        aligned_names = target_names

    if not do_convert:
        return

    save_name = save_name if save_name is not None else keras_model.name + "_imagenet.h5"
    if verbose > 0:
        print(">>>> Keras reload torch weights:")
    keras_reload_stacked_state_dict(keras_model, stacked_state_dict, aligned_names, additional_transfer, save_name=save_name, verbose=verbose)
    if verbose > 0:
        print()

    """ Keras run predict """
    if do_predict:
        try:
            pred = keras_model(img).numpy()
            # pred = tf.nn.softmax(pred).numpy()  # If classifier activation is not softmax
            print(">>>> Keras model prediction:", decode_predictions(pred)[0])
            print()
            if not is_state_dict:
                print(">>>> Torch model prediction:", torch_out)
        except:
            pass
