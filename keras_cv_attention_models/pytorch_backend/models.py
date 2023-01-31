import torch
import h5py
import numpy as np
from torch import nn
from keras_cv_attention_models import backend
from keras_cv_attention_models.pytorch_backend import layers

HDF5_OBJECT_HEADER_LIMIT = 64512


class Model(nn.Module):
    """
    >>> from keras_cv_attention_models.pytorch_backend import layers, models
    >>> inputs = layers.Input([3, 224, 224])
    >>> pre = layers.Conv2D(32, 3, padding="SAME", name="deep_pre_conv")(inputs)
    >>> deep_1 = layers.Conv2D(32, 3, padding="SAME", name="deep_1_1_conv")(pre)
    >>> deep_1 = layers.Conv2D(32, 3, padding="SAME", name="deep_1_2_conv")(deep_1)
    >>> deep_2 = layers.Conv2D(32, 3, padding="SAME", name="deep_2_conv")(pre)
    >>> deep = layers.Add(name="deep_add")([deep_1, deep_2])
    >>> short = layers.Conv2D(32, 3, padding="SAME", name="short_conv")(inputs)
    >>> outputs = layers.Add(name="outputs")([short, deep])
    >>> mm = models.Model(inputs, outputs)
    >>> print(mm(torch.ones([1, 3, 224, 224])).shape)
    >>> # torch.Size([1, 32, 224, 224])
    >>> mm.summary()

    >>> from keras_cv_attention_models.mlp_family import mlp_mixer
    >>> mm = mlp_mixer.MLPMixerB16(input_shape=(3, 224, 224))
    >>> >>>> Load pretrained from: /home/leondgarse/.keras/models/mlp_mixer_b16_imagenet.h5
    >>> from PIL import Image
    >>> from skimage.data import chelsea # Chelsea the cat
    >>> from keras_cv_attention_models.imagenet import decode_predictions
    >>> imm = Image.fromarray(chelsea()).resize(mm.input_shape[2:])
    >>> pred = mm(torch.from_numpy(np.array(imm)).permute([2, 0, 1])[None] / 255)
    >>> decode_predictions(pred.detach())

    >>> mm.decode_predictions(mm(mm.preprocess_input(chelsea())))
    """

    num_instances = 0  # Count instances

    @classmethod
    def __count__(cls):
        cls.num_instances += 1

    def __init__(self, inputs, outputs, name=None, **kwargs):
        super().__init__()
        self.name = "model_{}".format(self.num_instances) if name == None else name
        self.input_node, self.output_node = inputs, outputs
        self.outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
        self.inputs = inputs if isinstance(inputs, (list, tuple)) else [inputs]
        self.num_outputs = len(self.outputs)
        self.output_names = [ii.name for ii in self.outputs]
        self.input_shape = inputs.shape
        self.create_forward_pipeline()
        self.eval()  # Set eval mode by default

    def create_forward_pipeline(self, **kwargs):
        forward_pipeline, layers, outputs = [], {}, []
        dfs_queue = []
        for ii in self.inputs:
            dfs_queue.extend(ii.next_nodes)
        branch_nodes = set()  # node names with multi outputs, or before node with multi inputs
        while len(dfs_queue) > 0:
            cur_node = dfs_queue.pop(-1)
            # print(f">>>> {cur_node.name = }, {cur_node.pre_node_names = }, {cur_node.next_node_names = }")
            if len(cur_node.pre_nodes) > 1 and not all([ii.name in branch_nodes for ii in cur_node.pre_nodes]):
                continue
            dfs_queue.extend(cur_node.next_nodes)
            # print(cur_node.name)

            forward_pipeline.append(cur_node)
            setattr(self, cur_node.name, cur_node.layer)
            layers[cur_node.name] = cur_node.layer

            if len(cur_node.next_nodes) > 1:
                branch_nodes.add(cur_node.name)

            if any([len(ii.pre_nodes) > 1 for ii in cur_node.next_nodes]):
                branch_nodes.add(cur_node.name)

            if cur_node.name in self.output_names:
                outputs.append(cur_node.name)
            if all([ii in outputs for ii in self.output_names]):
                break
        self.forward_pipeline, self.branch_nodes, self.__layers__ = forward_pipeline, branch_nodes, layers

    def forward(self, inputs, **kwargs):
        # print(' -> '.join([ii.name for ii in self.forward_pipeline]))
        pre_node = self.inputs[0]
        if isinstance(inputs, (list, tuple)):  # Multi inputs in list or tuple format
            pre_output = inputs[0]
            branch_record = {kk.name: vv for kk, vv in zip(self.inputs, inputs)}
        elif isinstance(inputs, dict):  # Multi inputs in dict format
            pre_output = inputs[pre_node.name]
            branch_record = inputs
        else:  # Single input
            pre_output = inputs
            branch_record = {pre_node.name: inputs}

        outputs = {}
        for node in self.forward_pipeline:
            # print(f">>>> {node.name = }, {node.pre_node_names = }, {node.next_node_names = }")
            if len(node.pre_nodes) > 1:
                output = node.callable([branch_record[ii] for ii in node.pre_node_names])
            elif node.pre_nodes[0].name != pre_node.name:
                output = node.callable(branch_record[node.pre_node_names[0]])
            else:
                output = node.callable(pre_output)
            # print(f"     {output.shape = }")

            if node.name in self.branch_nodes:
                branch_record[node.name] = output
            if node.name in self.output_names:
                outputs[node.name] = output
            pre_node, pre_output = node, output
        return [outputs[ii] for ii in self.output_names] if self.num_outputs != 1 else outputs[self.output_names[0]]

    @property
    def layers(self):
        return list(self.__layers__.values())

    def get_layer(self, layer_name):
        return self.__layers__[layer_name]

    def load_weights(self, filepath, by_name=True, skip_mismatch=False):
        ff = h5py.File(filepath, mode="r")
        with h5py.File(filepath, "r") as h5_file:
            load_weights_from_hdf5_group(h5_file, self, skip_mismatch=skip_mismatch)

    def save_weights(self, filepath=None):
        with h5py.File(filepath if filepath else self.name + ".h5", "w") as h5_file:
            save_weights_to_hdf5_group(h5_file, self)

    def summary(self):
        from torchsummary import summary

        summary(self, tuple(self.input_shape[1:]))

    def export_onnx(self, filepath=None, *kwargs):
        torch.onnx.export(self, torch.ones([1, *self.input_shape[1:]]), self.name + ".onnx", *kwargs)
        print("Exported onnx:", filepath if filepath else self.name + ".onnx")

    def export_pth(self, filepath=None, *kwargs):
        traced_cell = torch.jit.trace(self, (torch.ones([1, *self.input_shape[1:]])))
        torch.jit.save(traced_cell, self.name + ".pth", *kwargs)
        print("Exported pth:", filepath if filepath else self.name + ".pth")


""" Save / load h5 weights from keras.saving.legacy.hdf5_format """


def load_weights_from_hdf5_group(h5_file, model, skip_mismatch=False):
    weights = h5_file["model_weights"] if "model_weights" in h5_file else h5_file  # full model or weights only

    for tt in model.layers:
        if len(tt.weights) == 0:
            continue
        # print(">>>> Load layer weights:", tt.name, {ii.name: ii.shape for ii in tt.weights})
        ss = weights[tt.name]
        # ss = {ww.decode().split("/")[-1] : tf.convert_to_tensor(ss[ww]) for ww in ss.attrs['weight_names']}
        # ss = {ww.decode("utf8") if hasattr(ww, "decode") else ww: np.array(ss[ww]) for ww in ss.attrs["weight_names"]}
        # ss = {kk.split("/")[-1]: vv for kk, vv in ss.items()}
        ss = [np.array(ss[ww]) for ww in ss.attrs["weight_names"]]
        # print("Before:", [ii.shape for ii in ss])
        if skip_mismatch and np.prod(tt.weights[0].shape) != np.prod(ss[0].shape):
            print("Warning: skip loading weights for layer: {}, required weights: {}, provided: {}".format(tt.name, tt.weights[0].shape, ss[0].shape))
            continue

        if hasattr(tt, "set_weights_channels_last"):
            tt.set_weights_channels_last(ss)
        else:
            tt.set_weights(ss)


def save_subset_weights_to_hdf5_group(group, weight_names, weight_values):
    """Save top-level weights of a model to a HDF5 group."""
    save_attributes_to_hdf5_group(group, "weight_names", weight_names)
    for name, val in zip(weight_names, weight_values):
        param_dset = group.create_dataset(name, val.shape, dtype=val.dtype)
        if not val.shape:
            # scalar
            param_dset[()] = val
        else:
            param_dset[:] = val


def save_attributes_to_hdf5_group(group, name, data):
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


def save_weights_to_hdf5_group(h5_file, model):
    """Saves the weights of a list of layers to a HDF5 group."""
    save_attributes_to_hdf5_group(h5_file, "layer_names", [layer.name.encode("utf8") for layer in model.layers])
    h5_file.attrs["backend"] = backend.backend().encode("utf8")
    for layer in sorted(model.layers, key=lambda x: x.name):
        layer_group = h5_file.create_group(layer.name)
        weight_names = [ww.name.encode("utf8") for ww in layer.weights]
        weight_values = layer.get_weights_channels_last() if hasattr(layer, "get_weights_channels_last") else layer.get_weights()
        save_subset_weights_to_hdf5_group(layer_group, weight_names, weight_values)
