import torch
import numpy as np
from torch import nn
from keras_cv_attention_models.pytorch_backend import layers


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

    def create_forward_pipeline(self, **kwargs):
        forward_pipeline, layers, outputs = [], {}, []
        dfs_queue = []
        for ii in self.inputs:
            dfs_queue.extend(ii.next_nodes)
        # processed_branch_inputs = set()
        branch_nodes = set()  # {layer_name: value}, node names with multi outputs, or before node with multi inputs
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
        if isinstance(inputs, (list, tuple)):
            pre_output = inputs[0]
            branch_record = {kk.name: vv for kk, vv in zip(self.inputs, inputs)}
        elif isinstance(inputs, dict):
            pre_output = inputs[pre_node.name]
            branch_record = inputs
        else:
            pre_output = inputs
            branch_record = {pre_node.name: inputs}

        outputs = {}
        for node in self.forward_pipeline:
            # print(f">>>> {node.name = }, {node.pre_node_names = }, {node.next_node_names = }")
            if len(node.pre_nodes) > 1 and isinstance(node.layer, layers._Merge):
                # Use module for _Merge layers, or will meet error: AttributeError: 'list' object has no attribute 'size'
                output = node.module([branch_record[ii] for ii in node.pre_node_names])
            elif len(node.pre_nodes) > 1:
                output = node.layer([branch_record[ii] for ii in node.pre_node_names])
            elif node.pre_nodes[0].name != pre_node.name:
                output = node.layer(branch_record[node.pre_node_names[0]])
            else:
                output = node.layer(pre_output)

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
        import h5py

        ff = h5py.File(filepath, mode="r")
        weights = ff["model_weights"] if "model_weights" in ff else ff  # full model or weights only

        for tt in self.layers:
            if len(tt.weights) == 0:
                continue
            # print(">>>> Load layer weights:", tt.name, [ii.shape for ii in tt.weights])
            ss = weights[tt.name]
            # ss = {ww.decode().split("/")[-1] : tf.convert_to_tensor(ss[ww]) for ww in ss.attrs['weight_names']}
            # ss = {ww.decode("utf8") if hasattr(ww, "decode") else ww: np.array(ss[ww]) for ww in ss.attrs["weight_names"]}
            # ss = {kk.split("/")[-1]: vv for kk, vv in ss.items()}
            ss = [np.array(ss[ww]) for ww in ss.attrs["weight_names"]]
            # print("Before:", [ii.shape for ii in ss])
            if skip_mismatch and np.prod(tt.weights[0].shape) != np.prod(ss[0].shape):
                print("Warning: skip loading weights for layer: {}, required weights: {}, provided: {}".format(tt.name, tt.weights[0].shape, ss[0].shape))
                continue

            if isinstance(tt, layers.DepthwiseConv2D):
                ss[0] = np.transpose(ss[0], (2, 3, 0, 1))
            elif isinstance(tt, layers.Conv2D):
                ss[0] = np.transpose(ss[0], (3, 2, 0, 1))
            elif isinstance(tt, layers.PReLU):
                ss[0] = np.squeeze(ss[0])
            elif isinstance(tt, layers.Conv1D):
                ss[0] = np.transpose(ss[0], (2, 1, 0))
            elif isinstance(tt, layers.Dense):
                ss[0] = ss[0].T
            elif tt.weights[0].shape != ss[0].shape:
                # ss[0] = ss[0].reshape(tt.weights[0].shape)
                ss = [ii.reshape(jj.shape) for ii, jj in zip(ss, tt.weights)]
            # print("After:", [ii.shape for ii in ss])
            tt.set_weights(ss)
        ff.close()

    def summary(self):
        from torchsummary import summary

        summary(self, tuple(self.input_shape[1:]))

    def export_onnx(self, *kwargs):
        torch.onnx.export(self, torch.ones([1, *self.input_shape[1:]]), self.name + '.onnx', *kwargs)
        print("Exported onnx:", self.name + '.onnx')

    def export_pth(self, *kwargs):
        traced_cell = torch.jit.trace(self, (torch.ones([1, *self.input_shape[1:]])))
        torch.jit.save(traced_cell, self.name + '.pth', *kwargs)
        print("Exported pth:", self.name + '.pth')
