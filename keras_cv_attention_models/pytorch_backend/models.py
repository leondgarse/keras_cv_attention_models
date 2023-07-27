import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from contextlib import nullcontext
from keras_cv_attention_models import backend
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

        self.outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
        self.output_shape = [tuple(ii.shape) for ii in self.outputs] if isinstance(outputs, (list, tuple)) else tuple(outputs.shape)
        self.output_names = [ii.name for ii in self.outputs]

        self.inputs = inputs if isinstance(inputs, (list, tuple)) else [inputs]
        self.input_shape = [tuple(ii.shape) for ii in self.inputs] if isinstance(inputs, (list, tuple)) else tuple(inputs.shape)
        self.input_names = [ii.name for ii in self.inputs]

        self.num_outputs = len(self.outputs)
        self.create_forward_pipeline()
        self.eval()  # Set eval mode by default
        self.debug = False

        # Add compile and fit
        self.trainer = Trainer(model=self)
        self._dataset_gen_ = self.trainer._dataset_gen_
        self.compile = self.trainer.compile
        self.fit = self.trainer.fit

        # Add summary, export_onnx, export_pth
        self.exporter = Exporter(model=self)
        self._create_fake_input_data_ = self.exporter._create_fake_input_data_
        self.summary = self.exporter.summary
        self.export_onnx = self.exporter.export_onnx
        self.export_pth = self.exporter.export_pth

    def create_forward_pipeline(self, **kwargs):
        forward_pipeline, layers, outputs = [], {}, []
        dfs_queue = []
        intra_nodes_ref = {}  # node names, and how many times they should be used
        for ii in self.inputs:
            dfs_queue.extend(list(set(ii.next_nodes)))
            intra_nodes_ref[ii.name] = len(ii.next_nodes)
        while len(dfs_queue) > 0:
            cur_node = dfs_queue.pop(-1)
            # print(cur_node.name, cur_node.next_nodes)
            if len(cur_node.pre_nodes) > 1 and not all([ii.name in intra_nodes_ref for ii in cur_node.pre_nodes]):
                continue
            if cur_node.name in intra_nodes_ref:
                raise ValueError("All nodes name should be unique: cur_node: {}, intra_nodes_ref: {}".format(cur_node.name, list(intra_nodes_ref.keys())))

            # `set` is used here in case current node outputs multi times to next node, like `layers.Add()([inputs, inputs])`.
            # dfs_queue.extend(list(set(cur_node.next_nodes)))
            dfs_queue.extend([ii for ii in cur_node.next_nodes if ii not in dfs_queue])
            # print(f"{dfs_queue = }")

            forward_pipeline.append(cur_node)
            setattr(self, cur_node.name.replace(".", "_"), cur_node.callable)
            layers[cur_node.layer.name] = cur_node.layer

            # print(f"{cur_node.name = }, {len(cur_node.next_nodes) = }")
            intra_nodes_ref[cur_node.name] = len(cur_node.next_nodes)
            if cur_node.name in self.output_names:
                intra_nodes_ref[cur_node.name] = intra_nodes_ref.get(cur_node.name, 0) + 1
            if all([ii in intra_nodes_ref for ii in self.output_names]):
                break
        self.forward_pipeline, self.intra_nodes_ref, self.__layers__ = forward_pipeline, intra_nodes_ref, layers

    def forward(self, inputs, **kwargs):
        # print(' -> '.join([ii.name for ii in self.forward_pipeline]))
        if isinstance(inputs, (list, tuple)):  # Multi inputs in list or tuple format
            intra_nodes = {kk: [vv] * self.intra_nodes_ref[kk] for kk, vv in zip(self.input_names, inputs)}
        elif isinstance(inputs, dict):  # Multi inputs in dict format
            intra_nodes = {kk: [vv] * self.intra_nodes_ref[kk] for kk, vv in inputs.items()}
        else:  # Single input
            intra_nodes = {self.input_names[0]: [inputs] * self.intra_nodes_ref[self.input_names[0]]}

        for node in self.forward_pipeline:
            if self.debug:
                print(">>>> [{}], pre_node_names: {}, next_node_names: {}".format(node.name, node.pre_node_names, node.next_node_names))
                print("     intra_nodes:", {kk: len(vv) for kk, vv in intra_nodes.items() if len(vv) > 0})

            if len(node.pre_nodes) > 1:
                cur_inputs = [intra_nodes[ii].pop() for ii in node.pre_node_names]
            else:
                cur_inputs = intra_nodes[node.pre_node_names[0]].pop()

            if self.debug:
                print("     inputs.shape:", [np.shape(ii) for ii in cur_inputs] if len(node.pre_nodes) > 1 else np.shape(cur_inputs))

            output = node.callable(cur_inputs)
            intra_nodes[node.name] = [output] * self.intra_nodes_ref[node.name]
            if self.debug:
                print("     output.shape:", np.shape(output))
                print("     intra_nodes:", {kk: len(vv) for kk, vv in intra_nodes.items() if len(vv) > 0})
        return [intra_nodes[ii][0] for ii in self.output_names] if self.num_outputs != 1 else intra_nodes[self.output_names[0]][0]

    @property
    def layers(self):
        return list(self.__layers__.values())

    @property
    def weights(self):
        skips = ["num_batches_tracked", "total_ops", "total_params"]
        buffers = [layers.Weight(name=name, value=value) for name, value in self.named_buffers() if not name.split(".")[-1] in skips]
        parameters = [layers.Weight(name=name, value=value) for name, value in self.named_parameters()]
        return parameters + buffers

    def get_layer(self, layer_name):
        return self.__layers__[layer_name]

    def load_weights(self, filepath, by_name=True, skip_mismatch=False):
        from keras_cv_attention_models.download_and_load import load_weights_from_hdf5_file

        load_weights_from_hdf5_file(filepath, self, skip_mismatch=skip_mismatch, debug=self.debug)

    def save_weights(self, filepath=None, **kwargs):
        from keras_cv_attention_models.download_and_load import save_weights_to_hdf5_file

        save_weights_to_hdf5_file(filepath if filepath else self.name + ".h5", self, **kwargs)

    def count_params(self):
        total_params = sum([np.prod(ii.shape) for ii in self.state_dict().values() if len(ii.shape) != 0])
        trainable_params = sum([np.prod(list(ii.shape)) for ii in self.parameters()])
        non_trainable_params = total_params - trainable_params
        print("Total params: {:,} | Trainable params: {:,} | Non-trainable params:{:,}".format(total_params, trainable_params, non_trainable_params))
        return total_params

    def set_debug(self, debug=True):
        self.debug = debug
        print(">>>> debug: {}".format(self.debug))


class Trainer:
    def __init__(self, model):
        self.model = model

    def compile(self, optimizer="RMSprop", loss=None, metrics=None, loss_weights=None, grad_accumulate=1, grad_max_norm=-1, **kwargs):
        self.optimizer = getattr(torch.optim, optimizer)(self.model.parameters()) if isinstance(optimizer, str) else optimizer
        self.loss = torch.functional.F.cross_entropy if loss is None else loss
        self.loss_weights, self.grad_accumulate, self.grad_max_norm = loss_weights, grad_accumulate, grad_max_norm

        self.metrics = {} if metrics is None else (metrics if isinstance(metrics, dict) else {ii.__name__: ii for ii in metrics})
        self.metrics.update({"loss": self.loss})

        device = next(self.model.parameters()).device
        device_type = device.type
        if device_type == "cpu":
            scaler = torch.cuda.amp.GradScaler(enabled=False)
            global_context = nullcontext()
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=True)
            global_context = torch.amp.autocast(device_type=device_type, dtype=torch.float16)
        self.device, self.device_type, self.scaler, self.global_context = device, device_type, scaler, global_context

    def fit(
        self, x=None, y=None, batch_size=None, epochs=1, verbose="auto", callbacks=None, validation_data=None, initial_epoch=0, steps_per_epoch=None, **kwargs
    ):
        bar_format = "{n_fmt}/{total_fmt} [{bar:30}] - ETA: {elapsed}<{remaining} {rate_fmt}{postfix}{desc}"

        for epoch in range(initial_epoch, epochs):
            print("Epoch {}/{}".format(epoch + 1, epochs))
            self.model.train()

            train_dataset, total = self._dataset_gen_(x, y, batch_size=batch_size)
            self.optimizer.zero_grad()
            passed_batches = 0
            process_bar = tqdm(enumerate(train_dataset), total=total, bar_format=bar_format, ascii=".>>=")
            mean_loss = 0.0
            for batch, (xx, yy) in process_bar:
                if isinstance(xx, (list, tuple)):
                    xx = [self._convert_data_(ii) for ii in xx]
                else:
                    xx = slf._convert_data_(xx)
                if isinstance(yy, (list, tuple)):
                    yy = [self._convert_data_(ii) for ii in yy]
                else:
                    yy = slf._convert_data_(yy)

                with self.global_context:
                    out = self.model(xx)
                    loss = self.loss(out, yy)

                self.scaler.scale(loss).backward()

                passed_batches += 1
                if passed_batches >= self.grad_accumulate:
                    self.scaler.unscale_(self.optimizer)
                    if self.grad_max_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_max_norm)  # clip gradients
                    self.scaler.step(self.optimizer)  # self.optimizer.step()
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    passed_batches = 0
                # print(">>>> Epoch {}, batch: {}, loss: {:.4f}".format(epoch, batch, loss.item()))
                mean_loss = (mean_loss * batch + loss) / (batch + 1)
                process_bar.desc = " - loss: {:.4f}".format(mean_loss)  # process_bar.set_description automatically add a : on the tail
                process_bar.refresh()
            print()

            if validation_data is not None:
                val_dataset, total = self._dataset_gen_(validation_data, batch_size=batch_size)

    def _convert_data_(self, data):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        if self.device_type == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            data = data.pin_memory().to(self.device, non_blocking=True)
        return data

    def _dataset_gen_(self, x=None, y=None, batch_size=None):
        if isinstance(x, (list, tuple)) and len(x) == 2 and y is None:
            x, y = x[0], x[1]

        if hasattr(x, "element_spec"):  # TF datsets
            data_shape = x.element_spec[0].shape
            if self.model.input_shape is not None and data_shape[-1] == self.model.input_shape[1]:
                perm = [0, len(data_shape) - 1] + list(range(1, len(data_shape) - 1))  # [0, 3, 1, 2]
                train_dataset = ((xx.transpose(perm), yy) for xx, yy in x.as_numpy_iterator())
            else:
                train_dataset = x.as_numpy_iterator()
        elif isinstance(x, np.ndarray) or isinstance(x, torch.Tensor):
            assert y is not None
            num_batches = xx.shape[0] if batch_size is None else int(np.ceil(xx.shape[0] / batch_size))

            def _convert_tensor(data, id):
                cur = data[id * batch_size : (id + 1) * batch_size] if batch_size is not None else data[id]
                cur = torch.from_numpy(cur) if isinstance(cur, np.ndarray) else cur
                cur = cur.float() if cur.dtype == torch.float64 else cur
                cur = cur.long() if cur.dtype == torch.int32 else cur
                return cur

            train_dataset = ((_convert_tensor(x, id), _convert_tensor(y, id)) for id in range(num_batches))
        else:  # generator or torch.utils.data.DataLoader
            train_dataset = x
        total = len(x) if hasattr(x, "__len__") else None
        return train_dataset, total


class Exporter:
    def __init__(self, model):
        self.model = model

    def _create_fake_input_data_(self, input_shape=None, batch_size=1):
        input_shape = self.model.input_shape if input_shape is None else input_shape
        input_shapes = input_shape if isinstance(input_shape[0], (list, tuple)) else [input_shape]  # Convert to list of input_shpae
        model_inputs = self.model.inputs
        assert len(input_shapes) == len(model_inputs), "provided input_shape: {} not matching model.inputs: {} in length".format(input_shape, model_inputs)

        input_datas = []
        for input_shape, model_input in zip(input_shapes, model_inputs):
            input_shape = list(input_shape).copy()
            if len(input_shape) == len(model_input.shape) - 1:
                input_shape = [batch_size] + input_shape
            assert len(input_shape) == len(model_input.shape), "provided input_shape={} not match with input={} in rank".format(input_shape, model_input.shape)

            if input_shape[0] is None or input_shape[0] == -1:
                input_shape[0] = batch_size
            if None in input_shape or -1 in input_shape:
                print("[WARNING] dynamic shape value in input_shape={}, set to 32".format(input_shape))
                input_shape = [32 if ii is None or ii == -1 else ii for ii in input_shape]

            dtype = model_input.dtype or torch.get_default_dtype()
            dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
            input_datas.append(torch.ones(input_shape, dtype=dtype))
        print(">>>> input_shape: {}, dtype: {}".format([ii.shape for ii in input_datas], [ii.dtype for ii in input_datas]))
        return input_datas

    def summary(self, input_shape=None, **kwargs):
        from torchinfo import summary

        input_datas = self._create_fake_input_data_(input_shape)
        print(summary(self.model, input_data=input_datas if len(self.model.inputs) == 1 else [input_datas], **kwargs))

    def export_onnx(self, filepath=None, input_shape=None, batch_size=1, simplify=False, **kwargs):
        input_datas = self._create_fake_input_data_(input_shape, batch_size=batch_size)

        dynamic_axes = kwargs.pop("dynamic_axes", None)
        input_names = kwargs.pop("input_names", self.model.input_names)
        output_names = kwargs.pop("output_names", self.model.output_names)
        if dynamic_axes is None and (batch_size is None or batch_size == -1):
            print("Set dynamic batch size")
            dynamic_axes = {ii: {0: "-1"} for ii in input_names}
            dynamic_axes.update({ii: {0: "-1"} for ii in output_names})

        filepath = (self.model.name + ".onnx") if filepath is None else (filepath if filepath.endswith(".onnx") else (filepath + ".onnx"))
        torch.onnx.export(self.model, input_datas, filepath, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes, **kwargs)
        print("Exported onnx:", filepath)

        if simplify:
            import onnx, onnxsim

            print("Running onnxsim.simplify...")
            tt = onnx.load(filepath)
            tt, check = onnxsim.simplify(tt)
            if check:
                with open(filepath, "wb") as ff:
                    ff.write(tt.SerializeToString())
                print("Exported simplified onnx:", filepath)
            else:
                print("[Error] failed to simplify onnx:", filepath)

    def export_pth(self, filepath=None, input_shape=None, batch_size=1, **kwargs):
        input_datas = self._create_fake_input_data_(input_shape, batch_size=batch_size)

        traced_cell = torch.jit.trace(self.model, example_inputs=input_datas if len(self.model.inputs) == 1 else [input_datas])
        filepath = (self.model.name + ".pth") if filepath is None else (filepath if filepath.endswith(".pth") else (filepath + ".pth"))
        torch.jit.save(traced_cell, filepath, **kwargs)
        print("Exported pth:", filepath)
