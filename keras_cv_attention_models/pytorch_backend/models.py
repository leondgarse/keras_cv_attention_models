import torch
import numpy as np
from tqdm import tqdm
from torch import nn
from contextlib import nullcontext
from keras_cv_attention_models import backend
from keras_cv_attention_models.pytorch_backend import layers, callbacks, metrics


class Model(nn.Module):
    """
    Examples:
    # Build custom model
    >>> os.environ["KECAM_BACKEND"] = "torch"
    >>> import torch
    >>> from keras_cv_attention_models.pytorch_backend import layers, models
    >>> inputs = layers.Input([3, 32, 32])
    >>> pre = layers.Conv2D(32, 3, padding="SAME", name="deep_pre_conv")(inputs)
    >>> deep_1 = layers.Conv2D(32, 3, padding="SAME", name="deep_1_1_conv")(pre)
    >>> deep_1 = layers.Conv2D(32, 3, padding="SAME", name="deep_1_2_conv")(deep_1)
    >>> deep_2 = layers.Conv2D(32, 3, padding="SAME", name="deep_2_conv")(pre)
    >>> deep = layers.Add(name="deep_add")([deep_1, deep_2])
    >>> short = layers.Conv2D(32, 3, padding="SAME", name="short_conv")(inputs)
    >>> outputs = layers.Add(name="outputs")([short, deep])
    >>> outputs = layers.GlobalAveragePooling2D()(outputs)
    >>> outputs = layers.Dense(10)(outputs)
    >>> mm = models.Model(inputs, outputs)
    >>> print(mm(torch.ones([1, 3, 32, 32])).shape)
    >>> # torch.Size([1, 10])
    >>> mm.summary()

    # Compile and fit
    >>> xx, yy = torch.rand([1000, 3, 32, 32]), torch.functional.F.one_hot(torch.randint(0, 10, size=[1000]), 10).float()
    >>> mm.compile(loss=loss, metrics='acc')  # Using default cross_entropy loss
    >>> mm.fit(xx, yy, epochs=2)

    # Run buildin model
    >>> from keras_cv_attention_models.mlp_family import mlp_mixer
    >>> mm = mlp_mixer.MLPMixerB16(input_shape=(3, 224, 224))
    >>> # >>>> Load pretrained from: /home/leondgarse/.keras/models/mlp_mixer_b16_imagenet.h5
    >>> from PIL import Image
    >>> from skimage.data import chelsea # Chelsea the cat
    >>> from keras_cv_attention_models.imagenet import decode_predictions
    >>> imm = Image.fromarray(chelsea()).resize(mm.input_shape[2:])
    >>> pred = mm(torch.from_numpy(np.array(imm)).permute([2, 0, 1])[None] / 255)
    >>> print(decode_predictions(pred.detach()))
    >>> # or just use preset preprocess_input
    >>> print(mm.decode_predictions(mm(mm.preprocess_input(chelsea()))))
    """

    num_instances = 0  # Count instances

    @classmethod
    def __count__(cls):
        cls.num_instances += 1

    def __init__(self, inputs, outputs, name=None, **kwargs):
        super().__init__()
        self.name = "model_{}".format(self.num_instances) if name == None else name
        self.nodes = None

        self.output = outputs
        self.outputs = outputs if isinstance(outputs, (list, tuple)) else [outputs]
        self.output_shape = [tuple(ii.shape) for ii in self.outputs] if isinstance(outputs, (list, tuple)) else tuple(outputs.shape)
        self.output_names = [ii.name for ii in self.outputs]

        self.input = inputs
        self.inputs = inputs if isinstance(inputs, (list, tuple)) else [inputs]
        self.input_shape = [tuple(ii.shape) for ii in self.inputs] if isinstance(inputs, (list, tuple)) else tuple(inputs.shape)
        self.input_names = [ii.name for ii in self.inputs]

        self.num_outputs = len(self.outputs)
        self.create_forward_pipeline()
        self.eval()  # Set eval mode by default
        self.debug = False

        # Add compile and fit
        self.trainer = Trainer(model=self)
        self.compile = self.trainer.compile
        self.fit = self.trainer.fit

        # Add summary, export_onnx, export_pth
        self.exporter = Exporter(model=self)
        self.summary = self.exporter.summary
        self.export_onnx = self.exporter.export_onnx
        self.export_pth = self.exporter.export_pth

    def create_forward_pipeline(self, **kwargs):
        forward_pipeline, all_layers, outputs = [], {}, []
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

            # `set` is used here in case current node outputs multi times to next node, like `all_layers.Add()([inputs, inputs])`.
            # dfs_queue.extend(list(set(cur_node.next_nodes)))
            dfs_queue.extend([ii for ii in cur_node.next_nodes if ii not in dfs_queue])
            # print(f"{dfs_queue = }")

            forward_pipeline.append(cur_node)
            setattr(self, cur_node.name.replace(".", "_"), cur_node.callable)
            all_layers[cur_node.layer.name] = cur_node.layer

            # print(f"{cur_node.name = }, {len(cur_node.next_nodes) = }")
            intra_nodes_ref[cur_node.name] = len(cur_node.next_nodes)
            if cur_node.name in self.output_names:
                intra_nodes_ref[cur_node.name] = intra_nodes_ref.get(cur_node.name, 0) + 1
            if all([ii in intra_nodes_ref for ii in self.output_names]):
                break
        self.forward_pipeline, self.intra_nodes_ref, self.__layers__ = forward_pipeline, intra_nodes_ref, all_layers

    def forward(self, inputs, **kwargs):
        if isinstance(inputs, layers.GraphNode) or (isinstance(inputs, (list, tuple)) and any([isinstance(ii, layers.GraphNode) for ii in inputs])):
            return self.graphnode_forward(inputs)

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

    def graphnode_forward(self, inputs):
        self.input_shape = [() if isinstance(ii, (int, float)) else ii.shape for ii in inputs] if isinstance(inputs, (list, tuple)) else inputs.shape
        cur_node = layers.GraphNode(self.output_shape, name=self.name if self.nodes is None else (self.name + "_{}".format(len(self.nodes))))
        cur_node.callable = self
        cur_node.layer = self
        cur_node.set_pre_nodes(inputs)

        inputs = inputs if isinstance(inputs, (list, tuple)) else [inputs]
        for ii in inputs:
            if isinstance(ii, layers.GraphNode):
                ii.set_next_nodes(cur_node)

        if self.nodes is None:
            self.nodes = [cur_node]
            self.node = cur_node
        else:
            self.nodes.append(cur_node)
        return cur_node

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
    """
    Examples:
    # compile and fit on buildin models
    >>> os.environ["KECAM_BACKEND"] = "torch"
    >>> import torch
    >>> from keras_cv_attention_models import aotnet
    >>> mm = aotnet.AotNet50(num_classes=10, input_shape=(32, 32, 3))
    >>> mm.compile(metrics='acc')  # Using default cross_entropy loss
    >>> xx, yy = torch.rand([256, 3, 32, 32]), torch.functional.F.one_hot(torch.randint(0, 10, size=[256]), 10).float()
    >>> mm.fit(xx, yy, epochs=2)

    # compile and fit on custom models
    >>> os.environ["KECAM_BACKEND"] = "torch"
    >>> import torch
    >>> from keras_cv_attention_models.backend import layers, models
    >>> inputs = layers.Input([3, 32, 32])
    >>> nn = layers.Conv2D(32, 3, 2, padding='same')(inputs)
    >>> nn = layers.GlobalAveragePooling2D()(nn)
    >>> nn = layers.Dense(10)(nn)
    >>> mm = models.Model(inputs, nn)
    >>> mm.summary()
    >>> xx, yy = torch.rand([1000, 3, 32, 32]), torch.functional.F.one_hot(torch.randint(0, 10, size=[1000]), 10).float()
    >>> loss = lambda y_pred, y_true: (y_true - y_pred.float()).abs().mean()
    >>> mm.compile(optimizer="AdamW", loss=loss, metrics='acc')
    >>> mm.fit(xx, yy, epochs=2)

    # compile and fit on raw torch models
    >>> os.environ["KECAM_BACKEND"] = "torch"
    >>> import torch
    >>> from keras_cv_attention_models.backend import models
    >>> torch_model = torch.nn.Sequential(
    >>>     torch.nn.Conv2d(3, 32, 3, 2, 1), torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(32, 10)
    >>> )
    >>> mm = models.Trainer(torch_model)
    >>> xx, yy = torch.rand([1000, 3, 32, 32]), torch.functional.F.one_hot(torch.randint(0, 10, size=[1000]), 10).float()
    >>> loss = torch.functional.F.mse_loss
    >>> mm.compile(optimizer=torch.optim.SGD(torch_model.parameters(), lr=0.1), loss=loss, metrics='acc')
    >>> mm.fit(xx, yy, batch_size=64, epochs=2)
    """

    def __init__(self, model, input_shape=None, output_names={}):
        self.model = model
        self.input_shape = getattr(self.model, "input_shape", None) if input_shape is None else input_shape
        self.output_names = getattr(self.model, "output_names", None) if output_names is None else output_names

    def init_metrics(self, cur_metrics=None):
        if cur_metrics is None:
            metrics_names, cur_metrics = [], []
        elif isinstance(cur_metrics, str):
            metrics_names, cur_metrics = [cur_metrics], [cur_metrics]
        elif isinstance(cur_metrics, (list, tuple)):
            metrics_names, cur_metrics = [ii if isinstance(ii, str) else ii.name for ii in cur_metrics], list(cur_metrics)
        elif isinstance(cur_metrics, dict):
            metrics_names, cur_metrics = list(cur_metrics.keys()), list(cur_metrics.values())
            # [TODO] match metrics_names with self.output_names
        else:
            metrics_names, cur_metrics = [cur_metrics.name], [cur_metrics]
        cur_metrics = [metrics.BUILDIN_METRICS[ii]() if isinstance(ii, str) else ii for ii in cur_metrics]
        return metrics_names, cur_metrics

    def compile(self, optimizer="RMSprop", loss=None, metrics=None, loss_weights=None, grad_accumulate=1, grad_max_norm=-1, **kwargs):
        self.optimizer = getattr(torch.optim, optimizer)(self.model.parameters()) if isinstance(optimizer, str) else optimizer
        self.loss = torch.functional.F.cross_entropy if loss is None else loss
        self.loss_weights, self.grad_accumulate, self.grad_max_norm = loss_weights, grad_accumulate, grad_max_norm
        self.metrics_names, self.metrics = self.init_metrics(metrics)

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
        self, x=None, y=None, batch_size=32, epochs=1, verbose="auto", callbacks=None, validation_data=None, initial_epoch=0, steps_per_epoch=None, **kwargs
    ):
        callbacks = callbacks or []
        [ii.set_model(self) for ii in callbacks if ii.model is None]
        self.hists = {"loss": []}

        bar_format = "{n_fmt}/{total_fmt} [{bar:30}] - ETA: {elapsed}<{remaining} {rate_fmt}{postfix}{desc}"
        for epoch in range(initial_epoch, epochs):
            print("Epoch {}/{}".format(epoch + 1, epochs))
            logs = {}  # Can be used as global value between different callbacks
            [ii.on_epoch_begin(self, epoch, logs) for ii in callbacks]
            [ii.reset_state() for ii in self.metrics]

            self.model.train()
            self.optimizer.zero_grad()

            avg_loss, accumulate_passed_batches = 0.0, 0
            train_dataset, total = self._dataset_gen_(x, y, batch_size=batch_size)
            process_bar = tqdm(enumerate(train_dataset), total=total, bar_format=bar_format, ascii=".>>=")
            for batch, (xx, yy) in process_bar:
                [ii.on_train_batch_begin(self, batch, logs) for ii in callbacks]

                xx = [self._convert_data_(ii) for ii in xx] if isinstance(xx, (list, tuple)) else self._convert_data_(xx)
                yy = [self._convert_data_(ii) for ii in yy] if isinstance(yy, (list, tuple)) else self._convert_data_(yy)
                with self.global_context:
                    out = self.model(xx)
                    loss = self.loss(out, yy)
                self.scaler.scale(loss).backward()

                accumulate_passed_batches += 1
                if accumulate_passed_batches >= self.grad_accumulate:
                    self.scaler.unscale_(self.optimizer)
                    if self.grad_max_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.grad_max_norm)  # clip gradients
                    self.scaler.step(self.optimizer)  # self.optimizer.step()
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    accumulate_passed_batches = 0
                # print(">>>> Epoch {}, batch: {}, loss: {:.4f}".format(epoch, batch, loss.item()))
                avg_loss += loss
                process_bar.desc = " - loss: {:.4f}".format(avg_loss / (batch + 1))  # process_bar.set_description automatically add a : on the tail

                if isinstance(yy, (list, tuple)):
                    [[ii.update_state(cur_out, cur_yy) for ii in self.metrics] for cur_out, cur_yy in zip(out, yy)]
                else:
                    [ii.update_state(out, yy) for ii in self.metrics]
                metrics_desc = [" - {}: {:.4f}".format(name, metric.result()) for name, metric in zip(self.metrics_names, self.metrics)]
                process_bar.desc += "".join(metrics_desc)
                process_bar.refresh()

                [ii.on_train_batch_end(self, batch, logs) for ii in callbacks]
            [ii.on_epoch_end(self, epoch, logs) for ii in callbacks]
            self.hists["loss"].append((avg_loss.item() if hasattr(avg_loss, "item") else avg_loss) / (batch + 1))
            for name, metric in zip(self.metrics_names, self.metrics):
                metric_result = metric.result()
                self.hists.setdefault(name, []).append(metric_result.item() if hasattr(metric_result, "item") else metric_result)

            if validation_data is not None:
                [ii.on_test_begin(self, epoch, logs) for ii in callbacks]
                val_dataset, total = self._dataset_gen_(validation_data, batch_size=batch_size)
                [ii.on_test_end(self, epoch, logs) for ii in callbacks]

            print()

    def _convert_data_(self, data):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        if self.device_type == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            data = data.pin_memory().to(self.device, non_blocking=True)
        return data

    def _dataset_gen_(self, x=None, y=None, batch_size=32):
        if isinstance(x, (list, tuple)) and len(x) == 2 and y is None:
            xx, yy = x[0], x[1]
        else:
            xx, yy = x, y

        if hasattr(xx, "element_spec"):  # TF datsets
            data_shape = xx.element_spec[0].shape
            if self.input_shape is not None and data_shape[-1] == self.input_shape[1]:
                perm = [0, len(data_shape) - 1] + list(range(1, len(data_shape) - 1))  # [0, 3, 1, 2]
                train_dataset = ((ii.transpose(perm), jj) for ii, jj in xx.as_numpy_iterator())
            else:
                train_dataset = xx.as_numpy_iterator()
            total = len(xx)
        elif isinstance(xx, np.ndarray) or isinstance(xx, torch.Tensor):
            assert yy is not None
            num_batches = xx.shape[0] if batch_size is None else int(np.ceil(xx.shape[0] / batch_size))

            def _convert_tensor(data, id):
                cur = data[id * batch_size : (id + 1) * batch_size] if batch_size is not None else data[id]
                cur = torch.from_numpy(cur) if isinstance(cur, np.ndarray) else cur
                cur = cur.float() if cur.dtype == torch.float64 else cur
                cur = cur.long() if cur.dtype == torch.int32 else cur
                return cur

            train_dataset = ((_convert_tensor(xx, id), _convert_tensor(yy, id)) for id in range(num_batches))
            total = num_batches
        else:  # generator or torch.utils.data.DataLoader
            train_dataset = xx
            total = len(xx)
        return train_dataset, total


class Exporter:
    def __init__(self, model, input_shapes=None, input_dtypes=None, input_names=None, output_names=None, name=None):
        self.model = model
        self.name = name or getattr(model, "name", "model")
        self.input_names = input_names or getattr(model, "input_names", None)
        self.output_names = output_names or getattr(model, "output_names", None)

        self.input_shapes = input_shapes or ([ii.shape for ii in model.inputs] if hasattr(model, "inputs") else None)
        self.input_dtypes = input_dtypes or ([ii.dtype for ii in model.inputs] if hasattr(model, "inputs") else None)

    def _create_fake_input_data_(self, input_shape=None, batch_size=1):
        input_shape = self.input_shapes if input_shape is None else input_shape
        input_shapes = input_shape if isinstance(input_shape[0], (list, tuple)) else [input_shape]  # Convert to list of input_shpae
        assert len(input_shapes) == len(self.input_shapes), "input_shape={} length not matching self.input_shapes={}".format(input_shape, self.input_shapes)

        input_datas = []
        for input_shape, model_input_shape, model_input_dtype in zip(input_shapes, self.input_shapes, self.input_dtypes):
            input_shape = list(input_shape).copy()
            if len(input_shape) == len(model_input_shape) - 1:
                input_shape = [batch_size] + input_shape
            assert len(input_shape) == len(model_input_shape), "input_shape={} rank not match with input={}".format(input_shape, model_input_shape)

            if input_shape[0] is None or input_shape[0] == -1:
                input_shape[0] = batch_size
            if None in input_shape or -1 in input_shape:
                print("[WARNING] dynamic shape value in input_shape={}, set to 32".format(input_shape))
                input_shape = [32 if ii is None or ii == -1 else ii for ii in input_shape]

            dtype = model_input_dtype or torch.get_default_dtype()
            dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
            input_datas.append(torch.ones(input_shape, dtype=dtype))
        print(">>>> input_shape: {}, dtype: {}".format([ii.shape for ii in input_datas], [ii.dtype for ii in input_datas]))
        return input_datas

    def summary(self, input_shape=None, **kwargs):
        from torchinfo import summary

        input_datas = self._create_fake_input_data_(input_shape)
        print(summary(self.model, input_data=input_datas if len(self.input_shapes) == 1 else [input_datas], **kwargs))

    def export_onnx(self, filepath=None, input_shape=None, batch_size=1, simplify=False, input_names=None, output_names=None, **kwargs):
        input_datas = self._create_fake_input_data_(input_shape, batch_size=batch_size)

        dynamic_axes = kwargs.pop("dynamic_axes", None)
        input_names = input_names or self.input_names
        output_names = output_names or self.output_names
        if dynamic_axes is None and (batch_size is None or batch_size == -1):
            print("Set dynamic batch size")
            dynamic_axes = {ii: {0: "-1"} for ii in input_names}
            dynamic_axes.update({ii: {0: "-1"} for ii in output_names})

        filepath = (self.name + ".onnx") if filepath is None else (filepath if filepath.endswith(".onnx") else (filepath + ".onnx"))
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

        traced_cell = torch.jit.trace(self.model, example_inputs=input_datas if len(self.input_shapes) == 1 else [input_datas])
        filepath = (self.name + ".pth") if filepath is None else (filepath if filepath.endswith(".pth") else (filepath + ".pth"))
        torch.jit.save(traced_cell, filepath, **kwargs)
        print("Exported pth:", filepath)
