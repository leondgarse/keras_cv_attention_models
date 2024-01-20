import torch
import numpy as np
from tqdm.auto import tqdm
from torch import nn
from contextlib import nullcontext
from keras_cv_attention_models import backend
from keras_cv_attention_models.pytorch_backend import layers, callbacks, metrics


class _Trainer_(object):
    def init_metrics(self, cur_metrics=None):
        if cur_metrics is None:
            metrics_names, cur_metrics = [], []
        elif isinstance(cur_metrics, str):
            metrics_names, cur_metrics = [cur_metrics], [cur_metrics]
        elif isinstance(cur_metrics, (list, tuple)):
            metrics_names, cur_metrics = [ii if isinstance(ii, str) else ii.name for ii in cur_metrics], list(cur_metrics)
        elif isinstance(cur_metrics, dict):
            assert self.output_names is not None, "self.output_names cannot be None when using dict metrics, provide output_names or use list metrics"
            assert len(self.output_names) == len(cur_metrics), "self.output_names={} not matching with metrics={}".format(self.output_names, metrics)
            metrics_names, cur_metrics = self.output_names, [cur_metrics[ii] for ii in self.output_names]
        else:
            metrics_names, cur_metrics = [cur_metrics.name], [cur_metrics]
        cur_metrics = [metrics.BUILDIN_METRICS[ii]() if isinstance(ii, str) else ii for ii in cur_metrics]
        return metrics_names, cur_metrics

    @property
    def compute_dtype(self):
        try:
            return next(self.parameters()).dtype
        except StopIteration:
            return torch.get_default_dtype()

    def train_compile(self, optimizer="RMSprop", loss=None, metrics=None, loss_weights=None, grad_accumulate=1, grad_max_norm=-1, **kwargs):
        # works like kers `model.compile`, but `compile` is took by `nn.Module`, rename as `train_compile`
        self.optimizer = getattr(torch.optim, optimizer)(self.parameters()) if isinstance(optimizer, str) else optimizer
        self.compiled_loss = self.loss = (lambda y_true, y_pred: torch.functional.F.cross_entropy(y_pred, y_true)) if loss is None else loss
        self.loss_weights, self.grad_accumulate, self.grad_max_norm = loss_weights or 1.0, grad_accumulate, grad_max_norm
        self.metrics_names, self.metrics = self.init_metrics(metrics)
        self.eval_metrics = [metric.name for metric in self.metrics if metric.eval_only]  # Mark metric like acc5 for eval only

        device = next(self.parameters()).device
        device_type = device.type
        if device_type == "cpu":
            scaler = torch.cuda.amp.GradScaler(enabled=False)
            global_context = nullcontext()
        else:
            scaler = torch.cuda.amp.GradScaler(enabled=True)
            global_context = torch.autocast(device_type=device_type, dtype=torch.float16)
        self.device, self.device_type, self.scaler, self.global_context = device, device_type, scaler, global_context

    def _convert_data_(self, data):
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
        if self.device_type == "cuda":
            # pin arrays x,y, which allows us to move them to GPU asynchronously (non_blocking=True)
            data = data.pin_memory().to(self.device, non_blocking=True)
        return data

    def train_step(self, xx, yy):
        # Split out for being able to overwrite
        with self.global_context:
            out = self(xx)
            loss = self.loss(yy, out) * self.loss_weights
        return out, loss

    def fit(self, x=None, y=None, batch_size=32, epochs=1, callbacks=None, validation_data=None, initial_epoch=0, validation_batch_size=None, **kwargs):
        callbacks = callbacks or []
        [ii.set_model(self) for ii in callbacks if ii.model is None]
        self.history = {"loss": []}
        validation_batch_size = validation_batch_size or batch_size
        self.batch_size, self.callbacks, self.validation_batch_size = batch_size, callbacks, validation_batch_size

        bar_format = "{n_fmt}/{total_fmt} [{bar:30}] - ETA: {elapsed}<{remaining} {rate_fmt}{postfix}{desc}"
        for epoch in range(initial_epoch, epochs):
            print("Epoch {}/{}".format(epoch + 1, epochs))
            epoch_logs = {}  # Can be used as global value between different callbacks
            [ii.on_epoch_begin(epoch, epoch_logs) for ii in callbacks]
            [ii.reset_state() for ii in self.metrics]

            self.train()
            self.optimizer.zero_grad()

            avg_loss, accumulate_passed_batches = 0.0, 0
            train_dataset, total = self._dataset_gen_(x, y, batch_size=batch_size)
            process_bar = tqdm(enumerate(train_dataset), total=total, bar_format=bar_format, ascii=".>>=")
            for batch, (xx, yy) in process_bar:
                batch_logs = {}  # Can be used as global value between different callbacks
                [ii.on_train_batch_begin(batch, batch_logs) for ii in callbacks]

                xx = [self._convert_data_(ii) for ii in xx] if isinstance(xx, (list, tuple)) else self._convert_data_(xx)
                yy = [self._convert_data_(ii) for ii in yy] if isinstance(yy, (list, tuple)) else self._convert_data_(yy)
                out, loss = self.train_step(xx, yy)
                self.scaler.scale(loss).backward()

                accumulate_passed_batches += 1
                if accumulate_passed_batches >= self.grad_accumulate:
                    self.scaler.unscale_(self.optimizer)
                    if self.grad_max_norm > 0:
                        torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=self.grad_max_norm)  # clip gradients
                    self.scaler.step(self.optimizer)  # self.optimizer.step()
                    self.scaler.update()
                    self.optimizer.zero_grad()
                    accumulate_passed_batches = 0
                # print(">>>> Epoch {}, batch: {}, loss: {:.4f}".format(epoch, batch, loss.item()))
                avg_loss += loss
                process_bar.desc = " - loss: {:.4f}".format(avg_loss / (batch + 1))  # process_bar.set_description automatically add a : on the tail

                # out = out.detach().cpu()
                if isinstance(yy, (list, tuple)):
                    [metric.update_state(cur_yy, cur_out) for cur_out, cur_yy, metric in zip(out, yy, self.metrics) if metric.name not in self.eval_metrics]
                else:
                    [metric.update_state(yy, out) for metric in self.metrics if metric.name not in self.eval_metrics]
                metrics_results = {ii: metric.result().item() for ii, metric in zip(self.metrics_names, self.metrics) if metric.name not in self.eval_metrics}
                process_bar.desc += "".join([" - {}: {:.4f}".format(name, metric) for name, metric in metrics_results.items()])

                batch_logs["loss"] = loss.item()
                batch_logs.update(metrics_results)
                [ii.on_train_batch_end(batch, logs=batch_logs) for ii in callbacks]

                """ Eval process, put inside for loop for a better display of process bar """
                if batch == total - 1 and validation_data is not None:
                    val_loss, val_metrics_results = self.evaluate(validation_data, batch_size=validation_batch_size, callbacks=callbacks)

                    self.history.setdefault("val_loss", []).append(val_loss.item())
                    for name, val_metric_result in val_metrics_results.items():
                        self.history.setdefault(name, []).append(val_metric_result.item() if hasattr(val_metric_result, "item") else val_metric_result)

                    process_bar.desc += " - val_loss: {:.4f}".format(val_loss)
                    process_bar.desc += "".join([" - {}: {:.4f}".format(name, metric) for name, metric in val_metrics_results.items()])
                    # process_bar.display()
                process_bar.refresh()

            loss = avg_loss / (batch + 1)
            self.history["loss"].append(loss.item() if hasattr(loss, "item") else loss)
            for name, metric_result in metrics_results.items():
                self.history.setdefault(name, []).append(metric_result)
            epoch_logs = {kk: vv[-1] for kk, vv in self.history.items()}
            with self.global_context, torch.no_grad():
                [ii.on_epoch_end(epoch, epoch_logs) for ii in callbacks]
            print()

    def evaluate(self, x=None, y=None, batch_size=None, verbose="auto", callbacks=None, **kwargs):
        callbacks = callbacks or []
        epoch_logs = {}  # Can be used as global value between different callbacks
        [ii.set_model(self) for ii in callbacks if ii.model is None]
        [ii.on_test_begin(epoch_logs) for ii in callbacks]
        [ii.reset_state() for ii in self.metrics]

        self.eval()
        avg_loss, accumulate_passed_batches = 0.0, 0
        val_dataset, total = self._dataset_gen_(x, y, batch_size=batch_size)
        for batch, (xx, yy) in enumerate(val_dataset):
            batch_logs = {}  # Can be used as global value between different callbacks
            [ii.on_test_batch_begin(batch, batch_logs) for ii in callbacks]

            xx = [self._convert_data_(ii) for ii in xx] if isinstance(xx, (list, tuple)) else self._convert_data_(xx)
            yy = [self._convert_data_(ii) for ii in yy] if isinstance(yy, (list, tuple)) else self._convert_data_(yy)
            with torch.no_grad():
                out, loss = self.train_step(xx, yy)
            avg_loss += loss

            # out = out.detach().cpu()
            if isinstance(yy, (list, tuple)):
                [metric.update_state(cur_yy, cur_out) for cur_out, cur_yy, metric in zip(out, yy, self.metrics)]
            else:
                [ii.update_state(yy, out) for ii in self.metrics]
            batch_logs["val_loss"] = loss.item()
            batch_logs.update({name: metric.result().item() for name, metric in zip(self.metrics_names, self.metrics)})
            [ii.on_test_batch_end(batch, batch_logs) for ii in callbacks]

        val_loss = avg_loss / (batch + 1)
        metrics_results = {}
        for name, metric in zip(self.metrics_names, self.metrics):
            metrics_results["val_" + name] = metric.result().item()
        epoch_logs["val_loss"] = val_loss.item()
        epoch_logs.update(metrics_results)
        [ii.on_test_end(epoch_logs) for ii in callbacks]
        return val_loss, metrics_results

    def _dataset_gen_(self, x=None, y=None, batch_size=32):
        if isinstance(x, (list, tuple)) and len(x) == 2 and y is None:
            xx, yy = x[0], x[1]
        else:
            xx, yy = x, y

        if hasattr(xx, "element_spec"):  # TF datsets
            data_shape = xx.element_spec[0].shape
            if self.input_shape is not None and data_shape[-1] == self.input_shape[1]:
                perm = [0, len(data_shape) - 1] + list(range(1, len(data_shape) - 1))  # [0, 3, 1, 2]
                dataset = ((ii.transpose(perm), jj) for ii, jj in xx.as_numpy_iterator())
            else:
                dataset = xx.as_numpy_iterator()
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

            dataset = ((_convert_tensor(xx, id), _convert_tensor(yy, id)) for id in range(num_batches))
            total = num_batches
        else:  # generator or torch.utils.data.DataLoader
            dataset = xx
            total = len(xx)
        return dataset, total


class _Exporter_(object):
    def _create_fake_input_data_(self, input_shape=None, batch_size=1):
        input_shapes = [ii.shape for ii in self.inputs]
        input_dtypes = [ii.dtype for ii in self.inputs]

        input_shape = self.input_shape if input_shape is None else input_shape
        input_shapes = input_shape if isinstance(input_shape[0], (list, tuple)) else [input_shape]  # Convert to list of input_shpae
        assert len(input_shapes) == len(self.inputs), "input_shape={} length not matching self.inputs={}".format(input_shape, self.inputs)

        input_datas = []
        for input_shape, model_input in zip(input_shapes, self.inputs):
            input_shape = list(input_shape).copy()
            if len(input_shape) == len(model_input.shape) - 1:
                input_shape = [batch_size] + input_shape
            assert len(input_shape) == len(model_input.shape), "input_shape={} rank not match with input={}".format(input_shape, model_input.shape)

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
        print(summary(self, input_data=input_datas if len(self.inputs) == 1 else [input_datas], **kwargs))

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
        torch.onnx.export(self, input_datas, filepath, input_names=input_names, output_names=output_names, dynamic_axes=dynamic_axes, **kwargs)
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

        traced_cell = torch.jit.trace(self, example_inputs=input_datas if len(self.inputs) == 1 else [input_datas])
        filepath = (self.name + ".pth") if filepath is None else (filepath if filepath.endswith(".pth") else (filepath + ".pth"))
        torch.jit.save(traced_cell, filepath, **kwargs)
        print("Exported pth:", filepath)

    def load_weights(self, filepath, by_name=True, skip_mismatch=False, **kwargs):
        if filepath.endswith("h5"):
            from keras_cv_attention_models.download_and_load import load_weights_from_hdf5_file

            load_weights_from_hdf5_file(filepath, self, skip_mismatch=skip_mismatch, debug=self.debug)
        else:
            weights = torch.load(filepath, map_location=torch.device("cpu"), **kwargs)
            weights = weights.state_dict() if hasattr(weights, "state_dict") else weights
            self.load_state_dict(weights.get("state_dict", weights.get("model", weights)))
            if hasattr(self, "optimizer") and "optimizer" in weights:
                print(">>>> Reload optimizer state_dict")
                self.optimizer.load_state_dict(weights["optimizer"])

    def save_weights(self, filepath=None, **kwargs):
        filepath = filepath if filepath else self.name + ".h5"
        if filepath.endswith("h5"):
            from keras_cv_attention_models.download_and_load import save_weights_to_hdf5_file

            save_weights_to_hdf5_file(filepath, self, **kwargs)
        else:
            save_items = {"state_dict": self.state_dict()}
            if hasattr(self, "optimizer"):
                save_items.update({"optimizer": self.optimizer.state_dict()})
            torch.save(save_items, filepath, **kwargs)

    def load(self, filepath, **kwargs):
        self.load_weights(filepath, **kwargs)

    def save(self, filepath=None, **kwargs):
        self.save_weights(filepath, **kwargs)

    def count_params(self):
        total_params = sum([np.prod(ii.shape) for ii in self.state_dict().values() if len(ii.shape) != 0])
        trainable_params = sum([np.prod(list(ii.shape)) for ii in self.parameters()])
        non_trainable_params = total_params - trainable_params
        print("Total params: {:,} | Trainable params: {:,} | Non-trainable params:{:,}".format(total_params, trainable_params, non_trainable_params))
        return total_params


class Model(nn.Module, _Trainer_, _Exporter_):
    """
    Examples:
    # compile and fit on buildin models
    >>> os.environ["KECAM_BACKEND"] = "torch"
    >>> import torch
    >>> from keras_cv_attention_models import aotnet
    >>> mm = aotnet.AotNet50(num_classes=10, input_shape=(32, 32, 3))
    >>> mm.compile(metrics='acc')  # Using default cross_entropy loss
    >>> xx, yy = torch.rand([300, 3, 32, 32]), torch.functional.F.one_hot(torch.randint(0, 10, size=[300]), 10).float()
    >>> mm.fit(xx[:256], yy[:256], epochs=2, validation_data=(xx[256:], yy[256:]))

    # Build custom model
    >>> os.environ["KECAM_BACKEND"] = "torch"
    >>> import torch
    >>> from keras_cv_attention_models.backend import layers, models
    >>> from keras_cv_attention_models.imagenet.callbacks import MyCheckpoint  # Add a callback
    >>> inputs = layers.Input([3, 32, 32])
    >>> nn = layers.Conv2D(32, 3, 2, padding='same')(inputs)
    >>> nn = layers.GlobalAveragePooling2D()(nn)
    >>> nn = layers.Dense(10)(nn)
    >>> mm = models.Model(inputs, nn)
    >>> mm.summary()
    >>> xx, yy = torch.rand([1128, 3, 32, 32]), torch.functional.F.one_hot(torch.randint(0, 10, size=[1128]), 10).float()
    >>> loss = lambda y_true, y_pred: (y_true - y_pred.float()).abs().mean()
    >>> mm.compile(optimizer="AdamW", loss=loss, metrics='acc')
    >>> callbacks = [MyCheckpoint(basic_save_name="test")]
    >>> mm.fit(xx[:1000], yy[:1000], epochs=2, callbacks=callbacks, validation_data=(xx[1000:], yy[1000:]))

    # Prediction using buildin model
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

    def __init__(self, inputs=[], outputs=[], name=None, **kwargs):
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
        self.input_dtype_dict = {ii.name: ii.dtype for ii in self.inputs}

        self.num_outputs = len(self.outputs)
        self.create_forward_pipeline()
        self.eval()  # Set eval mode by default
        self.debug = False

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

    def input_to_tensor(self, inputs, dtype=torch.float32):
        param = next(self.parameters())
        device = param.device
        dtype = param.dtype if dtype in (torch.float16, torch.float32) else dtype

        if not isinstance(inputs, torch.Tensor):
            inputs = torch.as_tensor(inputs, device=device)
        if inputs.dtype != dtype:
            inputs = inputs.to(dtype)
        if inputs.device != device:
            inputs = inputs.to(device)
        return inputs

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
        intra_nodes = {kk: [self.input_to_tensor(ii, dtype=self.input_dtype_dict[kk]) for ii in vv] for kk, vv in intra_nodes.items()}

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

    def set_debug(self, debug=True):
        self.debug = debug
        print(">>>> debug: {}".format(self.debug))


class Sequential(Model):
    """
    >>> os.environ["KECAM_BACKEND"] = "torch"
    >>> import torch
    >>> from keras_cv_attention_models.pytorch_backend import layers, models, functional
    >>> mm = models.Sequential([
    >>>     layers.Input([3, 32, 32]),
    >>>     layers.Conv2D(32, 3, 2, padding='same'),
    >>>     layers.GlobalAveragePooling2D(),
    >>>     layers.Dense(10),
    >>>     functional.softmax,  # Can also be an functional callable
    >>> ])
    >>> mm.summary()
    >>> print(mm(torch.ones([1, 3, 32, 32])).shape)
    >>> # torch.Size([1, 10])
    """

    def __init__(self, sequence_layers=None, name=None, **kwargs):
        self.sequence_layers, self.name, self.kwargs = sequence_layers, name, kwargs
        if isinstance(sequence_layers[0], layers.Input):
            self.build(sequence_layers[0].shape)
        else:
            self.input_shape, self.built = None, False

    def build(self, input_shape):
        inputs = layers.Input(input_shape[1:])
        next_node = inputs
        for layer in self.sequence_layers:
            if isinstance(layer, layers.Input):
                continue
            next_node = layer(next_node)
        super().__init__(inputs, next_node, name=self.name)
        self.built = True

    def add(self, layer):
        self.sequence_layers.append(layer)
        if self.input_shape is not None:
            self.build(self.input_shape)
