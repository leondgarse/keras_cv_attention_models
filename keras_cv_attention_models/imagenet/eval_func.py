import os
import json
import types
import numpy as np
from tqdm import tqdm
from keras_cv_attention_models.models import no_grad_if_torch
from keras_cv_attention_models.backend import models, is_tensorflow_backend, is_channels_last, get_file


CLASS_INDEX_IMAGENET = None
CLASS_INDEX_IMAGENET21K = None

""" Imagenet evaluation """


class TorchModelInterf:
    def __init__(self, model):
        import torch

        self.torch = torch
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        device_name = "cuda:0" if len(cvd) > 0 and int(cvd) != -1 else "cpu"
        self.device = self.torch.device(device_name)
        self.model = model.cuda(device_name)

    def __call__(self, imgs):
        # print(imgs.shape, imgs[0])
        imgs = imgs.numpy() if hasattr(imgs, "numpy") else imgs
        output = self.model(self.torch.from_numpy(imgs).permute([0, 3, 1, 2]).to(self.device).float())
        return output.cpu().detach().numpy()


class TFLiteModelInterf:
    """
    >>> import kecam
    >>> mm = kecam.models.AotNet50(input_shape=[None, None, 3])  # Dynamic shape
    >>> open(mm.name + ".tflite", "wb").write(tf.lite.TFLiteConverter.from_keras_model(mm).convert())
    >>> tt = kecam.imagenet.eval_func.TFLiteModelInterf('aotnet50.tflite')
    >>>
    >>> print(f"{tt.input_shape = }")
    >>> # tt.input_shape = (1, 1, 1, 3)
    >>> print(f"{tt([np.ones([224, 224, 3]).astype('float32')]).shape = }")
    >>> # >>>> Calling resize_tensor_input, input_shape (1, 1, 1, 3) -> (1, 224, 224, 3):
    >>> # tt([np.ones([224, 224, 3]).astype('float32')]).shape = (1, 1000)
    >>> print(f"{tt([np.ones([119, 75, 3]).astype('float32')]).shape = }")
    >>> # >>>> Calling resize_tensor_input, input_shape (1, 224, 224, 3) -> (1, 119, 75, 3):
    >>> # tt([np.ones([119, 75, 3]).astype('float32')]).shape = (1, 1000)
    """

    def __init__(self, model_path):
        import tensorflow as tf

        self.tf = tf
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.name = os.path.splitext(os.path.basename(model_path))[0]

        input_details = self.interpreter.get_input_details()[0]
        self.input_dtype = input_details["dtype"]
        self.input_index = input_details["index"]
        self.input_shape = tuple(input_details["shape"].tolist())

        output_details = self.interpreter.get_output_details()[0]
        self.output_dtype = output_details["dtype"]
        self.output_index = output_details["index"]
        self.output_shape = tuple(output_details["shape"].tolist())

        if self.input_dtype == tf.uint8 or self.output_dtype == tf.uint8:
            self.input_scale, self.input_zero_point = input_details.get("quantization", (1.0, 0.0))
            self.output_scale, self.output_zero_point = output_details.get("quantization", (1.0, 0.0))
            self.__interf__ = self.__uint8_interf__
        else:
            self.__interf__ = self.__float_interf__

        self.interpreter.allocate_tensors()

    def __float_interf__(self, img):
        img = self.tf.cast(img, self.input_dtype)
        self.interpreter.set_tensor(self.input_index, self.tf.expand_dims(img, 0))
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_index).copy()

    def __uint8_interf__(self, img):
        img = self.tf.cast(img / self.input_scale + self.input_zero_point, self.input_dtype)
        self.interpreter.set_tensor(self.input_index, self.tf.expand_dims(img, 0))
        self.interpreter.invoke()
        pred = self.interpreter.get_tensor(self.output_index).copy()
        return (pred.astype("float32") - self.output_zero_point) * self.output_scale

    def __call__(self, imgs):
        # print(imgs.shape, imgs[0])
        preds = []
        do_concat = True
        for id, img in enumerate(imgs):
            cur_input_shape = (1, *img.shape)  # For dynamic input shape. Keep batch size 1
            if cur_input_shape != self.input_shape:
                print(">>>> Calling resize_tensor_input, input_shape {} -> {}:".format(self.input_shape, cur_input_shape))
                self.interpreter.resize_tensor_input(self.input_index, cur_input_shape, strict=True)
                self.interpreter.allocate_tensors()
                self.input_shape = cur_input_shape
                do_concat = False if id > 0 else do_concat  # If shape changed after the first inference
            pred = self.__interf__(img)
            preds.append(pred)
        return np.concatenate(preds, 0) if do_concat else preds


class ONNXModelInterf:
    def __init__(self, model_file):
        import onnx
        import onnxruntime as ort

        ort.set_default_logger_severity(3)
        model_content = onnx.load_model(model_file)
        for input in model_content.graph.input:
            input.type.tensor_type.shape.dim[0].dim_param = "None"  # Set batch_size as dynamic
        for output in model_content.graph.output:
            output.type.tensor_type.shape.dim[0].dim_param = "None"  # Set batch_size as dynamic

        self.ort_session = ort.InferenceSession(model_content.SerializeToString())
        inputs, outputs = self.ort_session.get_inputs(), self.ort_session.get_outputs()
        self.input_names, self.output_names = [ii.name for ii in inputs], [ii.name for ii in outputs]
        self.input_name, self.output_name = self.input_names[0], self.output_names[0]

        self.input_shape = inputs[0].shape if len(inputs) == 1 else [ii.shape for ii in inputs]
        self.output_shape = outputs[0].shape if len(outputs) == 1 else [ii.shape for ii in outputs]

        if len(inputs[0].shape) == 4:
            # Regard channel shape as the smallest value, and if it's on dimension 1, regard as channel first NCHW format
            self.data_format = "NCHW" if min(inputs[0].shape[1:]) == inputs[0].shape[1] else "NHWC"
            self.input_channels = inputs[0].shape[1] if self.data_format == "NCHW" else inputs[0].shape[-1]
        else:
            self.data_format = "ND"
            self.input_channels = None

    def __call__(self, imgs, *args, **kwargs):
        imgs = imgs.numpy() if hasattr(imgs, "numpy") else imgs
        if self.data_format == "NCHW" and imgs.shape[1] != self.input_channels:
            imgs = imgs.transpose(0, 3, 1, 2)

        inputs = {name: ii for name, ii in zip(self.input_names, [imgs, *args])}
        inputs.update(kwargs)
        # print({kk: vv.shape for kk, vv in inputs.items()})
        outputs = self.ort_session.run(self.output_names, inputs)
        return outputs if len(self.output_names) > 1 else outputs[0]
        # return np.array([self.ort_session.run(self.output_names, {self.input_name: imgs[None]})[0][0] for img in imgs])


@no_grad_if_torch
def evaluation(
    model, data_name="imagenet2012", input_shape=None, batch_size=64, central_crop=1.0, resize_method="bicubic", antialias=True, rescale_mode="auto"
):
    from keras_cv_attention_models.model_surgery import change_model_input_shape
    from keras_cv_attention_models.imagenet import data  # avoiding circular import

    if isinstance(model, models.Model):
        input_shape = (model.input_shape[1:-1] if is_channels_last() else model.input_shape[2:]) if input_shape is None else input_shape[:2]
        model_interf = change_model_input_shape(model, input_shape) if is_tensorflow_backend else model
        print(">>>> Using input_shape {} for Keras model.".format(input_shape))
    elif isinstance(model, TFLiteModelInterf) or (isinstance(model, str) and model.endswith(".tflite")):
        model_interf = model if isinstance(model, TFLiteModelInterf) else TFLiteModelInterf(model)
        input_shape = model_interf.input_shape[1:-1]
        print(">>>> Using input_shape {} for TFLite model.".format(input_shape))
    elif isinstance(model, types.LambdaType):
        model_interf = model
    elif isinstance(model, str) and model.endswith(".onnx"):
        model_interf = ONNXModelInterf(model)
        input_shape = model_interf.input_shape[2:] if model_interf.data_format == "NCHW" else model_interf.input_shape[1:-1]
        print(">>>> Using input_shape {} for ONNX model, data_format: {}.".format(input_shape, model_interf.data_format))
    else:
        model_interf = TorchModelInterf(model)
        assert input_shape is not None

    if isinstance(rescale_mode, str) and rescale_mode.lower() == "auto":
        rescale_mode = getattr(model, "rescale_mode", "torch")
        print(">>>> rescale_mode:", rescale_mode)

    test_dataset = data.init_dataset(
        data_name,
        input_shape=input_shape,
        batch_size=batch_size,
        eval_central_crop=central_crop,
        resize_method=resize_method,
        resize_antialias=antialias,
        rescale_mode=rescale_mode,
    )[1]

    y_true, y_pred_top_1, y_pred_top_5 = [], [], []
    for img_batch, true_labels in tqdm(test_dataset, "Evaluating", total=len(test_dataset)):
        predicts = model_interf(img_batch)
        predicts = np.array(predicts.cpu().detach() if hasattr(predicts, "detach") else predicts)
        pred_argsort = predicts.argsort(-1)
        y_pred_top_1.extend(pred_argsort[:, -1])
        y_pred_top_5.extend(pred_argsort[:, -5:])
        y_true.extend(np.array(true_labels).argmax(-1))
    y_true, y_pred_top_1, y_pred_top_5 = np.array(y_true), np.array(y_pred_top_1), np.array(y_pred_top_5)
    accuracy_1 = np.sum(y_true == y_pred_top_1) / y_true.shape[0]
    accuracy_5 = np.sum([ii in jj for ii, jj in zip(y_true, y_pred_top_5)]) / y_true.shape[0]
    print(">>>> Accuracy top1:", accuracy_1, "top5:", accuracy_5)
    return y_true, y_pred_top_1, y_pred_top_5


""" Decode predictions """


def decode_predictions(preds, top=5):
    """Similar function from keras.applications.imagenet_utils.decode_predictions, just also supporting imagenet21k class index"""
    preds = np.array(preds.detach() if hasattr(preds, "detach") else preds)
    if len(preds.shape) != 2 or (preds.shape[-1] not in [1000, 21843, 21841]):
        print("[Error] not imagenet or imagenet21k prediction, not supported")
        return

    is_imagenet = preds.shape[-1] == 1000

    global CLASS_INDEX_IMAGENET
    global CLASS_INDEX_IMAGENET21K
    if (is_imagenet and CLASS_INDEX_IMAGENET is None) or (not is_imagenet and CLASS_INDEX_IMAGENET21K is None):
        if is_imagenet:
            url = "https://github.com/leondgarse/keras_cv_attention_models/releases/download/assets/imagenet_class_index.json"
            file_hash = "c2c37ea517e94d9795004a39431a14cb"
        else:
            url = "https://github.com/leondgarse/keras_cv_attention_models/releases/download/assets/imagenet21k_class_index.json"
            file_hash = "a07173727548feaea3cc855ed6341a4f"

        class_index_path = os.path.join(os.path.expanduser("~/.keras/datasets"), os.path.basename(url))
        print(">>>> Trying to load index file:", class_index_path)
        class_index_path = get_file(origin=url, file_hash=file_hash)
        with open(class_index_path) as ff:
            if is_imagenet:
                CLASS_INDEX_IMAGENET = json.load(ff)
            else:
                CLASS_INDEX_IMAGENET21K = json.load(ff)
    class_index = CLASS_INDEX_IMAGENET if is_imagenet else CLASS_INDEX_IMAGENET21K

    results = []
    for pred in preds:
        top_indices = pred.argsort()[-top:][::-1]
        result = [tuple(class_index[str(i)]) + (pred[i],) for i in top_indices]
        result.sort(key=lambda x: x[2], reverse=True)
        results.append(result)
    return results


def decode_predictions_imagenet21k(preds, top=5):
    return decode_predictions(preds, top=top)  # just keeping for compiling with ealier


""" Plotting related """


def parse_timm_log(log_file, pick_keys=None):
    with open(log_file, "r") as ff:
        aa = ff.readlines()

    """ Find pattern for train epoch end """
    train_epoch_started, train_epoch_end_pattern, previous_line = False, "", ""
    for ii in aa:
        if ii.startswith("Train:"):
            train_epoch_started = True
            previous_line = ii
        elif train_epoch_started and ii.startswith("Test:"):
            train_epoch_end_pattern = previous_line.split("[")[1].split("]")[0].strip()
            break

    """ Find pattern for test end """
    test_epoch_started, test_epoch_end_pattern, previous_line = False, "", ""
    for ii in aa:
        if ii.startswith("Test:"):
            test_epoch_started = True
            previous_line = ii
        elif test_epoch_started and not ii.startswith("Train:"):
            test_epoch_end_pattern = previous_line.split("[")[1].split("]")[0].strip()
            break
    print("train_epoch_end_pattern = {}, test_epoch_end_pattern = {}".format(train_epoch_end_pattern, test_epoch_end_pattern))

    split_func = lambda xx, ss, ee: float(xx.split(ss)[1].strip().split(ee)[0].split("(")[-1].split(")")[0])
    train_loss = [split_func(ii, "Loss:", "Time:") for ii in aa if train_epoch_end_pattern in ii]
    lr = [split_func(ii, "LR:", "Data:") for ii in aa if train_epoch_end_pattern in ii]
    val_loss = [split_func(ii, "Loss:", "Acc@1:") for ii in aa if test_epoch_end_pattern in ii]
    val_acc = [split_func(ii, "Acc@1:", "Acc@5:") for ii in aa if test_epoch_end_pattern in ii]
    if val_acc[-1] > 1:
        val_acc = [ii / 100.0 for ii in val_acc]

    # print(f"{len(train_loss) = }, {len(lr) = }, {len(val_loss) = }, {len(val_acc) = }")
    hh = {"loss": train_loss, "lr": lr, "val_loss": val_loss, "val_acc": val_acc}
    return hh if pick_keys is None else {kk: hh[kk] for kk in pick_keys}


def combine_hist_into_one(hist_list, save_file=None):
    hh = {}
    for hist in hist_list:
        with open(hist, "r") as ff:
            aa = json.load(ff)
        for kk, vv in aa.items():
            hh.setdefault(kk, []).extend(vv)

    if save_file:
        with open(save_file, "w") as ff:
            json.dump(hh, ff)
    return hh
