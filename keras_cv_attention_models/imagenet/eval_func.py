import os
import json
import numpy as np


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
    def __init__(self, model_path):
        import tensorflow as tf

        self.tf = tf
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        self.name = os.path.splitext(os.path.basename(model_path))[0]

        input_details = self.interpreter.get_input_details()[0]
        self.input_dtype = input_details["dtype"]
        self.input_index = input_details["index"]
        self.input_shape = input_details["shape"].tolist()

        output_details = self.interpreter.get_output_details()[0]
        self.output_dtype = output_details["dtype"]
        self.output_index = output_details["index"]
        self.output_shape = output_details["shape"].tolist()

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
        for img in imgs:
            pred = self.__interf__(img)
            preds.append(pred)
        return self.tf.concat(preds, 0).numpy()


class ONNXModelInterf:
    def __init__(self, model_file):
        import onnx
        import onnxruntime as ort

        ort.set_default_logger_severity(3)
        model_content = onnx.load_model(model_file)
        model_content.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "None"  # Set batch_size as dynamic
        model_content.graph.output[0].type.tensor_type.shape.dim[0].dim_param = "None"  # Set batch_size as dynamic

        self.ort_session = ort.InferenceSession(model_content.SerializeToString())
        self.input_name = self.ort_session.get_inputs()[0].name
        self.output_names = [self.ort_session.get_outputs()[0].name]
        self.input_shape = self.ort_session.get_inputs()[0].shape
        self.output_shape = self.ort_session.get_outputs()[0].shape

        # Regard channel shape as the smallest value, and if it's on dimension 1, regard as channel first NCHW format
        self.data_format = "NCHW" if min(self.input_shape[1:]) == self.input_shape[1] else "NHWC"

    def __call__(self, imgs):
        imgs = imgs.numpy() if hasattr(imgs, "numpy") else imgs
        if self.data_format == "NCHW" and imgs.shape[1] != self.input_shape[1]:
            imgs = imgs.transpose(0, 3, 1, 2)
        return self.ort_session.run(self.output_names, {self.input_name: imgs.astype("float32")})[0]
        # return np.array([self.ort_session.run(self.output_names, {self.input_name: imgs[None]})[0][0] for img in imgs])


def evaluation(
    model, data_name="imagenet2012", input_shape=None, batch_size=64, central_crop=1.0, resize_method="bicubic", antialias=True, rescale_mode="auto"
):
    import types
    from tqdm import tqdm
    from keras_cv_attention_models.backend import models
    from keras_cv_attention_models.model_surgery import change_model_input_shape
    from keras_cv_attention_models.imagenet import data

    if isinstance(model, models.Model):
        input_shape = model.input_shape[1:-1] if input_shape is None else input_shape[:2]
        model_interf = change_model_input_shape(model, input_shape)
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
    for img_batch, true_labels in tqdm(test_dataset.as_numpy_iterator(), "Evaluating", total=len(test_dataset)):
        predicts = model_interf(img_batch)
        predicts = np.array(predicts.detach() if hasattr(predicts, "detach") else predicts)
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
    from keras_cv_attention_models.backend import get_file

    preds = np.array(preds.detach() if hasattr(preds, "detach") else preds)
    if len(preds.shape) != 2 or (preds.shape[-1] != 1000 and preds.shape[-1] != 21843):
        print("[Error] not imagenet or imagenet21k prediction, not supported")
        return

    is_imagenet = preds.shape[-1] == 1000

    global CLASS_INDEX_IMAGENET
    global CLASS_INDEX_IMAGENET21K
    if (is_imagenet and CLASS_INDEX_IMAGENET is None) or CLASS_INDEX_IMAGENET21K is None:
        if is_imagenet:
            url = "https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/imagenet_class_index.json"
        else:
            url = "https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/imagenet21k_class_index.json"

        class_index_path = get_file(origin=url)
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


def curve_fit(source, target_len=10, skip=5, use_recent=40):
    from scipy.optimize import curve_fit

    def func_curv(x, a, b, c, d):
        pp = np.log(x)
        # pp = 1 / x
        return a * pp**3 + b * pp**2 + c * pp + d

    recent_source = source[skip:]
    use_recent = len(source) if use_recent == -1 else use_recent
    if len(recent_source) > use_recent:
        recent_source = recent_source[-use_recent:]
    start_pos = len(source) - len(recent_source)
    popt, pcov = curve_fit(func_curv, np.arange(start_pos, len(source)), recent_source)
    return list(source[: -len(recent_source)]) + func_curv(np.arange(start_pos, len(source) + target_len), *popt).tolist()


def plot_and_peak_scatter(ax, source_array, peak_method, label, skip_first=0, color=None, va="bottom", pred_curve=0, **kwargs):
    array = source_array[skip_first:]
    for id, ii in enumerate(array):
        if np.isnan(ii):
            array[id] = array[id - 1]
    ax.plot(range(skip_first, skip_first + len(array)), array, label=label, color=color, **kwargs)
    color = ax.lines[-1].get_color() if color is None else color
    pp = peak_method(array)
    vv = array[pp]
    ax.scatter(pp + skip_first, vv, color=color, marker="v")
    # ax.text(pp + skip_first, vv, "{:.4f}".format(vv), va=va, ha="right", color=color, fontsize=fontsize, rotation=0)
    ax.text(pp + skip_first, vv, "{:.4f}".format(vv), va=va, ha="right", color=color, rotation=0)

    if pred_curve > 0:
        kwargs.pop("linestyle", None)
        pred_array = curve_fit(source_array, pred_curve)[skip_first:]
        ax.plot(range(skip_first, skip_first + len(pred_array)), pred_array, color=color, linestyle=":", **kwargs)


def plot_hists(hists, names=None, base_size=6, addition_plots=["lr"], text_va=["bottom"], skip_first=0, pred_curve=0):
    import matplotlib as mpl
    import matplotlib.pyplot as plt

    mpl.rcParams["lines.linewidth"] = base_size / 8 * 1.5  # 8 -> 1.5
    mpl.rcParams["font.size"] = base_size * 2 - 4  # 8 -> 12, 6 -> 8
    mpl.rcParams["legend.fontsize"] = base_size * 2 - 4
    mpl.rcParams["xtick.labelsize"] = base_size * 2 - 4
    mpl.rcParams["ytick.labelsize"] = base_size * 2 - 4
    mpl.rcParams["xtick.major.pad"] = base_size / 2 - 2  # 8 -> 4, 6 -> 3
    mpl.rcParams["ytick.major.pad"] = base_size / 2 - 2

    num_axes = (2 + len(addition_plots)) if addition_plots is not None and len(addition_plots) != 0 else 2
    fig, axes = plt.subplots(1, num_axes, figsize=(num_axes * base_size, base_size))
    hists = [hists] if isinstance(hists, (str, dict)) else hists
    names = names if isinstance(names, (list, tuple)) else [names]
    for id, hist in enumerate(hists):
        name = names[min(id, len(names) - 1)] if names != None else None
        cur_va = text_va[id % len(text_va)]
        if isinstance(hist, str):
            name = name if name != None else os.path.splitext(os.path.basename(hist))[0]
            with open(hist, "r") as ff:
                hist = json.load(ff)
        name = name if name != None else str(id)

        acc_key = "acc"
        if acc_key not in hist:
            all_acc_key = [ii for ii in hist.keys() if "acc" in ii and "val" not in ii]
            acc_key = "acc" if len(all_acc_key) == 0 else all_acc_key[0]

        val_acc_key = "val_acc"
        if "val_ap_ar" in hist:  # It's coco hist
            hist["val_ap_0.50:0.95"] = [ii[0] for ii in hist["val_ap_ar"]]
            val_acc_key = "val_ap 0.50:0.95"
        elif val_acc_key not in hist:
            all_val_acc_key = [ii for ii in hist.keys() if "acc" in ii and "val" in ii]
            val_acc_key = "val_acc" if len(all_val_acc_key) == 0 else all_val_acc_key[0]

        cur_pred_curve = pred_curve[min(id, len(pred_curve) - 1)] if isinstance(pred_curve, (list, tuple)) else pred_curve
        plot_and_peak_scatter(axes[0], hist["loss"], np.argmin, name + " loss", skip_first, None, cur_va, pred_curve=cur_pred_curve)
        color = axes[0].lines[-1].get_color()
        val_loss = hist.get("val_loss", [])
        if len(val_loss) > 0 and "val_loss" not in addition_plots:
            plot_and_peak_scatter(axes[0], val_loss, np.argmin, name + " val_loss", skip_first, color, cur_va, cur_pred_curve, linestyle="--")

        acc = hist.get(acc_key, [])
        if len(acc) > 0:  # For timm log
            plot_and_peak_scatter(axes[1], acc, np.argmax, name + " " + acc_key, skip_first, color, cur_va, cur_pred_curve)

        val_acc = hist.get(val_acc_key, [])
        if len(val_acc) > 0:  # For timm log
            plot_and_peak_scatter(axes[1], val_acc, np.argmax, name + " " + val_acc_key, skip_first, color, cur_va, cur_pred_curve, linestyle="--")

        if addition_plots is not None and len(addition_plots) != 0:
            for id, ii in enumerate(addition_plots):
                if len(hist.get(ii, [])) > 0:
                    peak_method = np.argmin if "loss" in ii else np.argmax
                    plot_and_peak_scatter(axes[2 + id], hist[ii], peak_method, name + " " + ii, skip_first, color, cur_va, cur_pred_curve)
    for ax in axes:
        ax.legend()
        ax.grid(True)
    fig.tight_layout()
    fig.tight_layout()  # again
    plt.show()
    return fig
