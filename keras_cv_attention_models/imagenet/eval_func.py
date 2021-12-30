from keras_cv_attention_models.imagenet import data
from keras_cv_attention_models.model_surgery import change_model_input_shape
import tensorflow as tf


class TorchModelInterf:
    def __init__(self, model):
        import torch
        import os

        self.torch = torch
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
        device_name = "cuda:0" if len(cvd) > 0 and int(cvd) != -1 else "cpu"
        self.device = self.torch.device(device_name)
        self.model = model.cuda(device_name)

    def __call__(self, imgs):
        # print(imgs.shape, imgs[0])
        output = self.model(self.torch.from_numpy(imgs).permute([0, 3, 1, 2]).to(self.device).float())
        return output.cpu().detach().numpy()


class TFLiteModelInterf:
    def __init__(self, model_path):
        self.interpreter = tf.lite.Interpreter(model_path=model_path)
        input_details = self.interpreter.get_input_details()[0]
        output_details = self.interpreter.get_output_details()[0]
        self.input_dtype = input_details["dtype"]
        self.output_dtype = output_details["dtype"]
        self.input_index = input_details["index"]
        self.output_index = output_details["index"]
        self.input_shape = input_details["shape"].tolist()[1:-1]

        if self.input_dtype == tf.uint8 or self.output_dtype == tf.uint8:
            self.input_scale, self.input_zero_point = input_details.get("quantization", (1.0, 0.0))
            self.output_scale, self.output_zero_point = output_details.get("quantization", (1.0, 0.0))
            self.__interf__ = self.__uint8_interf__
        else:
            self.__interf__ = self.__float_interf__

        self.interpreter.allocate_tensors()

    def __float_interf__(self, img):
        img = tf.cast(img, self.input_dtype)
        self.interpreter.set_tensor(self.input_index, tf.expand_dims(img, 0))
        self.interpreter.invoke()
        return self.interpreter.get_tensor(self.output_index).copy()

    def __uint8_interf__(self, img):
        img = tf.cast(img / self.input_scale + self.input_zero_point, self.input_dtype)
        self.interpreter.set_tensor(self.input_index, tf.expand_dims(img, 0))
        self.interpreter.invoke()
        pred = self.interpreter.get_tensor(self.output_index).copy()
        return (pred.astype("float32") - self.output_zero_point) * self.output_scale

    def __call__(self, imgs):
        # print(imgs.shape, imgs[0])
        preds = []
        for img in imgs:
            pred = self.__interf__(img)
            preds.append(pred)
        return tf.concat(preds, 0).numpy()


def evaluation(
    model, data_name="imagenet2012", input_shape=None, batch_size=64, central_crop=1.0, resize_method="bicubic", antialias=False, rescale_mode="torch"
):
    from tqdm import tqdm
    import numpy as np

    if isinstance(model, tf.keras.models.Model):
        input_shape = model.input_shape[1:-1] if input_shape is None else input_shape[:2]
        model_interf = change_model_input_shape(model, input_shape)
        print(">>>> Using input_shape {} for Keras model.".format(input_shape))
    elif isinstance(model, TFLiteModelInterf) or (isinstance(model, str) and model.endswith(".tflite")):
        model_interf = model if isinstance(model, TFLiteModelInterf) else TFLiteModelInterf(model)
        input_shape = model_interf.input_shape
        print(">>>> Using input_shape {} for TFLite model.".format(input_shape))
    else:
        model_interf = TorchModelInterf(model)
        assert input_shape is not None

    _, test_dataset, _, _, _ = data.init_dataset(
        data_name,
        input_shape=input_shape,
        batch_size=batch_size,
        eval_central_crop=central_crop,
        resize_method=resize_method,
        resize_antialias=antialias,
        rescale_mode=rescale_mode,
    )

    y_true, y_pred_top_1, y_pred_top_5 = [], [], []
    for img_batch, true_labels in tqdm(test_dataset.as_numpy_iterator(), "Evaluating", total=len(test_dataset)):
        predicts = np.array(model_interf(img_batch))
        pred_argsort = predicts.argsort(-1)
        y_pred_top_1.extend(pred_argsort[:, -1])
        y_pred_top_5.extend(pred_argsort[:, -5:])
        y_true.extend(np.array(true_labels).argmax(-1))
    y_true, y_pred_top_1, y_pred_top_5 = np.array(y_true), np.array(y_pred_top_1), np.array(y_pred_top_5)
    accuracy_1 = np.sum(y_true == y_pred_top_1) / y_true.shape[0]
    accuracy_5 = np.sum([ii in jj for ii, jj in zip(y_true, y_pred_top_5)]) / y_true.shape[0]
    print(">>>> Accuracy top1:", accuracy_1, "top5:", accuracy_5)
    return y_true, y_pred_top_1, y_pred_top_5


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
    import json

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


def plot_and_peak_scatter(ax, array, peak_method, label, skip_first=0, color=None, va="bottom", fontsize=12, **kwargs):
    array = array[skip_first:]
    for id, ii in enumerate(array):
        if tf.math.is_nan(ii):
            array[id] = array[id - 1]
    ax.plot(range(skip_first, skip_first + len(array)), array, label=label, color=color, **kwargs)
    color = ax.lines[-1].get_color() if color is None else color
    pp = peak_method(array)
    vv = array[pp]
    ax.scatter(pp + skip_first, vv, color=color, marker="v")
    ax.text(pp + skip_first, vv, "{:.4f}".format(vv), va=va, ha="right", color=color, fontsize=fontsize, rotation=0)


def plot_hists(hists, names=None, base_size=6, addition_plots=["lr"], text_va=["bottom"], skip_first=0):
    import os
    import json
    import matplotlib.pyplot as plt
    import numpy as np

    num_axes = (2 + len(addition_plots)) if addition_plots is not None and len(addition_plots) != 0 else 2
    fig, axes = plt.subplots(1, num_axes, figsize=(num_axes * base_size, base_size))
    hists = [hists] if isinstance(hists, (str, dict)) else hists
    names = names if isinstance(names, (list, tuple)) else [names]
    fontsize = base_size * 2 - 2
    for id, hist in enumerate(hists):
        name = names[min(id, len(names) - 1)] if names != None else None
        cur_va = text_va[id % len(text_va)]
        if isinstance(hist, str):
            name = name if name != None else os.path.splitext(os.path.basename(hist))[0]
            with open(hist, "r") as ff:
                hist = json.load(ff)
        name = name if name != None else str(id)

        plot_and_peak_scatter(axes[0], hist["loss"], np.argmin, name + " loss", skip_first, color=None, va=cur_va, fontsize=fontsize)
        color = axes[0].lines[-1].get_color()
        val_loss = hist.get("val_loss", [])
        if len(val_loss) > 0 and "val_loss" not in addition_plots:
            plot_and_peak_scatter(axes[0], val_loss, np.argmin, name + " val_loss", skip_first, color, cur_va, fontsize, linestyle="--")
        acc = hist.get("acc", hist.get("accuracy", []))
        if len(acc) > 0:  # For timm log
            plot_and_peak_scatter(axes[1], acc, np.argmax, name + " accuracy", skip_first, color=color, va=cur_va, fontsize=fontsize)
        val_acc = hist.get("val_acc", hist.get("val_accuracy", []))
        plot_and_peak_scatter(axes[1], val_acc, np.argmax, name + " val_accuracy", skip_first, color, cur_va, fontsize, linestyle="--")
        if addition_plots is not None and len(addition_plots) != 0:
            for id, ii in enumerate(addition_plots):
                if len(hist.get(ii, [])) > 0:
                    peak_method = np.argmin if "loss" in ii else np.argmax
                    plot_and_peak_scatter(axes[2 + id], hist[ii], peak_method, name + " " + ii, skip_first, color, cur_va, fontsize=fontsize)
    for ax in axes:
        ax.legend()
        ax.grid(True)
    fig.tight_layout()
    return fig
