from keras_cv_attention_models.imagenet import data
from keras_cv_attention_models.model_surgery import change_model_input_shape
import tensorflow as tf


class Torch_model_interf:
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


def evaluation(model, data_name="imagenet2012", input_shape=None, batch_size=64, central_crop=1.0, resize_method="bicubic", rescale_mode="torch"):
    from tqdm import tqdm
    import numpy as np

    input_shape = model.input_shape[1:-1] if input_shape is None else input_shape[:2]
    _, test_dataset, _, _, _ = data.init_dataset(
        data_name, input_shape=input_shape, batch_size=batch_size, eval_central_crop=central_crop, resize_method=resize_method, rescale_mode=rescale_mode
    )

    model_interf = change_model_input_shape(model, input_shape) if isinstance(model, tf.keras.models.Model) else Torch_model_interf(model)

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


def parse_timm_log(log_filee):
    with open(log_filee, "r") as ff:
        aa = ff.readlines()

    """ Find pattern for train epoch end """
    train_epoch_started, train_epoch_end_pattern, previous_line = False, "", ""
    for ii in aa:
        if ii.startswith("Train:"):
            train_epoch_started = True
            previous_line = ii
        elif train_epoch_started and not ii.startswith("Train:"):
            train_epoch_end_pattern = previous_line.split("[")[1].split("]")[0].strip()
            break

    """ Find pattern for test end """
    test_epoch_started, test_epoch_end_pattern, previous_line = False, "", ""
    for ii in aa:
        if ii.startswith("Test:"):
            test_epoch_started = True
            previous_line = ii
        elif test_epoch_started and not ii.startswith("Test:"):
            test_epoch_end_pattern = previous_line.split("[")[1].split("]")[0].strip()
            break
    print(f"{train_epoch_end_pattern = }, {test_epoch_end_pattern = }")

    split_func = lambda xx, ss, ee: float(xx.split(ss)[1].strip().split(ee)[0].split("(")[-1].split(")")[0])
    train_loss = [split_func(ii, "Loss:", "Time:") for ii in aa if train_epoch_end_pattern in ii]
    lr = [split_func(ii, "LR:", "Data:") for ii in aa if train_epoch_end_pattern in ii]
    val_loss = [split_func(ii, "Loss:", "Acc@1:") for ii in aa if test_epoch_end_pattern in ii]
    val_acc = [split_func(ii, "Acc@1:", "Acc@5:") for ii in aa if test_epoch_end_pattern in ii]
    if val_acc[-1] > 1:
        val_acc = [ii / 100.0 for ii in val_acc]

    # train_loss = [float(ii.split('Loss:')[1].strip().split(" ")[1][1:-1]) for ii in aa if train_epoch_end_pattern in ii]
    # lr = [float(ii.split('LR:')[1].strip().split(" ")[0]) for ii in aa if train_epoch_end_pattern in ii]
    # val_loss = [float(ii.split('Loss:')[1].strip().split(" ")[1][1:-1]) for ii in aa if test_epoch_end_pattern in ii]
    # val_acc = [float(ii.split('Acc@1:')[1].strip().split("Acc@5:")[0].split("(")[1].split(")")[0]) for ii in aa if test_epoch_end_pattern in ii]

    print(f"{len(train_loss) = }, {len(lr) = }, {len(val_loss) = }, {len(val_acc) = }")
    return {"loss": train_loss, "lr": lr, "val_loss": val_loss, "val_acc": val_acc}


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


def plot_and_peak_scatter(ax, array, peak_method, label, color=None, **kwargs):
    for id, ii in enumerate(array):
        if tf.math.is_nan(ii):
            array[id] = array[id - 1]
    ax.plot(array, label=label, color=color, **kwargs)
    pp = peak_method(array)
    vv = array[pp]
    ax.scatter(pp, vv, color=color, marker="v")
    ax.text(pp, vv, "{:.4f}".format(vv), va="bottom", ha="right", fontsize=9, rotation=0)


def plot_hists(hists, names=None, base_size=6, addition_plots=["lr"]):
    import os
    import json
    import matplotlib.pyplot as plt
    import numpy as np

    num_axes = 3 if addition_plots is not None and len(addition_plots) != 0 else 2
    fig, axes = plt.subplots(1, num_axes, figsize=(num_axes * base_size, base_size))
    hists = [hists] if isinstance(hists, (str, dict)) else hists
    names = names if isinstance(names, (list, tuple)) else [names]
    for id, hist in enumerate(hists):
        name = names[min(id, len(names) - 1)] if names != None else None
        if isinstance(hist, str):
            name = name if name != None else os.path.splitext(os.path.basename(hist))[0]
            with open(hist, "r") as ff:
                hist = json.load(ff)
        name = name if name != None else str(id)

        plot_and_peak_scatter(axes[0], hist["loss"], peak_method=np.argmin, label=name + " loss", color=None)
        color = axes[0].lines[-1].get_color()
        plot_and_peak_scatter(axes[0], hist["val_loss"], peak_method=np.argmin, label=name + " val_loss", color=color, linestyle="--")
        acc = hist.get("acc", hist.get("accuracy", []))
        if len(acc) > 0:  # For timm log
            plot_and_peak_scatter(axes[1], acc, peak_method=np.argmax, label=name + " accuracy", color=color)
        val_acc = hist.get("val_acc", hist.get("val_accuracy", []))
        plot_and_peak_scatter(axes[1], val_acc, peak_method=np.argmax, label=name + " val_accuracy", color=color, linestyle="--")
        if addition_plots is not None and len(addition_plots) != 0:
            for ii in addition_plots:
                plot_and_peak_scatter(axes[2], hist[ii], peak_method=np.argmin, label=name + " " + ii, color=color)
    for ax in axes:
        ax.legend()
        ax.grid(True)
    fig.tight_layout()
    return fig
