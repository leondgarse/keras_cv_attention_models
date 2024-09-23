import numpy as np


def get_plot_cols_rows(total, rows=-1, ceil_mode=False):
    if rows == -1 and total < 8:
        rows = 1  # for total in [1, 7], plot 1 row only
    elif rows == -1:
        rr = int(np.floor(np.sqrt(total)))
        for ii in range(1, rr + 1)[::-1]:
            if total % ii == 0:
                rows = ii
                break
    if ceil_mode:
        cols = int(np.ceil(total / rows))
    else:
        cols = total // rows
    return cols, rows


def put_text_on_image(image, text, coord=(5, 5), color=(255, 0, 0)):
    from PIL import Image
    from PIL import ImageDraw

    # from PIL import ImageFont

    image = image * 255 if image.max() < 2 else image
    img = Image.fromarray(image.astype("uint8"))
    draw = ImageDraw.Draw(img)
    draw.text(coord, str(text), color)
    return np.array(img)


def stack_and_plot_images(images, texts=None, margin=5, margin_value=0, rows=-1, ax=None, base_size=3):
    """Stack and plot a list of images. Returns ax, stacked_images"""
    import matplotlib.pyplot as plt

    cols, rows = get_plot_cols_rows(len(images), rows, ceil_mode=True)
    if cols * rows > len(images):
        padded = cols * rows - len(images)
        images += [np.zeros_like(images[-1]) + 255 for ii in range(padded)]
    # images = images[: rows * cols]
    # print(">>>> rows:", rows, ", cols:", cols, ", total:", len(images))

    if texts is not None:
        images = [put_text_on_image(imm, itt) for imm, itt in zip(images, texts)] + list(images[len(texts) :])
        images = np.array(images)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(base_size * cols, base_size * rows))

    if margin > 0:
        ww_margin = np.zeros_like(images[0][:, :margin]) + margin_value
        ww_margined_images = [np.hstack([ii, ww_margin]) for ii in images]
        hstacked_images = [np.hstack(ww_margined_images[ii : ii + cols]) for ii in range(0, len(ww_margined_images), cols)]

        hh_margin = np.zeros_like(hstacked_images[0][:margin]) + margin_value
        hh_margined_images = [np.vstack([ii, hh_margin]) for ii in hstacked_images]
        vstacked_images = np.vstack(hh_margined_images)

        stacked_images = vstacked_images[:-margin, :-margin]
    else:
        stacked_images = np.vstack([np.hstack(images[ii * cols : (ii + 1) * cols]) for ii in range(rows)])

    ax.imshow(stacked_images)
    ax.set_axis_off()
    ax.grid(False)
    plt.tight_layout()
    # plt.show()
    return ax, stacked_images


""" Show recognition dataset """


def show_batch_sample(dataset, rescale_mode="tf", rows=-1, caption_tokenizer=None, base_size=3, indices_2_labels=None):
    from keras_cv_attention_models.common_layers import init_mean_std_by_rescale_mode
    from keras_cv_attention_models.imagenet.eval_func import decode_predictions
    from keras_cv_attention_models.backend import is_channels_last

    if isinstance(dataset, (list, tuple)):
        images, labels = dataset
    else:
        images, labels = next(iter(dataset))

        if isinstance(images, tuple):  # caption datasets
            images, labels = images
        elif isinstance(labels, tuple):  # token_label datasets
            labels, token_label = labels
    images, labels = np.array(images), np.array(labels)

    if caption_tokenizer is not None:
        labels = [caption_tokenizer(ii) for ii in labels]

    mean, std = init_mean_std_by_rescale_mode(rescale_mode)
    mean, std = (mean.numpy(), std.numpy()) if hasattr(mean, "numpy") else (mean, std)
    images = (images * std + mean) / 255
    images = images if is_channels_last() else images.transpose([0, 2, 3, 1])

    if isinstance(labels[0], str):
        pass  # caption datasets
    elif labels.shape[-1] == 1000:
        labels = [ii[0][1] for ii in decode_predictions(labels, top=1)]
    elif labels[0].ndim == 1:
        labels = np.argmax(labels, axis=-1)  # If 2 dimension

    if not isinstance(labels[0], str) and indices_2_labels is not None:
        labels = [indices_2_labels.get(label, indices_2_labels.get(str(label), str(label))) for label in labels]
    ax, _ = stack_and_plot_images(images, texts=labels, rows=rows, ax=None, base_size=base_size)
    return ax


def show_token_label_patches_single(image, token_label, rescale_mode="tf", top_k=3, resize_patch_shape=(160, 160)):
    from keras_cv_attention_models.backend import functional, numpy_image_resize

    mean, std = init_mean_std_by_rescale_mode(rescale_mode)
    mean, std = (mean.numpy(), std.numpy()) if hasattr(mean, "numpy") else (mean, std)
    image = (image * std + mean) / 255

    height, width = image.shape[:2]
    num_height_patch, num_width_patch = token_label.shape[0], token_label.shape[1]
    height_patch, width_patch = int(np.ceil(height / num_height_patch)), int(np.ceil(width / num_width_patch))

    token_label_scores, token_label_classes = functional.top_k(functional.convert_to_tensor(token_label), top_k)
    token_label_scores, token_label_classes = np.array(token_label_scores), np.array(token_label_classes)
    # fig, axes = plt.subplots(num_height_patch, num_width_patch)

    image_pathes, labels = [], []
    for hh_id in range(num_height_patch):
        hh_image = image[hh_id * height_patch : (hh_id + 1) * height_patch]
        for ww_id in range(num_width_patch):
            image_patch = hh_image[:, ww_id * width_patch : (ww_id + 1) * width_patch]
            image_pathes.append(numpy_image_resize(image_patch, resize_patch_shape))
            scores = ",".join(["{:.1f}".format(ii * 100) for ii in token_label_scores[hh_id, ww_id]])
            classes = ",".join(["{:d}".format(ii) for ii in token_label_classes[hh_id, ww_id].astype("int")])
            labels.append(classes + "\n" + scores)
    plot_func.stack_and_plot_images(image_pathes, labels)


""" Show detection results """


def draw_bboxes(bboxes, ax=None):
    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots()
    bboxes = np.array(bboxes).astype("int32")
    for bb in bboxes:
        ax.plot(bb[[1, 1, 3, 3, 1]], bb[[0, 2, 2, 0, 0]])
    plt.show()
    return ax


def show_image_with_bboxes(
    image, bboxes=None, labels=None, confidences=None, masks=None, is_bbox_width_first=False, ax=None, label_font_size=8, num_classes=80, indices_2_labels=None
):
    import matplotlib.pyplot as plt
    import numpy as np
    from keras_cv_attention_models.coco import info
    from keras_cv_attention_models.backend import numpy_image_resize

    need_plt_show = False
    if ax is None:
        fig, ax = plt.subplots()
        need_plt_show = True

    ax.imshow(image)
    masks = [] if masks is None else np.array(masks)
    for mask in masks:  # Show segmentation results
        random_color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)[None, None]
        resized_mask = numpy_image_resize(mask[:, :, None], image.shape[:2])
        ax.imshow(np.greater(resized_mask, 0.5) * random_color)

    bboxes = np.zeros([0, 4]) if bboxes is None else np.array(bboxes)
    if is_bbox_width_first:
        bboxes = bboxes[:, [1, 0, 3, 2]]
    for id, bb in enumerate(bboxes):
        # bbox is [top, left, bottom, right]
        bb = [bb[0] * image.shape[0], bb[1] * image.shape[1], bb[2] * image.shape[0], bb[3] * image.shape[1]]
        bb = np.array(bb).astype("int32")
        ax.plot(bb[[1, 1, 3, 3, 1]], bb[[0, 2, 2, 0, 0]])

        if labels is not None:
            label = int(labels[id])
            if indices_2_labels is not None:
                label = indices_2_labels.get(label, indices_2_labels.get(str(label), "None"))
            elif num_classes == 90:
                label = info.COCO_90_LABEL_DICT[label]
            elif num_classes == 80:
                label = info.COCO_80_LABEL_DICT[label]

            if confidences is not None:
                label += ": {:.4f}".format(float(confidences[id]))
            color = ax.lines[-1].get_color()
            # ax.text(bb[1], bb[0] - 5, "label: {}, {}".format(label, info.COCO_80_LABEL_DICT[label]), color=color, fontsize=8)
            ax.text(bb[1], bb[0] - 5, label, color=color, fontsize=label_font_size)
    ax.set_axis_off()
    plt.tight_layout()
    if need_plt_show:
        plt.show()
    return ax


def show_image_with_bboxes_and_masks(
    image, bboxes=None, labels=None, confidences=None, masks=None, is_bbox_width_first=False, ax=None, label_font_size=8, num_classes=80, indices_2_labels=None
):
    return show_image_with_bboxes(**locals())


def show_detection_batch_sample(
    dataset, rescale_mode="torch", rows=-1, label_font_size=8, base_size=3, anchors_mode="efficientdet", indices_2_labels=None, **anchor_kwargs
):
    import matplotlib.pyplot as plt
    from keras_cv_attention_models.common_layers import init_mean_std_by_rescale_mode
    from keras_cv_attention_models.coco import anchors_func
    from keras_cv_attention_models.backend import is_channels_last

    if isinstance(dataset, (list, tuple)):
        images, labels = dataset
    else:
        images, labels = next(iter(dataset))
    images, labels = np.array(images), np.array(labels)

    mean, std = init_mean_std_by_rescale_mode(rescale_mode)
    images = (images * std + mean) / 255
    images = images if is_channels_last() else images.transpose([0, 2, 3, 1])

    if anchors_mode == anchors_func.YOLOR_MODE:
        pyramid_levels = anchors_func.get_pyramid_levels_by_anchors(images.shape[1:-1], labels.shape[1])
        anchors = anchors_func.get_yolor_anchors(images.shape[1:-1], pyramid_levels=pyramid_levels, is_for_training=False)
    elif not anchors_mode == anchors_func.ANCHOR_FREE_MODE:
        pyramid_levels = anchors_func.get_pyramid_levels_by_anchors(images.shape[1:-1], labels.shape[1])
        anchors = anchors_func.get_anchors(images.shape[1:-1], pyramid_levels, **anchor_kwargs)

    cols, rows = get_plot_cols_rows(len(images), rows, ceil_mode=True)
    fig, axes = plt.subplots(rows, cols, figsize=(base_size * cols, base_size * rows))
    axes = axes.flatten()

    for ax, image, label in zip(axes, images, labels):
        if label.shape[-1] == 5:
            pick = label[:, -1] >= 0
            valid_preds = label[pick]
        else:
            pick = label[:, -1] == 1
            valid_preds = label[pick]
            valid_label = np.argmax(valid_preds[:, 4:-1], axis=-1)[:, None]
            valid_preds = np.concatenate([valid_preds[:, :4], valid_label], axis=-1)

        if anchors_mode == anchors_func.YOLOR_MODE:
            valid_anchors = anchors[pick]
            decoded_centers = (valid_preds[:, :2] + 0.5) * valid_anchors[:, 4:] + valid_anchors[:, :2]
            decoded_hw = valid_preds[:, 2:4] * valid_anchors[:, 4:]

            top_left = decoded_centers - decoded_hw * 0.5
            bottom_right = top_left + decoded_hw
            decoded_corner = np.concatenate([top_left, bottom_right], axis=-1)

            valid_preds = np.concatenate([decoded_corner, valid_preds[:, -1:]], axis=-1)
        elif not anchors_mode == anchors_func.ANCHOR_FREE_MODE:
            valid_anchors = anchors[pick]
            valid_preds = anchors_func.decode_bboxes(valid_preds, valid_anchors)
        show_image_with_bboxes(image, valid_preds[:, :4], valid_preds[:, -1], ax=ax, label_font_size=label_font_size, indices_2_labels=indices_2_labels)
    fig.tight_layout()
    plt.show()
    return fig


""" Show clip results """


def show_images_texts_similarity(images, texts, similarity, ax=None, base_size=8, title=None):
    """
    Copied and modified from: https://github.com/mlfoundations/open_clip/blob/main/docs/Interacting_with_open_clip.ipynb

    Args:
      similarity: in shape `[images.shape[0], texts.shape[0]]`

    Examples:
    >>> from keras_cv_attention_models import test_images, plot_func
    >>> images = [test_images.dog(), test_images.cat(), test_images.dog_cat()] * 3
    >>> texts = ["dog", "cat", "dog_cat"]
    >>> similarity = np.random.uniform(size=[9, 3])
    >>> _ = plot_func.show_images_texts_similarity(images, texts, similarity)
    """
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(base_size * 1.4, base_size))  # Wider one
    font_size = 9 * base_size / 8
    yticks_size = 10 * base_size / 8

    similarity = similarity.detach() if hasattr(similarity, "detach") else similarity
    similarity = similarity.numpy() if hasattr(similarity, "numpy") else similarity
    num_images, num_texts = similarity.shape[0], similarity.shape[1]

    ax.imshow(similarity.T, vmin=0.1, vmax=0.3)
    ax.set_yticks(range(num_texts), texts, fontsize=yticks_size)
    ax.set_xticks([])
    for ii, image in enumerate(images):
        ax.imshow(image, extent=(ii - 0.5, ii + 0.5, -1.6, -0.6), origin="lower")
    for xx in range(num_images):
        for yy in range(num_texts):
            ax.text(xx, yy, f"{similarity[xx, yy]:.2f}", ha="center", va="center", size=font_size)

    for side in ["left", "top", "right", "bottom"]:
        plt.gca().spines[side].set_visible(False)

    ax.set_xlim([-0.5, num_images - 0.5])
    ax.set_ylim([num_texts + 0.5, -2])
    ax.grid(False)

    if title:
        ax.set_title(title)
    plt.tight_layout()
    # plt.show()
    return ax


""" Plot training hists """


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
    import os
    import json
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
            hist["val_ap 0.50:0.95"] = [ii[0] for ii in hist["val_ap_ar"]]
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
        if len(acc) > 0:
            plot_and_peak_scatter(axes[1], acc, np.argmax, name + " " + acc_key, skip_first, color, cur_va, cur_pred_curve)

        val_acc = hist.get(val_acc_key, [])
        if len(val_acc) > 0:
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


""" tensorboard_parallel_coordinates_plot """


def tensorboard_parallel_coordinates_plot(dataframe, metrics_name, metrics_display_name=None, skip_columns=[], log_dir="logs/hparam_tuning"):
    """
    Simmilar results with: [Visualize the results in TensorBoard's HParams plugin](https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams#4_visualize_the_results_in_tensorboards_hparams_plugin).

    Wrapped function just plotting ignoring training in the tutorial.
    The logic is using `metrics_name` specified key as metrics, using other columns as `HParams`.
    For any other detail, refer original tutorial.

    Examples:
    >>> import pandas as pd
    >>> aotnet50_imagnet_results = {
    >>>     "optimizer": ["lamb", "lamb", "adamw", "adamw", "adamw"],
    >>>     "rescale_mode": ["torch", "tf", "torch", "torch", "torch"],
    >>>     "lr_base": [8e-3, 8e-3, 4e-3, 4e-3, 8e-3],
    >>>     "weight_decay": [0.05, 0.05, 0.05, 0.02, 0.02],
    >>>     "accuracy": [78.48, 78.31, 77.92, 78.06, 78.27],
    >>> }
    >>> aa = pd.DataFrame(aotnet50_imagnet_results)

    >>> from keras_cv_attention_models import plot_func
    >>> plot_func.tensorboard_parallel_coordinates_plot(aa, 'accuracy', log_dir="logs/aotnet50_imagnet_results")
    >>> # >>>> Start tensorboard by: ! tensorboard --logdir logs/aotnet50_imagnet_results
    >>> # >>>> Then select `HPARAMS` -> `PARALLEL COORDINATES VIEW`

    >>> ! tensorboard --logdir logs/aotnet50_imagnet_results
    """
    import os
    import pandas as pd
    import tensorflow as tf
    from tensorboard.plugins.hparams import api as hp

    skip_columns = skip_columns + [metrics_name]
    to_hp_discrete = lambda column: hp.HParam(column, hp.Discrete(np.unique(dataframe[column].values).tolist()))
    hp_params_dict = {column: to_hp_discrete(column) for column in dataframe.columns if column not in skip_columns}

    if dataframe[metrics_name].values.dtype == "object":  # Not numeric
        import json

        metrics_map = {ii: id for id, ii in enumerate(np.unique(dataframe[metrics_name]))}
        description = json.dumps(metrics_map)
    else:
        metrics_map, description = None, None

    METRICS = metrics_name if metrics_display_name is None else metrics_display_name
    with tf.summary.create_file_writer(log_dir).as_default():
        metrics = [hp.Metric(METRICS, display_name=METRICS, description=description)]
        hp.hparams_config(hparams=list(hp_params_dict.values()), metrics=metrics)

    for id in dataframe.index:
        log = dataframe.iloc[id]
        hparams = {hp_unit: log[column] for column, hp_unit in hp_params_dict.items()}
        print({hp_unit.name: hparams[hp_unit] for hp_unit in hparams})
        run_dir = os.path.join(log_dir, "run-%d" % id)
        with tf.summary.create_file_writer(run_dir).as_default():
            hp.hparams(hparams)  # record the values used in this trial
            metric_item = log[metrics_name] if metrics_map is None else metrics_map[log[metrics_name]]
            tf.summary.scalar(METRICS, metric_item, step=1)

    print()
    if metrics_map is not None:
        print(">>>> metrics_map:", metrics_map)
    print(">>>> Start tensorboard by: ! tensorboard --logdir {}".format(log_dir))
    print(">>>> Then select `HPARAMS` -> `PARALLEL COORDINATES VIEW`")


""" Plot model summary """


def plot_model_summary(
    plot_series, x_label="inference_qps", y_label="acc_metrics", model_table="model_summary.csv", allow_extras=None, log_scale_x=False, ax=None
):
    """
    Args:
      plot_series: list value for filtering itmes by model_table "series" column.
      x_label: string value from column names in `model_table`, x axis values.
      y_label: string value from column names in `model_table`, y axis values.
      model_table: a csv file path or loaded pandas DataFrame.
          If DataFrame, columns ["series", "model"] and [x_label, y_label] are required.
          "series" means which model series this model belongs to, and "model" is the actual model name.
      allow_extras: list value for allowing plotting data with extra pretrained.
          Default None for plotting imagenet without any extra pretrained only.
          Special string value "all" for allowing all extras.
      log_scale_x: boolean value if setting x scale in log distribution.
      ax: plotting on specific matplotlib ax. Default None for creating a new figure.

    Examples:
    >>> from keras_cv_attention_models import plot_func
    >>> plot_series = ["convnextv2", "efficientnetv2", "efficientvit_b", "fasternet", "fastervit"]
    >>> plot_func.plot_model_summary(plot_series, x_label='inference_qps', model_table="model_summary.csv", allow_extras=None)
    >>> plt.savefig('foo.png', dpi=150)  # Save if needed, with dpi speficified

    # Using custom DataFrame
    >>> from keras_cv_attention_models import plot_func
    >>> dd = pd.DataFrame({
    >>>     "series": ['Res', 'Res', 'Res', 'Res', 'Trans', 'Trans', 'Trans', 'cc'],
    >>>     'model': ['res50', 'res101', 'res201', 'aa4', 'trans_tiny', 'trans_small', 'trans_big', 'cc1'],
    >>>     'test_key': [0.2, 0.3, 0.5, 0.55, 0.1, 0.3, 0.4, 0.2],
    >>>     'valuable': [0.8, 0.92, 0.93, 0.96, 0.71, 0.74, 0.85, 0.98],
    >>> })
    >>> plot_func.plot_model_summary(plot_series=['Res', 'trans'], x_label='test_key', y_label='valuable', model_table=dd)
    """
    import matplotlib
    import matplotlib.pyplot as plt

    # ('o', 'v', '^', '<', '>', '8', 's', 'p', '*', 'h', 'H', 'D', 'd', 'P', 'X'), excluded ["1", "2", "3", "4", "+", "x", '.', ',', '|', '_']
    markers = matplotlib.markers.MarkerStyle.filled_markers
    fontsize = 9
    pre_x_labelsize, pre_y_labelsize = matplotlib.rcParams["xtick.labelsize"], matplotlib.rcParams["ytick.labelsize"]
    matplotlib.rcParams["xtick.labelsize"], matplotlib.rcParams["ytick.labelsize"] = fontsize, fontsize

    if isinstance(model_table, str):
        import pandas as pd

        dd = pd.read_csv(model_table)
    else:
        dd = model_table
    # dd = dd[dd.category == "Recognition"]
    dd = dd[dd[x_label].notnull()]
    dd = dd[dd[y_label].notnull()]

    if ax is None:
        fig, ax = plt.subplots()

    has_extra = "extra" in dd.columns
    plot_series = None if plot_series is None else [ii.lower() for ii in plot_series]
    gather_extras = []
    allow_extras = [] if allow_extras is None else allow_extras
    marker_id = 0
    for (name, is_deploy), group in dd.groupby([dd["series"], ["deploy" in ii for ii in dd["model"]]]):
        if plot_series is not None and name.lower() not in plot_series:
            continue

        if has_extra and allow_extras != "all":
            gather_extras.extend(group["extra"].values)
            extra_condition = group["extra"].isnull()
            if allow_extras:
                extra_condition = np.logical_or(extra_condition, [ii in allow_extras for ii in group["extra"]])
            group = group[extra_condition]
        xx = group[x_label].values
        yy = group[y_label].values
        if len(xx) == 0 or len(yy) == 0:
            print("Empty or all filtered for series", name)
            continue

        ax.plot(xx, yy)
        label = (name + "_deploy") if is_deploy else name
        prefix_len = len(name.split("_")[0])
        ax.scatter(xx, yy, label=label, marker=markers[marker_id])
        marker_id = (marker_id + 1) if marker_id < len(markers) - 1 else 0
        for _, cur in group.iterrows():
            # print(cur)
            text = cur["model"][prefix_len:]
            if has_extra and str(cur["extra"]) != "nan":
                text += "," + cur["extra"]
            ax.text(cur[x_label], cur[y_label], text[1:] if text.startswith("_") else text, fontsize=fontsize)
    if log_scale_x:
        ax.set_xscale("log")
        min_value_x, max_value_x = ax.get_xlim()
        # print(f"{min_value_x = }, {max_value_x = }")
        min_value_x_log, max_value_x_log = np.log(max(min_value_x, 1e-3)) / np.log(10), np.log(max_value_x) / np.log(10)
        ticks = [10**ii for ii in np.arange(min_value_x_log, max_value_x_log, (max_value_x_log - min_value_x_log) / 10)]
        ax.set_xticks(ticks, labels=["{:.3f}".format(ii) for ii in ticks], fontsize=fontsize)

    ax.set_xlabel(x_label + " (log distribution)" if log_scale_x else "", fontsize=fontsize + 1)
    ax.set_ylabel(y_label, fontsize=fontsize + 1)
    ax.legend(fontsize=fontsize)
    ax.grid(True)
    matplotlib.rcParams["xtick.labelsize"], matplotlib.rcParams["ytick.labelsize"] = pre_x_labelsize, pre_y_labelsize

    plt.tight_layout()
    plt.show()
    other_extras = [ii for ii in set(gather_extras) - set(allow_extras) if isinstance(ii, str)]
    if len(other_extras) > 0:
        print("All other extras:", other_extras)
    return ax
