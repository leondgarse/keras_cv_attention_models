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
        stacked_images = np.vstack([np.hstack(images[ii * cols : (rr + 1) * cols]) for ii in range(rows)])

    ax.imshow(stacked_images)
    ax.set_axis_off()
    ax.grid(False)
    plt.tight_layout()
    # plt.show()
    return ax, stacked_images


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
    image, bboxes, labels=None, confidences=None, is_bbox_width_first=False, ax=None, label_font_size=8, num_classes=80, indices_2_labels=None
):
    import matplotlib.pyplot as plt
    import numpy as np
    from keras_cv_attention_models.coco import info

    need_plt_show = False
    if ax is None:
        fig, ax = plt.subplots()
        need_plt_show = True
    ax.imshow(image)
    bboxes = np.array(bboxes)
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


""" tensorboard_parallel_coordinates_plot """


def tensorboard_parallel_coordinates_plot(dataframe, metrics_name, metrics_display_name=None, skip_columns=[], log_dir="logs/hparam_tuning"):
    """
    Simmilar results with: [Visualize the results in TensorBoard's HParams plugin](https://www.tensorflow.org/tensorboard/hyperparameter_tuning_with_hparams#4_visualize_the_results_in_tensorboards_hparams_plugin).

    Wrapped function just plotting ignoring training in the tutorial.
    The logic is using `metrics_name` specified key as metrics, using other columns as `HParams`.
    For any other detail, refer original tutorial.

    Example:
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

    Example
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
    for name, group in dd.groupby(dd["series"]):
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
        ax.scatter(xx, yy, label=name, marker=markers[marker_id])
        marker_id += 1
        for _, cur in group.iterrows():
            # print(cur)
            text = cur["model"][len(name) :]
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
