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


def plot_model_summary(plot_series, x_label, y_label="acc_metrics", model_table="model_summary.csv", filter_extra=True, ax=None):
    """
    >>> from keras_cv_attention_models import plot_func
    >>> plot_series = ["efficientvit_b", "efficientvit_m", "efficientnet", "efficientnetv2"]
    >>> plot_func.plot_model_summary(plot_series, x_label='inference', model_table="model_summary.csv")
    """
    import matplotlib.pyplot as plt

    if isinstance(model_table, str):
        import pandas as pd

        dd = pd.read_csv(model_table)
    else:
        dd = model_table
    # dd = dd[dd.category == "Recognition"]
    dd = dd[dd[x_label].notnull()]
    dd = dd[dd[y_label].notnull()]
    if filter_extra:
        dd = dd[dd["extra"].isnull()]

    if ax is None:
        fig, ax = plt.subplots()

    plot_series = None if plot_series is None else [ii.lower() for ii in plot_series]
    for name, group in dd.groupby(dd["series"]):
        if plot_series is not None and name.lower() not in plot_series:
            continue
        xx = group[x_label].values
        yy = group[y_label].values
        plt.scatter(xx, yy, label=name)
        plt.plot(xx, yy)
        for _, cur in group.iterrows():
            # print(cur)
            extra = "" if str(cur["extra"]) == "nan" else ("," + cur["extra"])
            text = cur["model"][len(name) :] + extra
            plt.text(cur[x_label], cur[y_label], text)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()

    plt.show()
    return ax
