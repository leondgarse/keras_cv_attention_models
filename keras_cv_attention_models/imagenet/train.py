import os
import tensorflow as tf
from tensorflow import keras
from keras_cv_attention_models.imagenet import callbacks
import tensorflow_addons as tfa


def train(
    compiled_model,
    epochs,
    train_dataset,
    test_dataset=None,
    initial_epoch=0,
    lr_scheduler=None,
    basic_save_name=None,
):
    if compiled_model.compiled_loss is None:
        print(">>>> Error: Model NOT compiled.")
        return None

    steps_per_epoch = len(train_dataset)
    if hasattr(lr_scheduler, "steps_per_epoch") and lr_scheduler.steps_per_epoch == -1:
        lr_scheduler.build(steps_per_epoch)
    is_lr_on_batch = True if hasattr(lr_scheduler, "steps_per_epoch") and lr_scheduler.steps_per_epoch > 0 else False

    if basic_save_name is None:
        basic_save_name = "{}".format(compiled_model.name)
    # ckpt_path = os.path.join("checkpoints", basic_save_name + "epoch_{epoch:03d}_val_acc_{val_acc:.4f}.h5")
    # cur_callbacks = [keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True)]
    # cur_callbacks = [keras.callbacks.ModelCheckpoint(os.path.join("checkpoints", basic_save_name + ".h5"))]
    cur_callbacks = [callbacks.MyCheckpoint(basic_save_name, monitor="val_acc")]
    hist_file = os.path.join("checkpoints", basic_save_name + "_hist.json")
    if initial_epoch == 0 and os.path.exists(hist_file):
        os.remove(hist_file)
    cur_callbacks.append(callbacks.MyHistory(initial_file=hist_file))
    cur_callbacks.append(keras.callbacks.TerminateOnNaN())
    if lr_scheduler is not None:
        cur_callbacks.append(lr_scheduler)

    compiled_opt = compiled_model.optimizer
    compiled_opt = compiled_opt.inner_optimizer if isinstance(compiled_opt, keras.mixed_precision.LossScaleOptimizer) else compiled_opt
    if lr_scheduler is not None and isinstance(compiled_opt, tfa.optimizers.weight_decay_optimizers.DecoupledWeightDecayExtension):
        print(">>>> Append weight decay callback...")
        lr_base, wd_base = compiled_model.optimizer.lr.numpy(), compiled_model.optimizer.weight_decay.numpy()
        wd_callback = callbacks.OptimizerWeightDecay(lr_base, wd_base, is_lr_on_batch=is_lr_on_batch)
        cur_callbacks.append(wd_callback)  # should be after lr_scheduler

    compiled_model.fit(
        train_dataset,
        epochs=epochs,
        verbose=1,
        callbacks=cur_callbacks,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_dataset,
        use_multiprocessing=True,
        workers=4,
    )


def plot_and_peak_scatter(ax, array, peak_method, label, color=None, **kwargs):
    for id, ii in enumerate(array):
        if np.isnan(ii):
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

    num_axes = 3 if addition_plots is not None and len(addition_plots) != 0 else 2
    fig, axes = plt.subplots(1, num_axes, figsize=(num_axes * base_size, base_size))
    for id, hist in enumerate(hists):
        name = names[id] if names != None else None
        if isinstance(hist, str):
            name = name if name != None else os.path.splitext(os.path.basename(hist))[0]
            with open(hist, "r") as ff:
                hist = json.load(ff)
        name = name if name != None else str(id)

        plot_and_peak_scatter(axes[0], hist["loss"], peak_method=np.argmin, label=name + " loss", color=None)
        color = axes[0].lines[-1].get_color()
        plot_and_peak_scatter(axes[0], hist["val_loss"], peak_method=np.argmin, label=name + " val_loss", color=color, linestyle="--")
        acc = hist.get("acc", hist.get("accuracy", []))
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


if __name__ == "__test__":
    gpus = tf.config.experimental.get_visible_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    strategy = tf.distribute.MirroredStrategy() if len(gpus) > 1 else tf.distribute.OneDeviceStrategy(device="/gpu:0")
    keras.mixed_precision.set_global_policy("mixed_float16")

    from keras_cv_attention_models import imagenet
    from keras_cv_attention_models.imagenet import data, callbacks, train
    from keras_cv_attention_models import model_surgery
    from keras_cv_attention_models import aotnet, coatnet, cmt
    import tensorflow_addons as tfa

    input_shape = (160, 160, 3)
    batch_size = 256 * strategy.num_replicas_in_sync
    lr_base_512 = 8e-3
    l2_weight_decay = 0
    optimizer_wd_mul = 0.02
    label_smoothing = 0
    lr_decay_steps = 100  # [30, 60, 90] for constant decay
    lr_warmup = 5
    lr_min = 1e-6
    epochs = 105
    initial_epoch = 0

    data_name = "imagenet2012"
    magnitude = 6
    mixup_alpha = 0.1
    cutmix_alpha = 1.0
    random_crop_min = 0.08

    train_dataset, test_dataset, total_images, num_classes, steps_per_epoch = data.init_dataset(
        data_name=data_name,
        input_shape=input_shape,
        batch_size=batch_size,
        mixup_alpha=mixup_alpha,
        cutmix_alpha=cutmix_alpha,
        rescale_mode="tf",
        central_crop=1.0,
        random_crop_min=random_crop_min,
        resize_method="bilinear",
        random_erasing_prob=0.0,
        magnitude=magnitude,
    )

    lr_base = lr_base_512 * batch_size / 512
    if isinstance(lr_decay_steps, list):
        constant_lr_sch = lambda epoch: callbacks.constant_scheduler(epoch, lr_base=lr_base, lr_decay_steps=lr_decay_steps, warmup=lr_warmup)
        lr_scheduler = keras.callbacks.LearningRateScheduler(constant_lr_sch)
        epochs = epochs if epochs != 0 else lr_decay_steps[-1] + lr_decay_steps[0] + lr_warmup  # 124 for lr_decay_steps=[30, 60, 90], lr_warmup=4
    else:
        lr_scheduler = callbacks.CosineLrScheduler(
            lr_base, first_restart_step=lr_decay_steps, m_mul=0.5, t_mul=2.0, lr_min=lr_min, warmup=lr_warmup, steps_per_epoch=-1
        )
        # lr_scheduler = callbacks.CosineLrSchedulerEpoch(lr_base, first_restart_step=lr_decay_steps, m_mul=0.5, t_mul=2.0, lr_min=lr_min, warmup=lr_warmup)
        epochs = epochs if epochs != 0 else lr_decay_steps * 3 + lr_warmup  # 94 for lr_decay_steps=30, lr_warmup=4

    with strategy.scope():
        # model = cmt.CMTTiny(input_shape=input_shape, num_classes=num_classes, drop_connect_rate=0.2, drop_rate=0.2)
        model = keras.applications.ResNet50(weights=None, input_shape=input_shape)
        # model = keras.models.load_model('checkpoints/resnet50_imagenet2012_batch_size_256_randaug_5_mixup_0.1_cutmix_1.0_RRC_0.08_LAMB_lr0.002_wd0.02_latest.h5')

        if model.optimizer is None:
            if l2_weight_decay != 0:
                model = model_surgery.add_l2_regularizer_2_model(model, weight_decay=l2_weight_decay, apply_to_batch_normal=False)
            if optimizer_wd_mul > 0:
                # optimizer = tfa.optimizers.AdamW(learning_rate=lr_base, weight_decay=lr_base * optimizer_wd_mul)
                optimizer = tfa.optimizers.LAMB(learning_rate=lr_base, weight_decay_rate=optimizer_wd_mul)
            else:
                optimizer = keras.optimizers.SGD(learning_rate=lr_base, momentum=0.9)
            model.compile(optimizer=optimizer, loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing), metrics=["acc"])

        compiled_opt = model.optimizer
        compiled_opt = compiled_opt.inner_optimizer if isinstance(compiled_opt, keras.mixed_precision.LossScaleOptimizer) else compiled_opt
        basic_save_name = "{}_{}_batch_size_{}".format(model.name, data_name, batch_size)
        basic_save_name += "_randaug_{}_mixup_{}_cutmix_{}_RRC_{}".format(magnitude, mixup_alpha, cutmix_alpha, random_crop_min)
        basic_save_name += "_{}_lr{}_wd{}".format(compiled_opt.__class__.__name__, lr_base_512, optimizer_wd_mul or l2_weight_decay)
        train(model, epochs, train_dataset, test_dataset, initial_epoch, lr_scheduler=lr_scheduler, basic_save_name=basic_save_name)
