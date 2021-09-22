import os
import tensorflow as tf
import numpy as np
from tensorflow import keras
from keras_cv_attention_models.imagenet import data
from keras_cv_attention_models.imagenet import callbacks
from keras_cv_attention_models import model_surgery
import tensorflow_addons as tfa


def train(
    compiled_model,
    epochs,
    initial_epoch=0,
    data_name="imagenet2012",
    lr_scheduler=None,
    input_shape=(224, 224, 3),
    batch_size=64,
    magnitude=0,
    mixup_alpha=0,
    cutmix_alpha=0,
    basic_save_name=None,
):
    if compiled_model.compiled_loss is None:
        print(">>>> Error: Model NOT compiled.")
        return None

    train_dataset, test_dataset, total_images, num_classes, steps_per_epoch = data.init_dataset(
        data_name=data_name, batch_size=batch_size, input_shape=input_shape, magnitude=magnitude, mixup_alpha=mixup_alpha, cutmix_alpha=cutmix_alpha
    )

    if hasattr(lr_scheduler, "steps_per_epoch") and lr_scheduler.steps_per_epoch == -1:
        lr_scheduler.build(steps_per_epoch)
    is_lr_on_batch = True if hasattr(lr_scheduler, "steps_per_epoch") and lr_scheduler.steps_per_epoch > 0 else False

    if basic_save_name is None:
        basic_save_name = "{}_{}_batch_size_{}_randaug_{}_mixup_{}".format(compiled_model.name, data_name, batch_size, magnitude, mixup_alpha)
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
    from keras_cv_attention_models.imagenet import data
    from keras_cv_attention_models.imagenet import callbacks
    from keras_cv_attention_models import model_surgery
    from keras_cv_attention_models import aotnet, coatnet, cmt
    import tensorflow_addons as tfa

    input_shape = (224, 224, 3)
    batch_size = 32 * strategy.num_replicas_in_sync
    lr_base_512 = 1e-2
    l2_weight_decay = 0
    optimizer_wd_mul = 1e-1
    magnitude = 10
    mixup_alpha = 0
    cutmix_alpha = 0
    label_smoothing = 0.1
    lr_decay_steps = 30  # [30, 60, 90] for constant decay
    lr_warmup = 4
    lr_min = 1e-5
    basic_save_name = None
    data_name = "imagenet2012"
    num_classes = 1000
    initial_epoch = 0

    with strategy.scope():
        # mm = keras.applications.ResNet50V2(include_top=False, input_shape=input_shape, weights=None)
        # nn = mm.outputs[0]
        # nn = keras.layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        # nn = keras.layers.Dropout(0.2)(nn)
        # nn = keras.layers.Dense(1000, activation="softmax", dtype="float32", name="predictions")(nn)
        # model = keras.models.Model(mm.inputs[0], nn, name=mm.name)
        model = coatnet.CoAtNet0(input_shape=input_shape, num_classes=num_classes, activation="gelu", drop_connect_rate=0.2, drop_rate=0.2)
        # model = cmt.CMTTiny(input_shape=input_shape, num_classes=num_classes, drop_connect_rate=0.2, drop_rate=0.2)
        # model = aotnet.AotNet(num_blocks=[3, 4, 6, 3], strides=[1, 2, 2, 2], activation='swish', preact=True, avg_pool_down=True, drop_connect_rate=0.2, drop_rate=0.2, model_name='aotnet50_swish_preact_avg_down_drop02_mixup_0')

        if l2_weight_decay != 0:
            model = model_surgery.add_l2_regularizer_2_model(model, weight_decay=l2_weight_decay, apply_to_batch_normal=False)

        lr_base = lr_base_512 * batch_size / 512
        if optimizer_wd_mul > 0:
            optimizer = tfa.optimizers.AdamW(learning_rate=lr_base, weight_decay=lr_base * optimizer_wd_mul)
        else:
            optimizer = keras.optimizers.SGD(learning_rate=lr_base, momentum=0.9)
        model.compile(optimizer=optimizer, loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing), metrics=["acc"])
        if isinstance(lr_decay_steps, list):
            constant_lr_sch = lambda epoch: callbacks.constant_scheduler(epoch, lr_base=lr_base, lr_decay_steps=lr_decay_steps, warmup=lr_warmup)
            lr_scheduler = keras.callbacks.LearningRateScheduler(constant_lr_sch)
            epochs = lr_decay_steps[-1] + lr_decay_steps[0] + lr_warmup  # 124 for lr_decay_steps=[30, 60, 90], lr_warmup=4
        else:
            lr_scheduler = callbacks.CosineLrScheduler(
                lr_base, first_restart_step=lr_decay_steps, m_mul=0.5, t_mul=2.0, lr_min=lr_min, warmup=lr_warmup, steps_per_epoch=-1
            )
            # lr_scheduler = callbacks.CosineLrSchedulerEpoch(lr_base, first_restart_step=lr_decay_steps, m_mul=0.5, t_mul=2.0, lr_min=lr_min, warmup=lr_warmup)
            epochs = lr_decay_steps * 3 + lr_warmup  # 94 for lr_decay_steps=30, lr_warmup=4

        imagenet.train(
            model,
            epochs=epochs,
            initial_epoch=0,
            data_name=data_name,
            lr_scheduler=lr_scheduler,
            input_shape=input_shape,
            batch_size=batch_size,
            magnitude=magnitude,
            mixup_alpha=mixup_alpha,
            cutmix_alpha=cutmix_alpha,
            basic_save_name=basic_save_name,
        )
