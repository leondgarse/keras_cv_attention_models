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
    lr_scheduler=None,
    initial_epoch=0,
    input_shape=(224, 224, 3),
    batch_size=64,
    magnitude=0,
    mixup_alpha=0,
    basic_save_name=None,
):
    if compiled_model.compiled_loss is None:
        print(">>>> Error: Model NOT compiled.")
        return None

    train_dataset, test_dataset, total_images, num_classes, steps_per_epoch = data.init_dataset(
        batch_size=batch_size, input_shape=input_shape, magnitude=magnitude, mixup_alpha=mixup_alpha
    )

    if hasattr(lr_scheduler, "steps_per_epoch") and lr_scheduler.steps_per_epoch == -1:
        lr_scheduler.build(steps_per_epoch)
    if basic_save_name is None:
        basic_save_name = "{}_imagenet_batch_size_{}_randaug_{}_mixup_{}".format(compiled_model.name, batch_size, magnitude, mixup_alpha)
    # ckpt_path = os.path.join("checkpoints", basic_save_name + "epoch_{epoch:02d}_val_acc_{val_acc:.2f}.h5")
    # cur_callbacks = [keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True)]
    cur_callbacks = [keras.callbacks.ModelCheckpoint(os.path.join("checkpoints", basic_save_name + ".h5"))]
    hist_file = os.path.join("checkpoints", basic_save_name + "hist.json")
    if initial_epoch == 0 and os.path.exists(hist_file):
        os.remove(hist_file)
    cur_callbacks.append(callbacks.MyHistory(initial_file=hist_file))
    cur_callbacks.append(keras.callbacks.TerminateOnNaN())
    if lr_scheduler is not None:
        cur_callbacks.append(lr_scheduler)
    if lr_scheduler is not None and isinstance(compiled_model.optimizer, tfa.optimizers.weight_decay_optimizers.DecoupledWeightDecayExtension):
        print(">>>> Append weight decay callback...")
        lr_base, wd_base = compiled_model.optimizer.lr.numpy(), compiled_model.optimizer.weight_decay.numpy()
        is_lr_on_batch = isinstance(lr_scheduler, callbacks.CosineLrScheduler)
        wd_callback = myCallbacks.OptimizerWeightDecay(lr_base, wd_base, is_lr_on_batch=is_lr_on_batch)
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


def plot_hists(hists, names=None, base_size=6):
    import os
    import json
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(2 * base_size, base_size))
    for id, hist in enumerate(hists):
        name = names[id] if names != None else None
        if isinstance(hist, str):
            name = name if name != None else os.path.splitext(os.path.basename(hist))[0]
            with open(hist, "r") as ff:
                hist = json.load(ff)
        name = name if name != None else str(id)

        axes[0].plot(hist["loss"], label=name + " loss")
        color = axes[0].lines[-1].get_color()
        axes[0].plot(hist["val_loss"], label=name + " val_loss", color=color, linestyle="--")
        axes[1].plot(hist["accuracy" if "accuracy" in hist else "acc"], label=name + " accuracy")
        color = axes[1].lines[-1].get_color()
        axes[1].plot(
            hist["val_accuracy" if "val_accuracy" in hist else "val_acc"],
            label=name + " val_accuracy",
            color=color,
            linestyle="--",
        )
    for ax in axes:
        ax.legend()
        ax.grid()
    fig.tight_layout()
    return fig


if __name__ == "__test__":
    gpus = tf.config.experimental.get_visible_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    strategy = tf.distribute.MirroredStrategy() if len(gpus) > 1 else tf.distribute.OneDeviceStrategy(device="/gpu:0")
    keras.mixed_precision.set_global_policy("mixed_float16")

    from keras_cv_attention_models import imagenet
    from keras_cv_attention_models import model_surgery
    from keras_cv_attention_models import aotnet, coatnet

    input_shape = (224, 224, 3)
    batch_size = 128 * strategy.num_replicas_in_sync
    lr_base_512 = 0.01
    l2_weight_decay = 0
    optimizer_wd_base = 1e-2
    magnitude = 5
    mixup_alpha = 0
    label_smoothing = 0.1
    lr_decay_steps = 30 # [30, 60, 90] for constant decay
    lr_warmup = 4
    basic_save_name = None

    with strategy.scope():
        # mm = keras.applications.ResNet50V2(include_top=False, input_shape=input_shape, weights=None)
        # nn = mm.outputs[0]
        # nn = keras.layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        # nn = keras.layers.Dropout(0.2)(nn)
        # nn = keras.layers.Dense(1000, activation="softmax", dtype="float32", name="predictions")(nn)
        # model = keras.models.Model(mm.inputs[0], nn, name=mm.name)
        model = coatnet.CoAtNet0(num_classes=1000, activation='gelu', drop_connect_rate=0.2, drop_rate=0.2)
        # model = aotnet.AotNet(num_blocks=[3, 4, 6, 3], strides=[1, 2, 2, 2], activation='swish', preact=True, avg_pool_down=True, drop_connect_rate=0.2, drop_rate=0.2, model_name='aotnet50_swish_preact_avg_down_drop02_mixup_0')

        if l2_weight_decay != 0:
            model = model_surgery.add_l2_regularizer_2_model(model, weight_decay=l2_weight_decay, apply_to_batch_normal=False)

        lr_base =  lr_base_512 * batch_size / 512
        # optimizer = keras.optimizers.SGD(learning_rate=lr_base, momentum=0.9)
        optimizer = tfa.optimizers.AdamW(lr=lr_base, weight_decay=lr_base * optimizer_wd_base)
        model.compile(optimizer=optimizer, loss=keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing), metrics=["acc"])
        if isinstance(lr_decay_steps, list):
            constant_lr_sch = lambda epoch: imagenet.constant_scheduler(epoch, lr_base=lr_base, lr_decay_steps=lr_decay_steps, warmup=lr_warmup)
            lr_scheduler = keras.callbacks.LearningRateScheduler(constant_lr_sch)
            epoch = lr_decay_steps[-1] +lr_decay_steps[0] + lr_warmup   # 124 for lr_decay_steps=[30, 60, 90], lr_warmup=4
        else:
            lr_scheduler = imagenet.CosineLrScheduler(lr_base, first_restart_step=lr_decay_steps, m_mul=0.5, t_mul=2.0, lr_min=1e-05, warmup=lr_warmup, steps_per_epoch=-1)
            epoch = lr_decay_steps * 3 + lr_warmup  # 94 for lr_decay_steps=30, lr_warmup=4

        imagenet.train(
            model,
            epochs=epoch,
            initial_epoch=0,
            lr_scheduler=lr_scheduler,
            input_shape=input_shape,
            batch_size=batch_size,
            magnitude=magnitude,
            mixup_alpha=mixup_alpha,
            basic_save_name=basic_save_name,
        )
