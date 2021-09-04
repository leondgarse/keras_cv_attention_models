import os
import tensorflow as tf
from tensorflow import keras
from keras_cv_attention_models.imagenet import data
from keras_cv_attention_models.imagenet import callbacks
from keras_cv_attention_models import model_surgery


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
        basic_save_name = "{}_imagenet_batch_size_{}_magnitude_{}_".format(compiled_model.name, batch_size, magnitude)
    ckpt_path = os.path.join("checkpoints", basic_save_name + "epoch_{epoch:02d}_val_acc_{val_acc:.2f}.h5")
    cur_callbacks = [keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True)]
    hist_file = os.path.join("checkpoints", basic_save_name + "hist.json")
    if initial_epoch == 0 and os.path.exists(hist_file):
        os.remove(hist_file)
    cur_callbacks.append(callbacks.MyHistory(initial_file=hist_file))
    cur_callbacks.append(keras.callbacks.TerminateOnNaN())
    if lr_scheduler is not None:
        cur_callbacks.append(lr_scheduler)

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

    input_shape, batch_size, l2_weight_decay, magnitude, mixup_alpha = (224, 224, 3), 128, 5e-5, 5, 0
    with strategy.scope():
        mm = keras.applications.ResNet50V2(include_top=False, input_shape=input_shape, weights=None)
        nn = mm.outputs[0]
        nn = keras.layers.GlobalAveragePooling2D(name="avg_pool")(nn)
        nn = keras.layers.Dropout(0.2)(nn)
        nn = keras.layers.Dense(1000, activation="softmax", dtype="float32", name="predictions")(nn)
        model = keras.models.Model(mm.inputs[0], nn, name=mm.name)

        if l2_weight_decay != 0:
            model = model_surgery.add_l2_regularizer_2_model(model, weight_decay=l2_weight_decay, apply_to_batch_normal=False)

        optimizer = keras.optimizers.SGD(learning_rate=0.05, momentum=0.9)
        model.compile(optimizer=optimizer, loss=keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=["acc"])
        lr_scheduler = imagenet.CosineLrScheduler(0.05, first_restart_step=16, m_mul=0.5, t_mul=2.0, lr_min=1e-05, warmup=2, steps_per_epoch=-1)
        imagenet.train(
            model,
            epochs=16 + 32 + 2,
            lr_scheduler=lr_scheduler,
            input_shape=input_shape,
            batch_size=batch_size,
            magnitude=magnitude,
            mixup_alpha=mixup_alpha,
        )
