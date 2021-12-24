import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_datasets as tfds
from tensorflow import keras
from tensorflow.keras.callbacks import LearningRateScheduler


class RandomProcessImage:
    def __init__(self, target_shape=(300, 300), magnitude=0, keep_shape=False):
        self.target_shape, self.magnitude, self.keep_shape = target_shape, magnitude, keep_shape
        self.target_shape = target_shape if len(target_shape) == 2 else target_shape[:2]
        if magnitude > 0:
            from keras_cv_attention_models.imagenet import augment

            translate_const, cutout_const = 100, 40
            # translate_const = int(target_shape[0] * 10 / magnitude)
            # cutout_const = int(target_shape[0] * 40 / 224)
            print(">>>> RandAugment: magnitude = %d, translate_const = %d, cutout_const = %d" % (magnitude, translate_const, cutout_const))
            aa = augment.RandAugment(magnitude=magnitude, translate_const=translate_const, cutout_const=cutout_const)
            # aa.available_ops = ["AutoContrast", "Equalize", "Invert", "Rotate", "Posterize", "Solarize", "Color", "Contrast", "Brightness", "Sharpness", "ShearX", "ShearY", "TranslateX", "TranslateY", "Cutout", "SolarizeAdd"]
            self.process = lambda img: aa.distort(img)
        elif magnitude == 0:
            self.process = lambda img: tf.image.random_flip_left_right(img)
        else:
            self.process = lambda img: img

    def __call__(self, datapoint):
        image = datapoint["image"]
        if self.keep_shape:
            cropped_shape = tf.reduce_min(tf.keras.backend.shape(image)[:2])
            image = tf.image.random_crop(image, (cropped_shape, cropped_shape, 3))

        input_image = tf.image.resize(image, self.target_shape)
        label = datapoint["label"]
        input_image = self.process(input_image)
        input_image = (tf.cast(input_image, tf.float32) - 127.5) / 128
        return input_image, label


def init_dataset(
    data_name="food101",
    target_shape=(300, 300),
    batch_size=64,
    buffer_size=1000,
    info_only=False,
    magnitude=0,
    keep_shape=False,
):
    dataset, info = tfds.load(data_name, with_info=True)
    num_classes = info.features["label"].num_classes
    total_images = info.splits["train"].num_examples
    if info_only:
        return total_images, num_classes

    AUTOTUNE = tf.data.AUTOTUNE
    train_process = RandomProcessImage(target_shape, magnitude, keep_shape=keep_shape)
    train = dataset["train"].map(lambda xx: train_process(xx), num_parallel_calls=AUTOTUNE)
    test_process = RandomProcessImage(target_shape, magnitude=-1, keep_shape=keep_shape)
    if "validation" in dataset:
        test = dataset["validation"].map(lambda xx: test_process(xx))
    elif "test" in dataset:
        test = dataset["test"].map(lambda xx: test_process(xx))

    as_one_hot = lambda xx, yy: (xx, tf.one_hot(yy, num_classes))
    train_dataset = train.shuffle(buffer_size).batch(batch_size).map(as_one_hot).prefetch(buffer_size=AUTOTUNE)
    test_dataset = test.batch(batch_size).map(as_one_hot)
    return train_dataset, test_dataset, total_images, num_classes


def exp_scheduler(epoch, lr_base=0.256, decay_step=2.4, decay_rate=0.97, lr_min=0, warmup=10):
    if epoch < warmup:
        lr = (lr_base - lr_min) * (epoch + 1) / (warmup + 1)
    else:
        lr = lr_base * decay_rate ** ((epoch - warmup) / decay_step)
        lr = lr if lr > lr_min else lr_min
    print("Learning rate for iter {} is {}".format(epoch + 1, lr))
    return lr


def progressive_with_dropout_randaug(
    model,
    data_name="cifar10",
    lr_scheduler=None,
    total_epochs=36,
    batch_size=64,
    target_shapes=[128],
    dropouts=[0.4],
    dropout_layer=-2,
    magnitudes=[0],
):
    if model.compiled_loss is None:
        print(">>>> Error: Model NOT compiled.")
        return None

    histories = []
    stages = min([len(target_shapes), len(dropouts), len(magnitudes)])
    for stage, target_shape, dropout, magnitude in zip(range(stages), target_shapes, dropouts, magnitudes):
        print(">>>> stage: {}/{}, target_shape: {}, dropout: {}, magnitude: {}".format(stage + 1, stages, target_shape, dropout, magnitude))
        if len(dropouts) > 1 and isinstance(model.layers[dropout_layer], keras.layers.Dropout):
            print(">>>> Changing dropout rate to:", dropout)
            model.layers[dropout_layer].rate = dropout
            # loss, optimizer, metrics = model.loss, model.optimizer, model.metrics
            # model = keras.models.clone_model(model)  # Make sure it do changed
            # model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

        target_shape = (target_shape, target_shape)
        train_dataset, test_dataset, total_images, num_classes = init_dataset(
            data_name=data_name, target_shape=target_shape, batch_size=batch_size, magnitude=magnitude, keep_shape=True
        )

        initial_epoch = stage * total_epochs // stages
        epochs = (stage + 1) * total_epochs // stages
        history = model.fit(
            train_dataset,
            epochs=epochs,
            initial_epoch=initial_epoch,
            validation_data=test_dataset,
            callbacks=[lr_scheduler] if lr_scheduler is not None else [],
        )
        histories.append(history)
    hhs = {kk: np.ravel([hh.history[kk] for hh in histories]).astype("float").tolist() for kk in history.history.keys()}
    return hhs


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


if __name__ == "__test_cifar10__":
    import json
    import tensorflow_addons as tfa
    from icecream import ic
    from keras_cv_attention_models import efficientnet
    from keras_cv_attention_models.efficientnet import progressive_train_test

    keras.mixed_precision.set_global_policy("mixed_float16")

    input_shape = (224, 224, 3)
    batch_size = 64
    train_dataset, test_dataset, total_images, num_classes = progressive_train_test.init_dataset(
        data_name="cifar10", target_shape=input_shape, batch_size=batch_size, magnitude=15
    )

    eb2s = efficientnet.EfficientNetV2S(input_shape=(224, 224, 3), num_classes=0)
    out = eb2s.output

    nn = keras.layers.GlobalAveragePooling2D(name="avg_pool")(out)
    nn = keras.layers.Dense(num_classes, activation="softmax", name="predictions")(nn)
    eb2_imagenet = keras.models.Model(eb2s.inputs[0], nn)

    eb2_imagenet.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["acc"])
    history_eb2_imagenet = eb2_imagenet.fit(train_dataset, epochs=15, validation_data=test_dataset)
    with open("history_eb2_imagenet.json", "w") as ff:
        json.dump(history_eb2_imagenet.history, ff)
