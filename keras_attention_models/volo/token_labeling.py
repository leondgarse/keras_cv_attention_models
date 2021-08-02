import numpy as np
import tensorflow as tf
from tqdm import tqdm


def extract_token_label(image_batch, teacher_model, top_k=5, return_one_hot=False):
    iaa = tf.stack([tf.image.resize(ii, teacher_model.input_shape[1:3]) for ii in image_batch]) # (0, 255)
    if teacher_model.layers[-1].activation.__name__ == "softmax":
        ipp = teacher_model(iaa).numpy()
    else:
        ipp = tf.nn.softmax(teacher_model(iaa), axis=-1).numpy()

    if return_one_hot:
        ipp_ids = np.argsort(ipp, axis=-1)[..., :-top_k]
        for ii, jj in zip(ipp.reshape(-1, ipp.shape[-1]), ipp_ids.reshape(-1, ipp_ids.shape[-1])):
            ii[jj] = 0
        return ipp
    else:
        ipp_scores = np.sort(ipp, axis=-1)[..., -top_k:]
        ipp_ids = np.argsort(ipp, axis=-1)[..., -top_k:]
        return ipp_ids, ipp_scores


def token_label_preprocessing(data, token_label, image_shape=(224, 224, 3), num_classes=10, num_pathes=14):
    image = tf.image.resize(data["image"], image_shape[:2])
    label = tf.one_hot(data["label"], depth=num_classes)

    cur_patches = token_label.shape[0]
    if num_pathes != cur_patches:
        pick = np.arange(0, cur_patches, cur_patches/num_pathes).astype('int').tolist()
        token_label = tf.gather(token_label, pick, axis=0)
        token_label = tf.gather(token_label, pick, axis=1)
    token_label = tf.reshape(token_label, (num_pathes * num_pathes, token_label.shape[-1]))
    return image, (label, token_label)


def load_cifar10_token_label(label_token_file, num_classes=10, batch_size=1024, image_shape=(32, 32), num_pathes=14):
    import tensorflow_datasets as tfds
    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds = tfds.load("cifar10", split="train")

    token_label_data = np.load(label_token_file)
    token_label_ds = tf.data.Dataset.from_tensor_slices(token_label_data)
    token_label_train_ds = tf.data.Dataset.zip((train_ds, token_label_ds))

    image_preprocessing = lambda img: tf.image.resize(img, image_shape[:2]) / 255.0
    cur_token_label_preprocessing = lambda data, token_label: token_label_preprocessing(data, token_label, image_shape, num_classes, num_pathes)
    token_label_train_ds = token_label_train_ds.shuffle(buffer_size=batch_size * 100).map(cur_token_label_preprocessing, num_parallel_calls=AUTOTUNE)
    token_label_train_ds = token_label_train_ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)

    # Load test dataset
    data_preprocessing = lambda data: (image_preprocessing(data["image"]), tf.one_hot(data["label"], depth=num_classes))
    test_ds = tfds.load("cifar10", split="test").map(data_preprocessing, num_parallel_calls=AUTOTUNE).batch(batch_size)
    return token_label_train_ds, test_ds


def token_label_class_loss(y_true, y_pred):
    # tf.print(", y_true:", y_true.shape, "y_pred:", y_pred.shape, end="")
    if y_pred.shape[-1] != y_true.shape[-1]:
        y_pred, cls_lambda = y_pred[:, :-1], y_pred[:, -1:]
        y_true = tf.cast(y_true, y_pred.dtype)
        y_true = cls_lambda * y_true + (1 - cls_lambda) * y_true[::-1]
    return keras.losses.categorical_crossentropy(y_true, y_pred, from_logits=True)


class TokenLabelEval(keras.callbacks.Callback):
    def __init__(self, test_dataset, **kwargs):
        super(TokenLabelEval, self).__init__(**kwargs)
        self.test_dataset = test_dataset
        self.hist = {}

    def on_epoch_end(self, epoch, logs=None):
        preds, labels = [], []
        for image, label in self.test_dataset:
            cls_pred, aux_pred = self.model.predict(image)
            pred = cls_pred[:, :aux_pred.shape[-1]] + tf.reduce_max(aux_pred, 1) * 0.5
            preds.extend(pred.numpy())
            labels.extend(label.numpy())
        preds, labels = np.stack(preds), np.stack(labels)
        preds = tf.nn.softmax(preds).numpy()

        val_loss = tf.reduce_mean(keras.losses.categorical_crossentropy(labels, preds)).numpy()
        val_acc = (preds.argmax(-1) == labels.argmax(-1)).sum() / preds.shape[0]
        tf.print("- val_loss: {:.4f} - val_acc: {:.4f}".format(val_loss, val_acc))
        self.hist.setdefault("val_loss", []).append(val_loss)
        self.hist.setdefault("val_acc", []).append(val_acc)
        # self.model.history.history.update(self.hist)
