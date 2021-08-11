import numpy as np
import tensorflow as tf
from tqdm import tqdm


def extract_token_label_batch(image_batch, teacher_model, top_k=5, return_one_hot=False):
    iaa = tf.stack([tf.image.resize(ii, teacher_model.input_shape[1:3]) for ii in image_batch])  # (0, 255)
    if teacher_model.layers[-1].activation.__name__ == "softmax":
        ipp = teacher_model(iaa).numpy()
    else:
        ipp = tf.nn.softmax(teacher_model(iaa), axis=-1).numpy()

    if return_one_hot:  # if num_classes is not large
        ipp_ids = np.argsort(ipp, axis=-1)[..., :-top_k]
        for ii, jj in zip(ipp.reshape(-1, ipp.shape[-1]), ipp_ids.reshape(-1, ipp_ids.shape[-1])):
            ii[jj] = 0
        return ipp
    else:
        ipp_scores = np.sort(ipp, axis=-1)[..., -top_k:]
        ipp_ids = np.argsort(ipp, axis=-1)[..., -top_k:]
        return np.stack([ipp_ids, ipp_scores], axis=1)


def token_label_preprocess(token_label, num_classes=10, num_pathes=14):
    if token_label.shape[-1] != num_classes:  # To one_hot like
        ipp_ids = tf.cast(tf.reshape(token_label[0], (-1, token_label.shape[-1], 1)), tf.int32)
        ipp_scores = tf.cast(tf.reshape(token_label[1], (-1, token_label.shape[-1])), tf.float32)

        iaa = tf.zeros(num_classes)
        ibb = tf.stack([tf.tensor_scatter_nd_update(iaa, ipp_ids[ii], ipp_scores[ii]) for ii in range(ipp_ids.shape[0])])
        # hhww = token_label.shape[1] * token_label.shape[2]
        # id_indexes = tf.expand_dims(tf.range(hhww), 1) + tf.expand_dims(tf.range(token_label.shape[-1]), 0)
        # indexed_ids = tf.concat([tf.expand_dims(id_indexes, -1), ipp_ids], -1)
        # ibb = tf.zeros([hhww, num_classes])
        # tf.print(ibb.shape, indexed_ids.shape, ipp_scores.shape)
        # ibb = tf.tensor_scatter_nd_update(ibb, indexed_ids, ipp_scores)
        token_label = tf.reshape(ibb, [token_label.shape[1], token_label.shape[2], ibb.shape[-1]])

    cur_patches = token_label.shape[0]
    if num_pathes != cur_patches:
        # token_label = tf.image.resize(token_label, (num_pathes, num_pathes))
        # token_label = tf.gather(token_label, pick_ceil, axis=0)
        # token_label = tf.gather(token_label, pick_ceil, axis=1)
        # pick = np.clip(np.arange(0, cur_patches, cur_patches / num_pathes), 0, cur_patches - 1)
        # pick_floor = np.floor(pick).astype('int')
        # pick_ceil = np.ceil(pick).astype('int')
        # pick_val = tf.reshape(tf.cast(pick - pick_floor, token_label.dtype), [-1, 1, 1])

        # token_label = tf.gather(token_label, pick_floor, axis=0) * pick_val + tf.gather(token_label, pick_ceil, axis=0) * (1 - pick_val)
        # pick_val = tf.transpose(pick_val, [1, 0, 2])
        # token_label = tf.gather(token_label, pick_floor, axis=1) * pick_val + tf.gather(token_label, pick_ceil, axis=1) * (1 - pick_val)
        xx, yy = np.meshgrid(np.arange(0, num_pathes), np.arange(0, num_pathes))
        xx, yy = xx.reshape(-1, 1), yy.reshape(-1, 1)
        boxes = np.concatenate([xx, yy, xx + 1, yy + 1], axis=-1) / num_pathes
        box_indices = [0] * (num_pathes * num_pathes)
        token_label = tf.image.crop_and_resize(tf.expand_dims(token_label, 0), boxes, box_indices, crop_size=(1, 1))
    token_label = tf.reshape(token_label, (num_pathes * num_pathes, token_label.shape[-1]))
    return token_label


def load_cifar10_token_label(label_token_file, num_classes=10, batch_size=1024, image_shape=(32, 32), num_pathes=14):
    import tensorflow_datasets as tfds

    AUTOTUNE = tf.data.experimental.AUTOTUNE

    train_ds = tfds.load("cifar10", split="train")

    token_label_data = np.load(label_token_file)
    token_label_ds = tf.data.Dataset.from_tensor_slices(token_label_data)
    token_label_train_ds = tf.data.Dataset.zip((train_ds, token_label_ds))

    image_preprocess = lambda data: tf.image.resize(data["image"], image_shape[:2]) / 255.0
    label_preprocess = lambda data: tf.one_hot(data["label"], depth=num_classes)

    train_preprocessing = lambda data, token_label: (
        image_preprocess(data),
        (label_preprocess(data), token_label_preprocess(token_label, num_classes, num_pathes)),
    )
    token_label_train_ds = token_label_train_ds.shuffle(buffer_size=batch_size * 100).map(train_preprocessing, num_parallel_calls=AUTOTUNE)
    token_label_train_ds = token_label_train_ds.batch(batch_size).prefetch(buffer_size=AUTOTUNE)

    # Load test dataset
    test_preprocessing = lambda data: (image_preprocess(data), label_preprocess(data))
    test_ds = tfds.load("cifar10", split="test").map(test_preprocessing, num_parallel_calls=AUTOTUNE).batch(batch_size)
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
            pred = cls_pred[:, : aux_pred.shape[-1]] + tf.reduce_max(aux_pred, 1) * 0.5
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
