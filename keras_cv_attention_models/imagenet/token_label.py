import tensorflow as tf
from tensorflow import keras


class TokenLabelAlign:
    """
    >>> dd = TokenLabelAlign(target_num_pathes=7)
    >>> print(f"{np.allclose(tf.math.top_k(dd(cc[0]), 5)[0], cc[0][1]) = }")
    >>> print(f"{np.allclose(tf.math.top_k(dd(cc[0]), 5)[1], cc[0][0]) = }")
    """

    def __init__(self, num_classes=10, target_num_pathes=14, align_method="bilinear"):
        self.num_classes, self.align_method = num_classes, align_method
        target_num_pathes = target_num_pathes[:2] if isinstance(target_num_pathes, (list, tuple)) else (target_num_pathes, target_num_pathes)
        self.target_num_pathes_h, self.target_num_pathes_w = target_num_pathes
        self.built = False

    def build(self, token_label_shape):
        # To one-hot
        self.source_num_pathes_h, self.source_num_pathes_w, num_topk = token_label_shape[1], token_label_shape[2], token_label_shape[3]
        self.target_num_pathes_h = self.target_num_pathes_h if self.target_num_pathes_h > 0 else self.source_num_pathes_h
        self.target_num_pathes_w = self.target_num_pathes_w if self.target_num_pathes_w > 0 else self.source_num_pathes_w
        hh, ww = tf.meshgrid(range(self.source_num_pathes_h), range(self.source_num_pathes_w), indexing="ij")
        hhww = tf.concat([tf.reshape(hh, [-1, 1, 1]), tf.reshape(ww, [-1, 1, 1])], axis=-1)
        self.one_hot_hhww = tf.repeat(hhww, num_topk, axis=1)

        # Align to target shape
        hh, ww = tf.meshgrid(range(0, self.target_num_pathes_h), range(0, self.target_num_pathes_w), indexing="ij")
        hh, ww = tf.reshape(hh, [-1, 1]), tf.reshape(ww, [-1, 1])
        boxes = tf.concat([hh, ww, hh + 1, ww + 1], axis=-1)
        self.boxes = tf.cast(boxes, "float32") / [self.target_num_pathes_h, self.target_num_pathes_w, self.target_num_pathes_h, self.target_num_pathes_w]
        self.box_indices = [0] * (self.target_num_pathes_h * self.target_num_pathes_w)  # 0 is indicating which batch
        # self.need_align = self.target_num_pathes_h != self.source_num_pathes_h or self.target_num_pathes_w != self.source_num_pathes_w
        self.built = True

        source_num_pathes, target_num_pathes = [self.source_num_pathes_h, self.source_num_pathes_w], [self.target_num_pathes_h, self.target_num_pathes_w]
        print(">>>> [TokenLabelAlign] source_num_pathes: {}, target_num_pathes: {}".format(source_num_pathes, target_num_pathes))

    def __call__(self, token_label, flip_left_right=False, scale_hh=1, scale_ww=1, crop_hh=0, crop_ww=0):
        if not self.built:
            self.build(token_label.shape)
        label_pos, label_score = tf.cast(token_label[0], "int32"), tf.cast(token_label[1], "float32")
        label_position = tf.concat([tf.reshape(self.one_hot_hhww, [-1, 2]), tf.reshape(label_pos, [-1, 1])], axis=-1)
        token_label_one_hot = tf.zeros([self.source_num_pathes_h, self.source_num_pathes_w, self.num_classes])
        token_label_one_hot = tf.tensor_scatter_nd_update(token_label_one_hot, label_position, tf.reshape(label_score, [-1]))

        token_label_one_hot = tf.cond(flip_left_right, lambda: tf.image.flip_left_right(token_label_one_hot), lambda: token_label_one_hot)
        boxes = (self.boxes + [crop_hh, crop_ww, crop_hh, crop_ww]) / [scale_hh, scale_ww, scale_hh, scale_ww]
        # if self.need_align:
        token_label_one_hot = tf.expand_dims(token_label_one_hot, 0)  # Expand a batch dimension, required by crop_and_resize
        token_label_one_hot = tf.image.crop_and_resize(token_label_one_hot, boxes, self.box_indices, crop_size=(1, 1), method=self.align_method)
        return tf.reshape(token_label_one_hot, (self.target_num_pathes_h, self.target_num_pathes_w, self.num_classes))


def build_token_label_file(
    data_name, model, input_shape=-1, batch_size=16, rescale_mode="auto", resize_method="bicubic", resize_antialias=True, token_label_top_k=5, save_path=None
):
    import pickle
    import numpy as np
    from tqdm import tqdm
    from keras_cv_attention_models.imagenet import data, train_func
    from keras_cv_attention_models import model_surgery

    total_images, num_classes, steps_per_epoch, num_channels = data.init_dataset(data_name, batch_size=batch_size, info_only=True)
    if isinstance(model, str) and model.endswith(".h5"):
        model = keras.models.load_model(model, compile=False)
    else:
        model = train_func.init_model(model, input_shape, num_classes)
    token_label_model = model_surgery.convert_to_token_label_model(model)

    if isinstance(rescale_mode, str) and rescale_mode.lower() == "auto":
        rescale_mode = getattr(model, "rescale_mode", "torch")
        print(">>>> rescale_mode:", rescale_mode)

    train_dataset = data.init_dataset(
        data_name,
        input_shape=model.input_shape[1:],
        batch_size=batch_size,
        rescale_mode=rescale_mode,
        resize_method=resize_method,
        resize_antialias=resize_antialias,
        magnitude=-1,  # Disable random_flip_left_right
        use_shuffle=False,  # Disable shuffle
    )[0]
    if save_path is None:
        data_name = data_name.replace("/", "_")
        patches = token_label_model.output_shape[1][1:-1]
        save_path = "{}_{}_{}_{}_{}.pkl".format(data_name, model.name, patches[0], patches[1], token_label_top_k)

    rrs = []
    need_softmax = False if token_label_model.layers[-1].activation.__name__ == "softmax" else True
    for image_batch, _ in tqdm(train_dataset):
        predictions = token_label_model(image_batch)[-1]
        if need_softmax:
            predictions = tf.nn.softmax(predictions, axis=-1)
        prediction_scores, prediction_ids = tf.math.top_k(predictions, k=token_label_top_k)
        rr = tf.stack([tf.cast(prediction_ids, prediction_scores.dtype), prediction_scores], axis=1)
        rrs.append(rr.numpy())
    rrs = np.concatenate(rrs, axis=0)
    # np.save(label_token_file, rrs)
    with open(save_path, "wb") as ff:
        pickle.dump(rrs, ff)
    print(">>>> Saved to:", save_path)
