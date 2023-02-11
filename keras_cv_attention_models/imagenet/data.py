import tensorflow as tf
from tensorflow import keras


def init_mean_std_by_rescale_mode(rescale_mode):
    if isinstance(rescale_mode, (list, tuple)):  # Specific mean and std
        mean, std = rescale_mode
    elif rescale_mode == "torch":
        mean = tf.constant([0.485, 0.456, 0.406]) * 255.0
        std = tf.constant([0.229, 0.224, 0.225]) * 255.0
    elif rescale_mode == "tf":  # [0, 255] -> [-1, 1]
        mean, std = 127.5, 127.5
        # mean, std = 127.5, 128.0
    elif rescale_mode == "tf128":  # [0, 255] -> [-1, 1]
        mean, std = 128.0, 128.0
    elif rescale_mode == "raw01":
        mean, std = 0, 255.0  # [0, 255] -> [0, 1]
    else:
        mean, std = 0, 1  # raw inputs [0, 255]
    return mean, std


def tf_imread(file_path):
    # tf.print('Reading file:', file_path)
    img = tf.io.read_file(file_path)
    # img = tf.image.decode_jpeg(img, channels=3)  # [0, 255]
    img = tf.image.decode_image(img, channels=3, expand_animations=False)  # [0, 255]
    img = tf.cast(img, "float32")  # [0, 255]
    return img


def random_crop_fraction(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333), log_distribute=True, compute_dtype="float32"):
    height, width = tf.cast(size[0], dtype=compute_dtype), tf.cast(size[1], dtype=compute_dtype)
    area = height * width
    scale_max = tf.minimum(tf.minimum(height * height * ratio[1] / area, width * width / ratio[0] / area), scale[1])
    target_area = tf.random.uniform((), scale[0], scale_max, dtype=compute_dtype) * area

    ratio_min = tf.maximum(target_area / (height * height), ratio[0])
    ratio_max = tf.minimum(width * width / target_area, ratio[1])
    if log_distribute:  # More likely to select a smaller value
        log_min, log_max = tf.math.log(ratio_min), tf.math.log(ratio_max)
        aspect_ratio = tf.random.uniform((), log_min, log_max, dtype=compute_dtype)
        aspect_ratio = tf.math.exp(aspect_ratio)
    else:
        aspect_ratio = tf.random.uniform((), ratio_min, ratio_max, dtype=compute_dtype)

    ww_crop = tf.cast(tf.math.floor(tf.sqrt(target_area * aspect_ratio)), "int32")
    hh_crop = tf.cast(tf.math.floor(tf.sqrt(target_area / aspect_ratio)), "int32")
    # tf.print(">>>> height, width, hh_crop, ww_crop, hh_fraction_min, hh_fraction_max:", height, width, hh_crop, ww_crop, hh_fraction_min, hh_fraction_max)
    # return hh_crop, ww_crop, target_area, hh_fraction_min, hh_fraction_max, hh_fraction
    return hh_crop, ww_crop
    # return hh_fraction, target_area / hh_fraction # float value will stay in scale and ratio range exactly


def random_crop_and_resize_image(image, target_shape, scale=(0.08, 1.0), ratio=(0.75, 1.3333333), method="bilinear", antialias=False):
    """Random crop and resize, return `image, scale_hh, scale_ww, crop_hh, crop_ww`"""
    height, width = tf.shape(image)[0], tf.shape(image)[1]
    cropped_hh, cropped_ww = random_crop_fraction((height, width), scale, ratio)
    cropped_hh, cropped_ww = tf.clip_by_value(cropped_hh, 1, height - 1), tf.clip_by_value(cropped_ww, 1, width - 1)
    crop_hh = tf.random.uniform((), 0, height - cropped_hh, dtype=cropped_hh.dtype)
    crop_ww = tf.random.uniform((), 0, width - cropped_ww, dtype=cropped_ww.dtype)
    image = image[crop_hh : crop_hh + cropped_hh, crop_ww : crop_ww + cropped_ww]
    image = tf.image.resize(image, target_shape, method=method, antialias=antialias)

    # crop_hh, crop_ww is crop size after image size applyied scale.
    # image size firstly rescale with (scale_hh, scale_ww), then crop as [crop_hh: crop_hh + target_shape, crop_ww: crop_ww + target_shape]
    # scale_hh = tf.cast(target_shape[0], "float32") / tf.cast(cropped_hh, "float32")
    # scale_ww = tf.cast(target_shape[1], "float32") / tf.cast(cropped_ww, "float32")
    # crop_hh = tf.cast(tf.cast(crop_hh, "float32") * scale_hh, "int32")
    # crop_ww = tf.cast(tf.cast(crop_ww, "float32") * scale_ww, "int32")

    # For coords or bbox value in (0, 1) -> * scale - offset
    # (bbox * height * scale_hh - crop_hh) / target_shape[0]
    scale_hh = tf.cast(height, "float32") / tf.cast(cropped_hh, "float32")
    scale_ww = tf.cast(width, "float32") / tf.cast(cropped_ww, "float32")
    crop_hh = tf.cast(crop_hh, "float32") / tf.cast(cropped_hh, "float32")
    crop_ww = tf.cast(crop_ww, "float32") / tf.cast(cropped_ww, "float32")

    return image, scale_hh, scale_ww, crop_hh, crop_ww


def random_erasing_per_pixel(image, num_layers=1, scale=(0.02, 0.33333333), ratio=(0.3, 3.3333333), probability=0.5):
    """https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/random_erasing.py"""
    if tf.random.uniform(()) > probability:
        return image

    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.120003, 57.375]
    height, width, _ = image.shape
    for _ in range(num_layers):
        hh, ww = random_crop_fraction((height, width), scale=scale, ratio=ratio)
        hh_ss = tf.random.uniform((), 0, height - hh, dtype="int32")
        ww_ss = tf.random.uniform((), 0, width - ww, dtype="int32")
        mask = tf.random.normal([hh, ww, 3], mean=mean, stddev=std)
        mask = tf.clip_by_value(mask, 0.0, 255.0)  # value in [0, 255]
        aa = tf.concat([image[:hh_ss, ww_ss : ww_ss + ww], mask, image[hh_ss + hh :, ww_ss : ww_ss + ww]], axis=0)
        image = tf.concat([image[:, :ww_ss], aa, image[:, ww_ss + ww :]], axis=1)
    return image


def sample_beta_distribution(shape, concentration_0=0.4, concentration_1=0.4):
    gamma_1_sample = tf.random.gamma(shape=shape, alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=shape, alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


def mixup(images, labels, token_label=None, alpha=0.4, min_mix_weight=0):
    # mix_weight = tfp.distributions.Beta(alpha, alpha).sample([batch_size, 1])
    batch_size = tf.shape(images)[0]
    mix_weight = sample_beta_distribution([batch_size], alpha, alpha)
    mix_weight = tf.maximum(mix_weight, 1.0 - mix_weight)

    # For min_mix_weight=0.1, regard values with `> 0.9` as no mixup, this probability is near `1 - alpha`
    # alpha: no_mixup --> {0.1: 0.8128, 0.2: 0.6736, 0.4: 0.4793, 0.6: 0.3521, 0.8: 0.2636, 1.0: 0.2000}
    if min_mix_weight > 0:
        mix_weight = tf.where(mix_weight > 1 - min_mix_weight, tf.ones_like(mix_weight), mix_weight)

    label_mix_weight = tf.cast(tf.expand_dims(mix_weight, -1), "float32")
    img_mix_weight = tf.cast(tf.reshape(mix_weight, [batch_size, 1, 1, 1]), images.dtype)

    labels = tf.cast(labels, "float32")
    # images = images * img_mix_weight + images[::-1] * (1.0 - img_mix_weight)
    # labels = labels * label_mix_weight + labels[::-1] * (1 - label_mix_weight)
    shuffle_index = tf.random.shuffle(tf.range(batch_size))
    images = images * img_mix_weight + tf.gather(images, shuffle_index) * (1.0 - img_mix_weight)
    labels = labels * label_mix_weight + tf.gather(labels, shuffle_index) * (1 - label_mix_weight)
    if token_label is None:
        return images, labels
    else:
        # token_label shape `[batch, path_height, patch_width, one_hot_labels]`
        token_label = token_label * img_mix_weight + tf.gather(token_label, shuffle_index) * (1 - img_mix_weight)
        return images, labels, token_label


def get_box(mix_weight, height, width, dtype="int32"):
    cut_rate_half = tf.math.sqrt(1.0 - mix_weight) / 2
    cut_h_half, cut_w_half = tf.cast(cut_rate_half * float(height), dtype), tf.cast(cut_rate_half * float(width), dtype)
    cut_h_half, cut_w_half = tf.maximum(cut_h_half, 1), tf.maximum(cut_w_half, 1)
    # center_y = tf.random.uniform((), minval=cut_h_half, maxval=height - cut_h_half, dtype=dtype)
    # center_x = tf.random.uniform((), minval=cut_w_half, maxval=width - cut_w_half, dtype=dtype)
    # return center_y - cut_h_half, center_x - cut_w_half, cut_h_half * 2, cut_w_half * 2

    # Can be non-square on border
    center_y = tf.random.uniform((), minval=0, maxval=height, dtype=dtype)
    center_x = tf.random.uniform((), minval=0, maxval=width, dtype=dtype)
    yl = tf.clip_by_value(center_y - cut_h_half, 0, height)
    yr = tf.clip_by_value(center_y + cut_h_half, 0, height)
    xl = tf.clip_by_value(center_x - cut_w_half, 0, width)
    xr = tf.clip_by_value(center_x + cut_w_half, 0, width)
    return yl, xl, yr - yl, xr - xl


def cutmix(images, labels, token_label=None, alpha=0.5, min_mix_weight=0):
    # Get a sample from the Beta distribution
    batch_size = tf.shape(images)[0]
    _, hh, ww, _ = images.shape
    mix_weight = sample_beta_distribution((), alpha, alpha)  # same value in batch
    if token_label is None:
        offset_height, offset_width, target_height, target_width = get_box(mix_weight, hh, ww)
        mix_weight = 1.0 - tf.cast(target_height * target_width, "float32") / tf.cast(hh * ww, "float32")
    else:
        # token_label shape `[batch, path_height, patch_width, one_hot_labels]`
        # Limit box within patchs
        _, path_height_int, patch_width_int, _ = token_label.shape
        path_height, patch_width = float(path_height_int), float(patch_width_int)
        tl_offset_height, tl_offset_width, tl_target_height, tl_target_width = get_box(mix_weight, path_height, patch_width, dtype="float32")
        offset_height, offset_width = tf.cast(tl_offset_height / path_height * float(hh), "int32"), tf.cast(tl_target_width / patch_width * float(ww), "int32")
        target_height, target_width = tf.cast(tl_target_height / path_height * float(hh), "int32"), tf.cast(tl_target_width / patch_width * float(ww), "int32")
        tl_offset_height, tl_offset_width = tf.cast(tl_offset_height, "int32"), tf.cast(tl_offset_width, "int32")
        tl_target_height, tl_target_width = tf.cast(tl_target_height, "int32"), tf.cast(tl_target_width, "int32")
        target_height, target_width = tf.clip_by_value(target_height, 0, hh - offset_height), tf.clip_by_value(target_width, 0, ww - offset_width)

    if mix_weight < min_mix_weight or 1 - mix_weight < min_mix_weight:
        # For input_shape=224, min_mix_weight=0.1, min_height = 224 * sqrt(0.1) = 70.835
        return (images, labels) if token_label is None else (images, labels, token_label)

    crops = tf.image.crop_to_bounding_box(images, offset_height, offset_width, target_height, target_width)
    pad_crops = tf.image.pad_to_bounding_box(crops, offset_height, offset_width, hh, ww)

    labels = tf.cast(labels, "float32")
    # images = images - pad_crops + pad_crops[::-1]
    # labels = labels * mix_weight + labels[::-1] * (1.0 - mix_weight)
    shuffle_index = tf.random.shuffle(tf.range(batch_size))
    images = images - pad_crops + tf.gather(pad_crops, shuffle_index)
    labels = labels * mix_weight + tf.gather(labels, shuffle_index) * (1.0 - mix_weight)
    if token_label is None:
        return images, labels
    else:
        # token_label shape `[batch, path_height, patch_width, one_hot_labels]`
        # tf.print((path_height_int, patch_width_int), (tl_offset_height, tl_offset_width), (tl_target_height, tl_target_width))
        token_label_crops = tf.image.crop_to_bounding_box(token_label, tl_offset_height, tl_offset_width, tl_target_height, tl_target_width)
        token_label_pad_crops = tf.image.pad_to_bounding_box(token_label_crops, tl_offset_height, tl_offset_width, path_height_int, patch_width_int)
        token_label = token_label - token_label_pad_crops + tf.gather(token_label_pad_crops, shuffle_index)
        # token_label = (token_label - token_label_pad_crops) * mix_weight + (tf.gather(token_label_pad_crops, shuffle_index)) * (1.0 - mix_weight)
        return images, labels, token_label


def apply_mixup_cutmix(train_dataset, mixup_alpha, cutmix_alpha, switch_prob=0.5):
    if mixup_alpha > 0 and mixup_alpha <= 1 and cutmix_alpha > 0 and cutmix_alpha <= 1:
        print(">>>> Both mixup_alpha and cutmix_alpha provided: mixup_alpha = {}, cutmix_alpha = {}".format(mixup_alpha, cutmix_alpha))
        mix_func = lambda *args: tf.cond(
            tf.random.uniform(()) > switch_prob,  # switch_prob = 0.5
            lambda: mixup(*args, alpha=mixup_alpha),
            lambda: cutmix(*args, alpha=cutmix_alpha),
        )
    elif mixup_alpha > 0 and mixup_alpha <= 1:
        print(">>>> mixup_alpha provided:", mixup_alpha)
        mix_func = lambda *args: mixup(*args, alpha=mixup_alpha)
    elif cutmix_alpha > 0 and cutmix_alpha <= 1:
        print(">>>> cutmix_alpha provided:", cutmix_alpha)
        mix_func = lambda *args: cutmix(*args, alpha=cutmix_alpha)
    else:
        return train_dataset
    return train_dataset.map(mix_func, num_parallel_calls=tf.data.AUTOTUNE)


class RandomProcessDatapoint:
    def __init__(
        self,
        target_shape=(224, 224),
        central_crop=1.0,
        random_crop_min=1.0,
        resize_method="bilinear",
        resize_antialias=False,
        random_erasing_prob=0.0,
        random_erasing_layers=1,
        magnitude=0,
        num_layers=2,
        use_cutout=False,
        use_relative_translate=True,
        use_color_increasing=True,
        use_positional_related_ops=True,
        use_token_label=False,
        token_label_target_patches=-1,
        num_classes=1000,
        **randaug_kwargs,
    ):
        self.magnitude, self.random_erasing_prob, self.use_token_label = magnitude, random_erasing_prob, use_token_label
        self.target_shape = target_shape if len(target_shape) == 2 else target_shape[:2]
        self.central_crop, self.random_crop_min = central_crop, random_crop_min
        self.resize_method, self.resize_antialias = resize_method, resize_antialias

        if random_erasing_prob > 0:
            self.random_erasing = lambda img: random_erasing_per_pixel(img, num_layers=random_erasing_layers, probability=random_erasing_prob)
            use_cutout = False

        if magnitude > 0:
            from keras_cv_attention_models.imagenet import augment

            # for target_shape = 224, translate_const = 100 and cutout_const = 40
            translate_const = 0.45 if use_relative_translate else min(self.target_shape) * 0.45
            cutout_const = min(self.target_shape) * 0.18
            print(">>>> RandAugment: magnitude = %d, translate_const = %f, cutout_const = %f" % (magnitude, translate_const, cutout_const))

            self.randaug = augment.RandAugment(
                num_layers=num_layers,
                magnitude=magnitude,
                translate_const=translate_const,
                cutout_const=cutout_const,
                use_cutout=use_cutout,
                use_relative_translate=use_relative_translate,
                use_color_increasing=use_color_increasing,
                use_positional_related_ops=use_positional_related_ops,
                **randaug_kwargs,
            )

        if use_token_label:
            from keras_cv_attention_models.imagenet import token_label

            self.token_label_align = token_label.TokenLabelAlign(num_classes=num_classes, target_num_pathes=token_label_target_patches)

    def __call__(self, datapoint, token_label=None):
        image = datapoint["image"]
        if len(image.shape) < 2:
            image = tf_imread(image)
        channel = image.shape[-1]

        flip_left_right, scale_hh, scale_ww, crop_hh, crop_ww = tf.cast(False, tf.bool), 1, 1, 0, 0  # Init value
        if self.random_crop_min > 0 and self.random_crop_min < 1:
            image, scale_hh, scale_ww, crop_hh, crop_ww = random_crop_and_resize_image(
                image, self.target_shape, scale=(self.random_crop_min, 1.0), method=self.resize_method, antialias=self.resize_antialias
            )
        # elif self.central_crop > 0:
        #     image = tf.image.central_crop(image, self.central_crop)
        else:
            image = tf.image.resize(image, self.target_shape, method=self.resize_method, antialias=self.resize_antialias)

        if self.magnitude >= 0:
            # tf.image.random_flip_left_right
            flip_left_right = tf.random.uniform(()) < 0.5
            image = tf.cond(flip_left_right, lambda: tf.image.flip_left_right(image), lambda: image)
        if self.magnitude > 0:
            image = self.randaug(image)
        if self.random_erasing_prob > 0:
            image = self.random_erasing(image)

        image = tf.cast(image, tf.float32)
        image.set_shape([*self.target_shape[:2], channel])

        label = datapoint["label"]
        if self.use_token_label and token_label is not None:
            token_label = self.token_label_align(token_label, flip_left_right, scale_hh, scale_ww, crop_hh, crop_ww)
            return image, label, token_label
        else:
            return image, label


def evaluation_process_crop_resize(datapoint, target_shape=(224, 224), central_crop=1.0, resize_method="bilinear", antialias=False):
    image = datapoint["image"]
    if len(image.shape) < 3:
        image = tf_imread(image)
    if central_crop > 0:  # Do not crop if central_crop == -1
        shape = tf.shape(image)
        height, width = shape[0], shape[1]
        crop_size = tf.cast((central_crop * tf.cast(tf.minimum(height, width), tf.float32)), tf.int32)
        y, x = (height - crop_size) // 2, (width - crop_size) // 2
        image = tf.image.crop_to_bounding_box(image, y, x, crop_size, crop_size)
    image = tf.image.resize(image, target_shape, method=resize_method, antialias=antialias)
    label = datapoint["label"]
    return image, label


# Not using
def evaluation_process_resize_crop(datapoint, target_shape=(224, 224), central_crop=1.0, resize_method="bilinear", antialias=False):
    image = datapoint["image"]
    if len(image.shape) < 3:
        image = tf_imread(image)
    shape = tf.shape(image)
    height, width = shape[0], shape[1]
    min_border = tf.cast(tf.minimum(height, width), tf.float32)
    scale_size = tf.cast(tf.minimum(*target_shape), tf.float32) / central_crop
    hh_scale = tf.cast(tf.floor(tf.cast(height, tf.float32) * scale_size / min_border), tf.int32)
    ww_scale = tf.cast(tf.floor(tf.cast(width, tf.float32) * scale_size / min_border), tf.int32)
    image = tf.image.resize(image, (hh_scale, ww_scale), method=resize_method, antialias=antialias)

    y, x = (hh_scale - target_shape[0]) // 2, (ww_scale - target_shape[1]) // 2
    image = tf.image.crop_to_bounding_box(image, y, x, target_shape[0], target_shape[1])

    label = datapoint["label"]
    return image, label


def recognition_dataset_from_custom_json(data_path, with_info=False):
    import json

    with open(data_path, "r") as ff:
        aa = json.load(ff)

    test_key = "validation" if "validation" in aa else "test"
    train, test, info = aa["train"], aa[test_key], aa["info"]
    total_images, num_classes = len(train), info["num_classes"]
    num_channels = tf_imread(aa["train"][0]["image"]).shape[-1]

    output_signature = {"image": tf.TensorSpec(shape=(), dtype=tf.string), "label": tf.TensorSpec(shape=(), dtype=tf.int64)}
    train_ds = tf.data.Dataset.from_generator(lambda: (ii for ii in train), output_signature=output_signature)
    test_ds = tf.data.Dataset.from_generator(lambda: (ii for ii in test), output_signature=output_signature)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    train_ds = train_ds.apply(tf.data.experimental.assert_cardinality(len(train))).with_options(options)
    test_ds = test_ds.apply(tf.data.experimental.assert_cardinality(len(test))).with_options(options)
    dataset = {"train": train_ds, test_key: test_ds}
    return (dataset, total_images, num_classes, num_channels) if with_info else dataset


def build_token_label_dataset(train_dataset, token_label_file):
    import pickle

    with open(token_label_file, "rb") as ff:
        token_label_data = pickle.load(ff)
    token_label_ds = tf.data.Dataset.from_tensor_slices(token_label_data)
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    token_label_ds = token_label_ds.with_options(options)

    token_label_train_ds = tf.data.Dataset.zip((train_dataset, token_label_ds))
    return token_label_train_ds


def build_distillation_dataset(ds, teacher_model, input_shape, resize_method="bilinear", resize_antialias=False):
    teacher_model.trainable = False
    # Using teacher_model outputs instead of actual labels
    # Using `ds = ds.map(lambda xx, yy: (xx, teacher_model(xx)))` will run teacher_model on CPU, though will on GPU if XLA enabled

    # print(f">>>> {ds.element_spec[0].shape = }, {input_shape = }")
    # gen_func = lambda: ((tf.image.resize(xx, input_shape[:2], method=resize_method, antialias=resize_antialias), (yy, teacher_model(xx))) for xx, yy in ds)
    # image_signature = tf.TensorSpec(shape=(None, input_shape[0], input_shape[1], ds.element_spec[0].shape[-1]), dtype=tf.float32)
    # output_signature = (image_signature, (ds.element_spec[1], ds.element_spec[1]))
    # new_ds = tf.data.Dataset.from_generator(gen_func, output_signature=output_signature)

    output_signature = (ds.element_spec[0], (ds.element_spec[1], ds.element_spec[1]))
    new_ds = tf.data.Dataset.from_generator(lambda: ((xx, (yy, teacher_model(xx))) for xx, yy in ds), output_signature=output_signature)
    if ds.element_spec[0].shape[1] != input_shape[0] or ds.element_spec[0].shape[2] != input_shape[1]:
        resize_func = lambda xx, yy: (tf.image.resize(xx, input_shape[:2], method=resize_method, antialias=resize_antialias), yy)
        new_ds = new_ds.map(resize_func, num_parallel_calls=tf.data.AUTOTUNE)

    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    new_ds = new_ds.apply(tf.data.experimental.assert_cardinality(len(ds))).with_options(options)
    return new_ds


def init_dataset(
    data_name="imagenet2012",  # dataset params
    input_shape=(224, 224),
    batch_size=64,
    buffer_size=1000,
    info_only=False,
    mixup_alpha=0,  # mixup / cutmix params
    cutmix_alpha=0,
    rescale_mode="tf",  # rescale mode, ["tf", "torch"], or specific `(mean, std)` like `(128.0, 128.0)`
    eval_central_crop=1.0,  # augment params
    random_crop_min=1.0,
    resize_method="bilinear",  # ["bilinear", "bicubic"]
    resize_antialias=False,
    random_erasing_prob=0.0,
    magnitude=0,
    num_layers=2,
    use_positional_related_ops=True,
    use_shuffle=True,
    seed=None,
    token_label_file=None,
    token_label_target_patches=-1,
    teacher_model=None,
    teacher_model_input_shape=-1,  # -1 means same with input_shape
    **augment_kwargs,  # Too many...
):
    import tensorflow_datasets as tfds

    # print(">>>> Dataset args:", locals())
    is_tpu = True if len(tf.config.list_logical_devices("TPU")) > 0 else False  # Set True for try_gcs and drop_remainder
    use_token_label = False if token_label_file is None else True
    use_distill = False if teacher_model is None else True
    teacher_model_input_shape = input_shape if teacher_model_input_shape == -1 else teacher_model_input_shape

    if data_name.endswith(".json"):
        dataset, total_images, num_classes, num_channels = recognition_dataset_from_custom_json(data_name, with_info=True)
    else:
        dataset, info = tfds.load(data_name, with_info=True, try_gcs=is_tpu)
        num_classes = info.features["label"].num_classes
        num_channels = info.features["image"].shape[-1]
        total_images = info.splits["train"].num_examples
    steps_per_epoch = int(tf.math.ceil(total_images / float(batch_size)))

    if info_only:
        return total_images, num_classes, steps_per_epoch, num_channels  # return num_channels in case it's not 3

    """ Train dataset """
    train_dataset = dataset["train"]
    if use_token_label:
        train_dataset = build_token_label_dataset(train_dataset, token_label_file)

    AUTOTUNE = tf.data.AUTOTUNE
    train_pre_batch = RandomProcessDatapoint(
        target_shape=teacher_model_input_shape if use_distill else input_shape,
        central_crop=-1,  # Resize directly w/o crop, if random_crop_min not in (0, 1)
        random_crop_min=random_crop_min,
        resize_method=resize_method,
        resize_antialias=resize_antialias,
        random_erasing_prob=random_erasing_prob,
        magnitude=magnitude,
        num_layers=num_layers,
        use_positional_related_ops=use_positional_related_ops,
        use_token_label=use_token_label,
        token_label_target_patches=token_label_target_patches,
        num_classes=num_classes,
        **augment_kwargs,
    )
    if use_shuffle:
        train_dataset = train_dataset.shuffle(buffer_size, seed=seed)
    train_dataset = train_dataset.map(train_pre_batch, num_parallel_calls=AUTOTUNE).batch(batch_size, drop_remainder=is_tpu)

    mean, std = init_mean_std_by_rescale_mode(rescale_mode)
    if use_token_label:
        train_post_batch = lambda xx, yy, token_label: ((xx - mean) / std, tf.one_hot(yy, num_classes), token_label)
    else:
        train_post_batch = lambda xx, yy: ((xx - mean) / std, tf.one_hot(yy, num_classes))
    train_dataset = train_dataset.map(train_post_batch, num_parallel_calls=AUTOTUNE)
    train_dataset = apply_mixup_cutmix(train_dataset, mixup_alpha, cutmix_alpha, switch_prob=0.5)

    if use_token_label:
        train_dataset = train_dataset.map(lambda xx, yy, token_label: (xx, (yy, token_label)))
    elif use_distill:
        print(">>>> KLDivergence teacher model provided.")
        train_dataset = build_distillation_dataset(train_dataset, teacher_model, input_shape)

    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)
    # return train_dataset

    """ Test dataset """
    test_dataset = dataset.get("validation", dataset.get("test", None))
    if test_dataset is not None:
        # test_pre_batch = lambda xx: evaluation_process_resize_crop(xx, input_shape[:2], eval_central_crop, resize_method, resize_antialias)  # timm
        test_pre_batch = lambda xx: evaluation_process_crop_resize(xx, input_shape[:2], eval_central_crop, resize_method, resize_antialias)
        test_dataset = test_dataset.map(test_pre_batch, num_parallel_calls=AUTOTUNE)
        # Have to drop_remainder also for test set...
        test_dataset = test_dataset.batch(batch_size, drop_remainder=is_tpu)
        if use_token_label:
            test_post_batch = lambda xx, yy: ((xx - mean) / std, (tf.one_hot(yy, num_classes), None))  # just give None on token_label data position
        elif use_distill:
            test_post_batch = lambda xx, yy: ((xx - mean) / std, (tf.one_hot(yy, num_classes), None))
        else:
            test_post_batch = lambda xx, yy: ((xx - mean) / std, tf.one_hot(yy, num_classes))
        test_dataset = test_dataset.map(test_post_batch)
    return train_dataset, test_dataset, total_images, num_classes, steps_per_epoch


""" Show """


def show_batch_sample(dataset, rescale_mode="tf", rows=-1, base_size=3):
    from keras_cv_attention_models import visualizing
    from keras_cv_attention_models.imagenet.eval_func import decode_predictions

    if isinstance(dataset, (list, tuple)):
        images, labels = dataset
    elif isinstance(dataset.element_spec[1], tuple):
        images, (labels, token_label) = dataset.as_numpy_iterator().next()
    else:
        images, labels = dataset.as_numpy_iterator().next()
    mean, std = init_mean_std_by_rescale_mode(rescale_mode)
    mean, std = (mean.numpy(), std.numpy()) if hasattr(mean, "numpy") else (mean, std)
    images = (images * std + mean) / 255

    if tf.shape(labels)[-1] == 1000:
        labels = [ii[0][1] for ii in decode_predictions(labels, top=1)]
    elif tf.rank(labels[0]) == 1:
        labels = tf.argmax(labels, axis=-1).numpy()  # If 2 dimension
    ax, _ = visualizing.stack_and_plot_images(images, texts=labels, rows=rows, ax=None, base_size=base_size)
    return ax


def show_token_label_patches_single(image, token_label, rescale_mode="tf", top_k=3, resize_patch_shape=(160, 160)):
    from keras_cv_attention_models import visualizing

    mean, std = init_mean_std_by_rescale_mode(rescale_mode)
    mean, std = (mean.numpy(), std.numpy()) if hasattr(mean, "numpy") else (mean, std)
    image = (image * std + mean) / 255

    height, width = image.shape[:2]
    num_height_patch, num_width_patch = token_label.shape[0], token_label.shape[1]
    height_patch, width_patch = int(tf.math.ceil(height / num_height_patch)), int(tf.math.ceil(width / num_width_patch))
    token_label_scores, token_label_classes = tf.math.top_k(token_label, top_k)
    token_label_scores, token_label_classes = token_label_scores.numpy(), token_label_classes.numpy()
    # fig, axes = plt.subplots(num_height_patch, num_width_patch)

    image_pathes, labels = [], []
    for hh_id in range(num_height_patch):
        hh_image = image[hh_id * height_patch : (hh_id + 1) * height_patch]
        for ww_id in range(num_width_patch):
            image_patch = hh_image[:, ww_id * width_patch : (ww_id + 1) * width_patch]
            image_pathes.append(tf.image.resize(image_patch, resize_patch_shape).numpy())
            scores = ",".join(["{:.1f}".format(ii * 100) for ii in token_label_scores[hh_id, ww_id]])
            classes = ",".join(["{:d}".format(ii) for ii in token_label_classes[hh_id, ww_id].astype("int")])
            labels.append(classes + "\n" + scores)
    visualizing.stack_and_plot_images(image_pathes, labels)
