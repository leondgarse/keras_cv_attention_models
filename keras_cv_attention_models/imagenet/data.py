import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras


def random_crop_fraction(size, scale=(0.08, 1.0), ratio=(0.75, 1.3333333333333333)):
    """https://github.com/tensorflow/models/blob/master/official/vision/image_classification/preprocessing.py
    RandomResizedCrop related function.
    As outputs are converted int, for running 1e5 times, results are about scale in (0.0806, 1.0) , ratio in (0.735, 1.36).

    hh_crop * ww_crop = crop_area = crop_fraction * area, crop_fraction in scale=(0.08, 1.0)
    hh_crop / ww_crop > ratio[0]
    hh_crop / ww_crop < ratio[1]
    ==> ww_crop > sqrt(crop_area / ratio[1])
        ww_crop < sqrt(crop_area / ratio[0])

    Args:
      size (tuple of int): input image shape. `area = size[0] * size[1]`.
      scale (tuple of float): scale range of the cropped image. crop_area in range `(scale[0] * area, sacle[1] * area)`.
      ratio (tuple of float): aspect ratio range of the cropped image. hh_crop / ww_crop in range `(ratio[0], ratio[1])`.

    Returns: cropped size `hh_crop, ww_crop`.
    """
    area = size[0] * size[1]
    crop_area = tf.random.uniform((), *scale) * area
    ww_fraction_min = tf.maximum(crop_area / size[0], tf.sqrt(crop_area / ratio[1]))
    ww_fraction_max = tf.minimum(tf.cast(size[1], "float32"), tf.sqrt(crop_area / ratio[0]))
    ww_fraction = tf.random.uniform((), ww_fraction_min, ww_fraction_max)
    # hh_crop, ww_crop = tf.cast(tf.math.ceil(crop_area / ww_fraction), "int32"), tf.cast(tf.math.ceil(ww_fraction), "int32")
    hh_crop, ww_crop = tf.cast(tf.math.floor(crop_area / ww_fraction), "int32"), tf.cast(tf.math.floor(ww_fraction), "int32")
    # return hh_crop, ww_crop, crop_area, ww_fraction_min, ww_fraction_max, ww_fraction
    return tf.minimum(hh_crop, size[0] - 1), tf.minimum(ww_crop, size[1] - 1)


class RandomProcessImage:
    def __init__(self, target_shape=(300, 300), magnitude=0, central_crop=1.0, random_crop_min=1.0, resize_method="bilinear"):
        self.target_shape, self.magnitude = target_shape, magnitude
        self.target_shape = target_shape if len(target_shape) == 2 else target_shape[:2]
        self.central_crop, self.random_crop_min, self.resize_method = central_crop, random_crop_min, resize_method

        if magnitude > 0:
            from keras_cv_attention_models.imagenet import augment

            translate_const, cutout_const = 100, 40
            # translate_const = int(target_shape[0] * 10 / magnitude)
            # cutout_const = int(target_shape[0] * 40 / 224)
            print(">>>> RandAugment: magnitude = %d, translate_const = %d, cutout_const = %d" % (magnitude, translate_const, cutout_const))
            aa = augment.RandAugment(magnitude=magnitude, translate_const=translate_const, cutout_const=cutout_const)
            # aa.available_ops = ["AutoContrast", "Equalize", "Invert", "Rotate", "Posterize", "Solarize", "Color", "Contrast", "Brightness", "Sharpness", "ShearX", "ShearY", "TranslateX", "TranslateY", "Cutout", "SolarizeAdd"]
            self.process = lambda img: tf.image.random_flip_left_right(aa.distort(img))
        elif magnitude == 0:
            self.process = lambda img: tf.image.random_flip_left_right(img)
        else:
            self.process = lambda img: img

    def __call__(self, datapoint):
        image = datapoint["image"]
        # if self.keep_shape:
        #     cropped_shape = tf.reduce_min(tf.keras.backend.shape(image)[:2])
        #     image = tf.image.random_crop(image, (cropped_shape, cropped_shape, 3))

        if self.random_crop_min > 0 and self.random_crop_min < 1:
            # cropped_shape = tf.cast(tf.cast(tf.keras.backend.shape(image)[:2], tf.float32) * self.random_crop_min, tf.int32)
            # input_image = random_crop_with_min_fraction(image, self.random_crop_min)
            hh, ww = random_crop_fraction(image.shape, scale=(self.random_crop_min, 1.0))
            input_image = tf.image.random_crop(image, (hh, ww, 3))
        else:
            input_image = tf.image.central_crop(image, self.central_crop)
        input_image = tf.image.resize(input_image, self.target_shape, method=self.resize_method)
        label = datapoint["label"]
        input_image = self.process(input_image)
        input_image = tf.cast(input_image, tf.float32)
        return input_image, label


def sample_beta_distribution(size, concentration_0=0.4, concentration_1=0.4):
    gamma_1_sample = tf.random.gamma(shape=[size], alpha=concentration_1)
    gamma_2_sample = tf.random.gamma(shape=[size], alpha=concentration_0)
    return gamma_1_sample / (gamma_1_sample + gamma_2_sample)


def mixup(image, label, alpha=0.4):
    """Applies Mixup regularization to a batch of images and labels.

    [1] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
    Mixup: Beyond Empirical Risk Minimization.
    ICLR'18, https://arxiv.org/abs/1710.09412
    """
    # mix_weight = tfp.distributions.Beta(alpha, alpha).sample([batch_size, 1])
    batch_size = tf.shape(image)[0]
    mix_weight = sample_beta_distribution(batch_size, alpha, alpha)
    mix_weight = tf.maximum(mix_weight, 1.0 - mix_weight)

    # Regard values with `> 0.9` as no mixup, this probability is near `1 - alpha`
    # alpha: no_mixup --> {0.2: 0.6714, 0.4: 0.47885, 0.6: 0.35132, 0.8: 0.26354, 1.0: 0.19931}
    mix_weight = tf.where(mix_weight > 0.9, tf.ones_like(mix_weight), mix_weight)

    label_mix_weight = tf.cast(tf.expand_dims(mix_weight, -1), "float32")
    img_mix_weight = tf.cast(tf.reshape(mix_weight, [batch_size, 1, 1, 1]), image.dtype)

    shuffle_index = tf.random.shuffle(tf.range(batch_size))
    image = image * img_mix_weight + tf.gather(image, shuffle_index) * (1.0 - img_mix_weight)
    label = tf.cast(label, "float32")
    label = label * label_mix_weight + tf.gather(label, shuffle_index) * (1 - label_mix_weight)
    return image, label


def get_box(mix_weight, height, width):
    cut_rate = tf.math.sqrt(1.0 - mix_weight) / 2
    cut_h, cut_w = tf.cast(cut_rate * float(height), tf.int32), tf.cast(cut_rate * float(width), tf.int32)
    center_y = tf.random.uniform((1,), minval=cut_h, maxval=height - cut_h, dtype=tf.int32)[0]
    center_x = tf.random.uniform((1,), minval=cut_w, maxval=width - cut_w, dtype=tf.int32)[0]
    return center_y - cut_h, center_x - cut_w, cut_h, cut_w


def cutmix(images, labels, alpha=0.5, min_mix_weight=0.01):
    """
    Copied and modified from https://keras.io/examples/vision/cutmix/

    Example:
    >>> from keras_cv_attention_models.imagenet import data
    >>> import tensorflow_datasets as tfds
    >>> dataset = tfds.load('cifar10', split='train').batch(16)
    >>> dd = dataset.as_numpy_iterator().next()
    >>> images, labels = dd['image'], tf.one_hot(dd['label'], depth=10)
    >>> aa, bb = data.cutmix(images, labels)
    >>> print(bb.numpy()[bb.numpy() != 0])
    >>> plt.imshow(np.hstack(aa))
    """
    # Get a sample from the Beta distribution
    batch_size = tf.shape(images)[0]
    _, hh, ww, _ = images.shape
    mix_weight = sample_beta_distribution(1, alpha, alpha)[0]  # same value in batch
    if mix_weight < min_mix_weight or 1 - mix_weight < min_mix_weight:
        # For input_shape=224, min_mix_weight=0.01, min_height = 224 * 0.1 = 22.4
        return images, labels

    offset_height, offset_width, target_height, target_width = get_box(mix_weight, hh, ww)
    crops = tf.image.crop_to_bounding_box(images, offset_height, offset_width, target_height, target_width)
    pad_crops = tf.image.pad_to_bounding_box(crops, offset_height, offset_width, hh, ww)

    shuffle_index = tf.random.shuffle(tf.range(batch_size))
    images = images - pad_crops + tf.gather(pad_crops, shuffle_index)
    labels = tf.cast(labels, "float32")
    label_mix_weight = tf.cast(tf.expand_dims(mix_weight, -1), "float32")
    labels = labels * label_mix_weight + tf.gather(labels, shuffle_index) * (1 - label_mix_weight)
    return images, labels


def random_erasing_per_pixel(image, num_layers=1, scale=(0.02, 1/3), ratio=(0.3, 10/3), probability=0.5):
    """ https://github.com/rwightman/pytorch-image-models/blob/master/timm/data/random_erasing.py """
    if tf.random.uniform(()) > probability:
        return image

    mean = [123.675, 116.28, 103.53]
    std = [58.395, 57.120003, 57.375]
    height, width, _ = image.shape
    for _ in range(num_layers):
        hh, ww = random_crop_fraction((height, width), scale=scale, ratio=ratio)
        hh_ss = tf.random.uniform((), 0, height - hh, dtype='int32')
        ww_ss = tf.random.uniform((), 0, width - ww, dtype='int32')
        mask = tf.random.normal([hh, ww, 3], mean=mean, stddev=std)
        mask = tf.clip_by_value(mask, 0.0, 255.0)    # value in [0, 255]
        aa = tf.concat([image[:hh_ss, ww_ss:ww_ss + ww], mask, image[hh_ss + hh:, ww_ss:ww_ss + ww]], axis=0)
        image = tf.concat([image[:, :ww_ss], aa, image[:, ww_ss + ww:]], axis=1)
    return image


def init_dataset(
    data_name="imagenet2012",
    input_shape=(224, 224),
    batch_size=64,
    buffer_size=1000,
    info_only=False,
    magnitude=0,
    mixup_alpha=0,
    cutmix_alpha=0,
    central_crop=1.0,
    random_crop_min=1.0,
    resize_method="bilinear",
    mode="tf",
):
    """Init dataset by name.
    returns train_dataset, test_dataset, total_images, num_classes, steps_per_epoch.
    """
    dataset, info = tfds.load(data_name, with_info=True)
    num_classes = info.features["label"].num_classes
    total_images = info.splits["train"].num_examples
    steps_per_epoch = int(tf.math.ceil(total_images / float(batch_size)))
    if info_only:
        return total_images, num_classes, steps_per_epoch

    AUTOTUNE = tf.data.AUTOTUNE
    train_process = RandomProcessImage(input_shape, magnitude, central_crop=central_crop, random_crop_min=random_crop_min, resize_method=resize_method)
    train = dataset["train"].map(lambda xx: train_process(xx), num_parallel_calls=AUTOTUNE)
    test_process = RandomProcessImage(input_shape, magnitude=-1, central_crop=central_crop, random_crop_min=1.0, resize_method=resize_method)
    if "validation" in dataset:
        test = dataset["validation"].map(lambda xx: test_process(xx))
    elif "test" in dataset:
        test = dataset["test"].map(lambda xx: test_process(xx))

    if mode == "torch":
        mean = tf.constant([0.485, 0.456, 0.406]) * 255.0
        std = tf.constant([0.229, 0.224, 0.225]) * 255.0
        rescaling = lambda xx: (xx - mean) / std
    else:
        rescaling = lambda xx: (xx - 127.5) * 0.0078125

    as_one_hot = lambda yy: tf.one_hot(yy, num_classes)
    train_dataset = train.shuffle(buffer_size).batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    test_dataset = test.batch(batch_size).map(lambda xx, yy: (rescaling(xx), as_one_hot(yy)))

    if mixup_alpha > 0 and mixup_alpha <= 1 and cutmix_alpha > 0 and cutmix_alpha <= 1:
        print(">>>> Both mixup_alpha and cutmix_alpha provided: mixup_alpha = {}, cutmix_alpha = {}".format(mixup_alpha, cutmix_alpha))
        mixup_cutmix = lambda xx, yy: tf.cond(
            tf.random.uniform(()) > 0.5,    # switch_prob = 0.5
            lambda: mixup(rescaling(xx), as_one_hot(yy), alpha=mixup_alpha),
            lambda: cutmix(rescaling(xx), as_one_hot(yy), alpha=cutmix_alpha),
        )
        train_dataset = train_dataset.map(mixup_cutmix)
    elif mixup_alpha > 0 and mixup_alpha <= 1:
        print(">>>> mixup_alpha provided:", mixup_alpha)
        train_dataset = train_dataset.map(lambda xx, yy: mixup(rescaling(xx), as_one_hot(yy), alpha=mixup_alpha))
    elif cutmix_alpha > 0 and cutmix_alpha <= 1:
        print(">>>> cutmix_alpha provided:", cutmix_alpha)
        train_dataset = train_dataset.map(lambda xx, yy: cutmix(rescaling(xx), as_one_hot(yy), alpha=cutmix_alpha))
    else:
        train_dataset = train_dataset.map(lambda xx, yy: (rescaling(xx), as_one_hot(yy)))
    return train_dataset, test_dataset, total_images, num_classes, steps_per_epoch
