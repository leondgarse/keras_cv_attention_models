import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow import keras


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


def init_dataset(
    data_name="imagenet2012",
    input_shape=(224, 224),
    batch_size=64,
    buffer_size=1000,
    info_only=False,
    magnitude=0,
    mixup_alpha=0,
    keep_shape=False,
):
    dataset, info = tfds.load(data_name, with_info=True)
    num_classes = info.features["label"].num_classes
    total_images = info.splits["train"].num_examples
    steps_per_epoch = int(np.ceil(total_images / float(batch_size)))
    if info_only:
        return total_images, num_classes, steps_per_epoch

    AUTOTUNE = tf.data.AUTOTUNE
    train_process = RandomProcessImage(input_shape, magnitude, keep_shape=keep_shape)
    train = dataset["train"].map(lambda xx: train_process(xx), num_parallel_calls=AUTOTUNE)
    test_process = RandomProcessImage(input_shape, magnitude=-1, keep_shape=keep_shape)
    if "validation" in dataset:
        test = dataset["validation"].map(lambda xx: test_process(xx))
    elif "test" in dataset:
        test = dataset["test"].map(lambda xx: test_process(xx))

    rescaling = lambda xx: (xx - 127.5) * 0.0078125
    as_one_hot = lambda yy: tf.one_hot(yy, num_classes)
    train_dataset = train.shuffle(buffer_size).batch(batch_size).prefetch(buffer_size=AUTOTUNE)
    test_dataset = test.batch(batch_size).map(lambda xx, yy: (rescaling(xx), as_one_hot(yy)))

    if mixup_alpha > 0 and mixup_alpha <= 1:
        print(">>>> mixup_alpha provided:", mixup_alpha)
        train_dataset = train_dataset.map(lambda xx, yy: mixup(rescaling(xx), as_one_hot(yy), alpha=mixup_alpha))
    else:
        train_dataset = train_dataset.map(lambda xx, yy: (rescaling(xx), as_one_hot(yy)))
    return train_dataset, test_dataset, total_images, num_classes, steps_per_epoch
