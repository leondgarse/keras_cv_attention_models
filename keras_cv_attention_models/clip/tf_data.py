import tensorflow as tf
from keras_cv_attention_models.imagenet.data import init_mean_std_by_rescale_mode, tf_imread, random_crop_and_resize_image, build_custom_dataset


def image_process(image, image_size=(224, 224), is_train=True):
    image = tf_imread(image)
    if is_train:
        image = random_crop_and_resize_image(image, image_size, scale=(0.9, 1.0), method="bicubic", antialias=True)[0]
    else:
        image = tf.image.resize(image, image_size, method="bicubic", antialias=True)
    image = tf.cast(image, tf.float32)
    image.set_shape([*image_size, 3])
    return image


def init_dataset(data_path, caption_tokenizer, batch_size=64, image_size=224, rescale_mode="torch"):
    dataset, total_images, num_classes, num_channels = build_custom_dataset(data_path, with_info=True, caption_tokenizer=caption_tokenizer)

    mean, std = init_mean_std_by_rescale_mode(rescale_mode)
    image_size = image_size if isinstance(image_size, (list, tuple)) else [image_size, image_size]

    AUTOTUNE, buffer_size, seed = tf.data.AUTOTUNE, batch_size * 100, None
    train_pre_batch = lambda data_point: (image_process(data_point["image"], image_size, is_train=True), data_point["caption"])
    y_true = tf.range(batch_size)
    train_post_batch = lambda xx, caption: (((xx - mean) / std, caption), y_true)

    train_dataset = dataset["train"]
    train_dataset = train_dataset.shuffle(buffer_size, seed=seed).map(train_pre_batch, num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True).map(train_post_batch, num_parallel_calls=AUTOTUNE)
    train_dataset = train_dataset.prefetch(buffer_size=AUTOTUNE)

    test_dataset = dataset.get("validation", dataset.get("test", None))
    if test_dataset is not None:
        test_pre_batch = lambda data_point: (image_process(data_point["image"], image_size, is_train=False), data_point["caption"])
        test_dataset = test_dataset.map(test_pre_batch, num_parallel_calls=AUTOTUNE)
        test_dataset = test_dataset.batch(batch_size, drop_remainder=True).map(train_post_batch)

    return train_dataset, test_dataset
