import numpy as np
import tensorflow as tf
from tqdm import tqdm
from keras_cv_attention_models.visualizing.plot_func import get_plot_cols_rows, stack_and_plot_images

""" visualize_filters """


def __gradient_ascent_step__(feature_extractor, image_var, filter_index, optimizer):
    with tf.GradientTape() as tape:
        tape.watch(image_var)
        activation = feature_extractor(tf.expand_dims(image_var[0], 0))
        # We avoid border artifacts by only involving non-border pixels in the loss.
        # filter_activation = activation[:, 2:-2, 2:-2, filter_index]
        filter_activation = activation[..., filter_index]
        loss = tf.reduce_mean(filter_activation)
    # Compute gradients.
    grads = tape.gradient(loss, image_var)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    # image_var += 10.0 * grads # 10.0 is learning_rate
    optimizer.apply_gradients(zip(grads * -1, image_var))  # For SGD, w = w - learning_rate * g
    return loss, image_var


def __initialize_image__(img_width, img_height, rescale_mode="tf", value_range=0.125):
    # We start from a gray image with some random noise. Pick image vvalue range in middle
    min_val, max_val = int(128 - 128 * value_range), int(128 + 128 * value_range)
    img = tf.random.uniform((img_width, img_height, 3), min_val, max_val)
    # For ResNet50V2 expects inputs in the range [-1, +1], results in `[-0.25, 0.25]`
    return tf.keras.applications.imagenet_utils.preprocess_input(img, mode=rescale_mode).numpy()


def __deprocess_image__(img, crop_border=0.1):
    # Normalize array: center on 0., ensure variance is 0.15
    img -= img.mean(axis=(0, 1))
    img /= img.std(axis=(0, 1)) + 1e-5
    img *= 0.15

    # Center crop
    hh_crop, ww_crop = int(img.shape[0] * crop_border), int(img.shape[1] * crop_border)
    img = img[hh_crop:-hh_crop, ww_crop:-ww_crop]

    # Clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)

    # Convert to RGB array
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")
    return img


def visualize_filters(
    model,
    layer_name="auto",
    filter_index_list=[0],
    input_shape=None,
    rescale_mode="auto",
    iterations=30,
    optimizer="SGD",  # "SGD" / "RMSprop" / "Adam"
    learning_rate="auto",
    value_range=0.125,
    random_magnitude=1.0,  # basic random value for `tf.roll` and `random_rotation` is `4` and `1`.
    crop_border=0.1,
    base_size=3,
):
    from tensorflow.keras.preprocessing.image import random_rotation

    # Set up a model that returns the activation values for our target layer
    # model = tf.keras.models.clone_model(model)
    if layer_name == "auto":
        layer = model.layers[-1]
        layer_name = layer.name
    else:
        layer = model.get_layer(name=layer_name)
    feature_extractor = tf.keras.Model(inputs=model.inputs[0], outputs=layer.output)
    input_shape = model.input_shape[1:-1] if input_shape is None else input_shape[:2]
    assert input_shape[0] is not None and input_shape[1] is not None
    print(">>>> Total filters for layer {}: {}, input_shape: {}".format(layer_name, layer.output_shape[-1], input_shape))

    auto_lr = {"ada": 0.1, "rms": 1.0, "sgd": 10.0}
    if isinstance(optimizer, str):
        # After TF 2.11.0, seems we cannot call `apply_gradients` on `image_var`, use legacy instead...
        tf_optimizers = tf.optimizers.legacy if hasattr(tf.optimizers, "legacy") else tf.optimizers

        optimizer = optimizer.lower()
        learning_rate = auto_lr.get(optimizer[:3], 1.0) if learning_rate == "auto" else learning_rate
        if optimizer.startswith("rms"):
            optimizer = tf_optimizers.RMSprop(learning_rate, rho=0.999)
        elif optimizer == "adam":
            optimizer = tf_optimizers.Adam(learning_rate)
        elif optimizer == "sgd":
            optimizer = tf_optimizers.SGD(learning_rate)

    if rescale_mode.lower() == "auto":
        rescale_mode = getattr(model, "rescale_mode", "torch")
        print(">>>> rescale_mode:", rescale_mode)

    # We run gradient ascent for [iterations] steps
    losses, filter_images = [], []
    for filter_index in filter_index_list:
        image = __initialize_image__(input_shape[0], input_shape[1], rescale_mode, value_range=value_range)
        for iteration in tqdm(range(iterations), desc="Processing filter %d" % (filter_index,)):
            image = random_rotation(image, 1 * random_magnitude, row_axis=0, col_axis=1, channel_axis=2, fill_mode="reflect")
            image = tf.roll(image, shift=np.random.randint(-4 * random_magnitude, 4 * random_magnitude, size=2), axis=[0, 1])
            image_var = [tf.Variable(image)]
            # optimizer.build(image_var + feature_extractor.trainable_variables)
            loss, image_var = __gradient_ascent_step__(feature_extractor, image_var, filter_index, optimizer)
            image = image_var[0].numpy()
        # Decode the resulting input image
        image = __deprocess_image__(image, crop_border)
        losses.append(loss.numpy())
        filter_images.append(image)

    ax, _ = stack_and_plot_images(filter_images, base_size=base_size)
    return losses, np.stack(filter_images), ax
