import numpy as np
import tensorflow as tf

def __compute_loss__(feature_extractor, input_image, filter_index):
    activation = feature_extractor(input_image)
    # We avoid border artifacts by only involving non-border pixels in the loss.
    filter_activation = activation[:, 2:-2, 2:-2, filter_index]
    return tf.reduce_mean(filter_activation)

@tf.function
def __gradient_ascent_step__(feature_extractor, img, filter_index, learning_rate):
    with tf.GradientTape() as tape:
        tape.watch(img)
        loss = __compute_loss__(feature_extractor, img, filter_index)
    # Compute gradients.
    grads = tape.gradient(loss, img)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    img += learning_rate * grads
    return loss, img

def __initialize_image__(img_width, img_height):
    # We start from a gray image with some random noise
    img = tf.random.uniform((1, img_width, img_height, 3))
    # ResNet50V2 expects inputs in the range [-1, +1].
    # Here we scale our random inputs to [-0.125, +0.125]
    return (img - 0.5) * 0.25

def __deprocess_image__(img):
    # Normalize array: center on 0., ensure variance is 0.15
    img -= img.mean()
    img /= img.std() + 1e-5
    img *= 0.15

    # Center crop
    img = img[25:-25, 25:-25, :]

    # Clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)

    # Convert to RGB array
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")
    return img


def visualize_filters(model, layer_name, filter_index_list, img_width, img_height, iterations=30, learning_rate=10.0):
    """
    Copied and modified from https://keras.io/examples/vision/visualizing_what_convnets_learn/

    Example:
    >>> from keras_cv_attention_models import visualizing
    >>> model = keras.applications.ResNet50V2(weights="imagenet", include_top=False)
    >>> losses, all_images = visualizing.visualize_filters(model, "conv3_block4_out", [0], 180, 180)
    >>> print(f"{losses[0].numpy() = }, {all_images[0].shape = }")
    # losses[0].numpy() = 13.749493, all_images[0].shape = (130, 130, 3)
    >>> plt.imshow(all_images[0])
    """
    # Set up a model that returns the activation values for our target layer
    layer = model.get_layer(name=layer_name)
    feature_extractor = tf.keras.Model(inputs=model.inputs, outputs=layer.output)

    # We run gradient ascent for [iterations] steps
    losses, all_images = [], []
    for filter_index in filter_index_list:
        print("Processing filter %d" % (filter_index,))
        image = __initialize_image__(img_width, img_height)
        for iteration in range(iterations):
            loss, image = __gradient_ascent_step__(feature_extractor, image, filter_index, learning_rate)
        # Decode the resulting input image
        image = __deprocess_image__(image[0].numpy())
        losses.append(loss)
        all_images.append(image)

    return losses, all_images


def visualize_filters_result_to_single_image(all_images, margin=5, width=-1):
    """
    Copied and modified from https://keras.io/examples/vision/visualizing_what_convnets_learn/

    Example:
    >>> from keras_cv_attention_models import visualizing
    >>> model = keras.applications.ResNet50V2(weights="imagenet", include_top=False)
    >>> losses, all_images = visualizing.visualize_filters(model, "conv3_block4_out", range(10), 180, 180)
    >>> print(f"{losses[0].numpy() = }, {len(all_images) = }, {all_images[0].shape = }")
    # losses[0].numpy() = 13.749493, len(all_images) = 10, all_images[0].shape = (130, 130, 3)
    >>> image = visualizing.visualize_filters_result_to_single_image(all_images)
    >>> print(f"{image.shape = }")
    # image.shape = (265, 670, 3)
    >>> plt.imshow(image)
    """
    channel = all_images[0].shape[-1]
    total = len(all_images)
    width = int(np.ceil(np.sqrt(total))) if width < 1 else width
    for ww in range(width, total + 1):
        if total % ww == 0:
            width = ww
            break
    height = total // width
    all_images = all_images[:height * width]
    print(">>>> width:", width, ", height:", height, ", len(all_images):", len(all_images))

    ww_margin = np.zeros([all_images[0].shape[0], margin, channel], dtype=all_images[0].dtype)
    ww_margined_images = [np.hstack([ii, ww_margin]) for ii in all_images]
    hstacked_images = [np.hstack(ww_margined_images[ii: ii + width]) for ii in range(0, len(ww_margined_images), width)]

    hh_margin = np.zeros([margin, hstacked_images[0].shape[1], channel], dtype=hstacked_images[0].dtype)
    hh_margined_images = [np.vstack([ii, hh_margin]) for ii in hstacked_images]
    vstacked_images = np.vstack(hh_margined_images)
    return vstacked_images[:-margin, :-margin]


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Copied From: https://keras.io/examples/vision/grad_cam/

    Example:
    >>> from keras_cv_attention_models import visualizing
    >>> from skimage.data import chelsea
    >>> mm = keras.applications.Xception()
    >>> orign_image = chelsea().astype('float32')
    >>> img = tf.expand_dims(tf.image.resize(orign_image, mm.input_shape[1:-1]), 0)
    >>> img = keras.applications.imagenet_utils.preprocess_input(img, mode='tf')
    >>> heatmap, preds = visualizing.make_gradcam_heatmap(img, mm, 'block14_sepconv2_act')
    >>> print(f"{preds.shape = }, {heatmap.shape = }, {heatmap.max() = }, {heatmap.min() = }")
    # preds.shape = (1, 1000), heatmap.shape = (10, 10), heatmap.max() = 1.0, heatmap.min() = 0.0
    >>> print(keras.applications.imagenet_utils.decode_predictions(preds)[0][0])
    # ('n02124075', 'Egyptian_cat', 0.8749054)

    >>> plt.imshow(heatmap)
    """

    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    grad_model = tf.keras.models.Model([model.inputs], [model.get_layer(last_conv_layer_name).output, model.output])

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    # This is a vector where each entry is the mean intensity of the gradient
    # over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy(), preds.numpy()


def make_and_apply_gradcam_heatmap(orign_image, processed_image, model, last_conv_layer_name, pred_index=None, alpha=0.4):
    """
    Copied From: https://keras.io/examples/vision/grad_cam/

    Example:
    >>> from keras_cv_attention_models import visualizing
    >>> from skimage.data import chelsea
    >>> mm = keras.applications.Xception()
    >>> orign_image = chelsea().astype('float32')
    >>> img = tf.expand_dims(tf.image.resize(orign_image, mm.input_shape[1:-1]), 0)
    >>> img = keras.applications.imagenet_utils.preprocess_input(img, mode='tf')
    >>> superimposed_img, heatmap, preds = visualizing.make_and_apply_gradcam_heatmap(orign_image, img, mm, "block14_sepconv2_act")
    >>> # Ouput info
    >>> print(f"{preds.shape = }, {heatmap.shape = }, {heatmap.max() = }, {heatmap.min() = }")
    # preds.shape = (1, 1000), heatmap.shape = (10, 10), heatmap.max() = 1.0, heatmap.min() = 0.0
    >>> print(keras.applications.imagenet_utils.decode_predictions(preds)[0][0])
    # ('n02124075', 'Egyptian_cat', 0.8749054)
    >>> print(f"{superimposed_img.shape = }, {superimposed_img.max() = }, {superimposed_img.min() = }")
    # superimposed_img.shape = (300, 451, 3), superimposed_img.max() = 1.0, superimposed_img.min() = 0.0

    >>> plt.imshow(superimposed_img)
    """

    import matplotlib.cm as cm

    heatmap, preds = make_gradcam_heatmap(processed_image, model, last_conv_layer_name, pred_index=pred_index)

    # Use jet colormap to colorize heatmap. Use RGB values of the colormap
    jet = cm.get_cmap("jet")
    jet_colors = jet(tf.range(256))[:, :3]
    jet_heatmap = jet_colors[tf.cast(heatmap * 255, "uint8").numpy()]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.image.resize(jet_heatmap, (orign_image.shape[:2]))  # [0, 1]

    # Superimpose the heatmap on original image
    orign_image = orign_image.astype("float32") / 255 if orign_image.max() > 127 else orign_image
    superimposed_img = (jet_heatmap * alpha + orign_image).numpy()
    superimposed_img /= superimposed_img.max()
    return superimposed_img, heatmap, preds
