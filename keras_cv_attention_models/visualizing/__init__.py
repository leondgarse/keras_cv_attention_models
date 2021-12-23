from keras_cv_attention_models.visualizing.visualizing import (
    visualize_filters,
    visualize_filters_result_to_single_image,
    make_gradcam_heatmap,
    make_and_apply_gradcam_heatmap,
    plot_attention_score_maps,
)

visualize_filters.__doc__ = """
Copied and modified from https://keras.io/examples/vision/visualizing_what_convnets_learn/
Set up a model that returns the activation values for our target layer.

Args:
  model: keras model used for visualizing.
  layer_name: target layer name in model.
  filter_index_list: channel indexes for visualizing.
  input_shape: model input_shape. Default `None` means using `model.input_shape`.
  iterations: total steps runnung gradient ascent.
  learning_rate: learning rate in runnung gradient ascent.
  
Returns: losses, all_images

Example:
>>> from keras_cv_attention_models import visualizing
>>> model = keras.applications.ResNet50V2(input_shape=(224, 224, 3), weights="imagenet", include_top=False)
>>> losses, all_images = visualizing.visualize_filters(model, "conv3_block4_out", [0], 180, 180)
>>> print(f"{losses[0].numpy() = }, {all_images[0].shape = }")
# losses[0].numpy() = 13.749493, all_images[0].shape = (130, 130, 3)
>>> plt.imshow(all_images[0])
"""

visualize_filters_result_to_single_image.__doc__ = """
Copied and modified from https://keras.io/examples/vision/visualizing_what_convnets_learn/

Returns: stacked_image

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

make_gradcam_heatmap.__doc__ = """
Copied From: https://keras.io/examples/vision/grad_cam/

Returns: heatmap, preds

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

make_and_apply_gradcam_heatmap.__doc__ = """
Copied From: https://keras.io/examples/vision/grad_cam/

Returns: superimposed_img, heatmap, preds

Example:
>>> from keras_cv_attention_models import visualizing
>>> from skimage.data import chelsea
>>> mm = keras.applications.Xception()
>>> orign_image = chelsea().astype('float32')
>>> superimposed_img, heatmap, preds = visualizing.make_and_apply_gradcam_heatmap(mm, orign_image, "block14_sepconv2_act")

>>> # Ouput info
>>> print(f"{preds.shape = }, {heatmap.shape = }, {heatmap.max() = }, {heatmap.min() = }")
# preds.shape = (1, 1000), heatmap.shape = (10, 10), heatmap.max() = 1.0, heatmap.min() = 0.0
>>> print(keras.applications.imagenet_utils.decode_predictions(preds)[0][0])
# ('n02124075', 'Egyptian_cat', 0.8749054)
>>> print(f"{superimposed_img.shape = }, {superimposed_img.max() = }, {superimposed_img.min() = }")
# superimposed_img.shape = (300, 451, 3), superimposed_img.max() = 1.0, superimposed_img.min() = 0.0
"""

plot_attention_score_maps.__doc__ = """

Returns: mask, cum_mask
"""
