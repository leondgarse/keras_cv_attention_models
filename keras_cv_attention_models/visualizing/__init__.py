from keras_cv_attention_models.visualizing.visualizing import (
    get_plot_cols_rows,
    stack_and_plot_images,
    visualize_filters,
    make_gradcam_heatmap,
    make_and_apply_gradcam_heatmap,
    plot_attention_score_maps,
)

visualize_filters.__doc__ = """
Copied and modified from: https://keras.io/examples/vision/visualizing_what_convnets_learn/
Displaying the visual patterns that convnet filters respond to.
Will create input images that maximize the activation of specific filters in a target layer.
Such images represent a visualization of the pattern that the filter responds to.

Note: models using `Conv2D` with `groups != 1` not supporting on CPU. Needs backward steps.

Args:
  model: keras model used for visualizing.
  layer_name: target layer name in model for visualizing. Default "auto" means using the last layer.
  filter_index_list: channel indexes for visualizing.
  input_shape: model input_shape. Default `None` means using `model.input_shape`.
  rescale_mode: one of ["tf", "torch", "caffe"]. Default `auto` means using model preset.
  iterations: total steps running gradient ascent.
  optimizer: one of ["SGD", "RMSprop", "Adam"] or specified inited optimizer.
  learning_rate: learning rate in runnung gradient ascent. Default "auto" means {"SGD": 10.0, "RMSprop": 1.0, "Adam": 0.1}.
  value_range: indicates input image value range, processed to `[128 - 128 * value_range, 128 + 128 * value_range]`.
  random_magnitude: random magnitude for `tf.roll` and `random_rotation`. basic random value is `4` and `1`.
  crop_border: crop output image border pixels, calculated as:
      `hh_crop, ww_crop = input_shape[0] * crop_border, input_shape[1] * crop_border`
      `img = img[hh_crop:-hh_crop, ww_crop:-ww_crop]`
  base_size: base plotting size for a single image.

Returns:
  losses, filter_images, ax

Example:
>>> from keras_cv_attention_models import visualizing, resnest
>>> mm = resnest.ResNest50()
>>> losses, filter_images, ax = visualizing.visualize_filters(mm, "stack3_block6_out", range(10))
>>> print(f"{losses[0] = }, {filter_images[0].shape = }")
# losses[0] = 23.950336, filter_images[0].shape = (174, 174, 3)
"""

make_gradcam_heatmap.__doc__ = """
Copied and modified from: https://keras.io/examples/vision/grad_cam/
Grad-CAM class activation visualization. Obtain a class activation heatmap for an image classification model.

Args:
  model: keras model used for visualizing.
  processed_image: preprocessed image that can be directly used as model input.
  layer_name: target layer name in model for visualizing. Default "auto" means using the last layer with `len(output_shape) == 4`.
  pred_index: specified visualizing prediction index. Used for image containing multi classes.
      Default `None` means using max probability one.
  use_v2: set False for a simple version, not a big difference.

Returns:
  heatmap, preds

Example:
>>> from keras_cv_attention_models import visualizing, resnest
>>> mm = resnest.ResNest50()
>>> url = 'https://upload.wikimedia.org/wikipedia/commons/b/bc/Free%21_%283987584939%29.jpg'
>>> img = plt.imread(keras.utils.get_file('aa.jpg', url))
>>> img = tf.expand_dims(tf.image.resize(img, mm.input_shape[1:-1]), 0)
>>> img = keras.applications.imagenet_utils.preprocess_input(img, mode='torch')
>>> heatmap, preds = visualizing.make_gradcam_heatmap(mm, img, 'stack4_block3_out')
>>> print(f"{preds.shape = }, {heatmap.shape = }, {heatmap.max() = }, {heatmap.min() = }")
# preds.shape = (1, 1000), heatmap.shape = (7, 7), heatmap.max() = 1.0, heatmap.min() = 0.0
>>> print(keras.applications.imagenet_utils.decode_predictions(preds)[0][0])
# ('n02110063', 'malamute', 0.54736596)

>>> plt.imshow(heatmap)
"""

make_and_apply_gradcam_heatmap.__doc__ = """
Copied and modified from: https://keras.io/examples/vision/grad_cam/
Grad-CAM class activation visualization. Create and plot a superimposed visualization heatmap on image.

Args:
  model: keras model used for visualizing.
  image: Original image for visualizing.
  layer_name: target layer name in model for visualizing. Default "auto" means using the last layer with `len(output_shape) == 4`.
  rescale_mode: image value rescale mode. Mostly used one is "tf" or "torch". Default `auto` means using model preset.
  pred_index: specified visualizing prediction index. Used for image containing multi classes.
      Default `None` means using max probability one.
  alpha: heatmap superimposed alpha over image.
  use_v2: set False for a simple version, not a big difference.
  plot: set False to disable plot image.

Returns:
  superimposed_img, heatmap, preds

Example:
>>> from keras_cv_attention_models import visualizing, resnest
>>> mm = resnest.ResNest50()
>>> url = 'https://upload.wikimedia.org/wikipedia/commons/b/bc/Free%21_%283987584939%29.jpg'
>>> img = plt.imread(keras.utils.get_file('aa.jpg', url))
>>> superimposed_img, heatmap, preds = visualizing.make_and_apply_gradcam_heatmap(mm, img, "stack4_block3_out")
"""

plot_attention_score_maps.__doc__ = """
Visualizing model attention score maps, superimposed with specific image.

Args:
  model: keras model used for visualizing.
  image: Original image for visualizing.
  rescale_mode: image value rescale mode. Mostly used one is "tf" or "torch". Default `auto` means using model preset.
  attn_type: Specify model attention type. Currently supporting `["beit", "levit", "bot", "coatnet", "halo"]`.
      Default "auto" means decide from model name. Technically any attention scores in the same format with
      specified `attn_type` can be supported. Like using `attn_type="beit"` for `VIT` models.
  rows: Specify number of plotted rows. Default `-1` means auto adjust.
  base_size: base plotting size for a single image.

Returns:
  mask, cum_mask, fig

Example:
>>> from keras_cv_attention_models import botnet, halonet, beit, levit, visualizing
>>> url = 'https://upload.wikimedia.org/wikipedia/commons/b/bc/Free%21_%283987584939%29.jpg'
>>> imm = plt.imread(keras.utils.get_file('aa.jpg', url))
>>> _ = visualizing.plot_attention_score_maps(beit.BeitBasePatch16(), imm, rescale_mode='tf', rows=2)
"""
