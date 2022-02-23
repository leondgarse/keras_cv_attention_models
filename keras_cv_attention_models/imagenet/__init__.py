from keras_cv_attention_models.imagenet.eval_func import evaluation, plot_hists, combine_hist_into_one, parse_timm_log
from keras_cv_attention_models.imagenet.train_func import (
    compile_model,
    init_global_strategy,
    init_loss,
    init_lr_scheduler,
    init_model,
    init_optimizer,
    is_decoupled_weight_decay,
    train,
)
from keras_cv_attention_models.imagenet import data
from keras_cv_attention_models.imagenet.data import init_dataset
from keras_cv_attention_models.imagenet import callbacks
from keras_cv_attention_models.imagenet import losses

data.random_crop_fraction.__doc__ = """ https://github.com/tensorflow/models/blob/master/official/vision/image_classification/preprocessing.py
RandomResizedCrop related function.

For hh_crop = height, max_ww_crop = height * ratio[1], max_area_crop_1 = height * height * ratio[1]
For ww_crop = width, max_hh_crop = width / ratio[0], max_area_crop_2 = width * width / ratio[0]
==> scale_max < min(max_area_crop_1, max_area_crop_2, scale[1])

As target_area selected:
For ww_crop = width, max_aspect_ratio = width / (target_area / width) = width * width / target_area
For hh_crop = height, min_aspect_ratio = (target_area / height) / height = target_area / (height * height)
==> ratio in range (min_aspect_ratio, max_aspect_ratio)

Result:
ww_crop * hh_crop = target_area
ww_crop / hh_crop = aspect_ratio
==> ww_crop = int(round(math.sqrt(target_area * aspect_ratio)))
    hh_crop = int(round(math.sqrt(target_area / aspect_ratio)))

As outputs are converted int, for running 1e5 times, results are not exactly in scale and ratio range:
>>> from keras_cv_attention_models.imagenet import data
>>> aa = np.array([data.random_crop_fraction(size=(100, 100), ratio=(0.75, 4./3)) for _ in range(100000)])
>>> hhs, wws = aa[:, 0], aa[:, 1]
>>> print("Scale range:", ((hhs * wws).min() / 1e4, (hhs * wws).max() / 1e4))
# Scale range: (0.075, 0.9801)
>>> print("Ratio range:", ((wws / hhs).min(), (wws / hhs).max()))
# Ratio range: (0.7272727272727273, 1.375)

>>> fig, axes = plt.subplots(4, 1, figsize=(6, 8))
>>> pp = {
>>>     "ratio distribute": wws / hhs,
>>>     "scale distribute": wws * hhs / 1e4,
>>>     "height distribute": hhs,
>>>     "width distribute": wws,
>>> }
>>> for ax, kk in zip(axes, pp.keys()):
>>>     _ = ax.hist(pp[kk], bins=1000, label=kk)
>>>     ax.set_title(kk)
>>> fig.tight_layout()

Args:
  size (tuple of int): input image shape. `area = size[0] * size[1]`.
  scale (tuple of float): scale range of the cropped image. target_area in range `(scale[0] * area, sacle[1] * area)`.
  ratio (tuple of float): aspect ratio range of the cropped image. cropped `width / height`  in range `(ratio[0], ratio[1])`.

Returns: cropped size `hh_crop, ww_crop`.
"""

data.init_mean_std_by_rescale_mode.__doc__ = """
Args:
  rescale_mode: one of ["tf", "torch", "raw01", "raw"].
    - "tf": mean 127.5, std: 128.0, converts [0, 255] -> [-1, 1].
    - "torch": mean [0.485, 0.456, 0.406] * 255.0, std [0.229, 0.224, 0.225] * 255.0.
    - "raw01": mean 0, std 127.5, converts [0, 255] -> [0, 1].
    - "raw": mean 0, std 1, raw output [0, 255].

Returns: mean, std
"""

data.mixup.__doc__ = """ Applies Mixup regularization to a batch of images and labels.
[1] Hongyi Zhang, Moustapha Cisse, Yann N. Dauphin, David Lopez-Paz
Mixup: Beyond Empirical Risk Minimization.
ICLR'18, https://arxiv.org/abs/1710.09412
"""
data.cutmix.__doc__ = """ Copied and modified from https://keras.io/examples/vision/cutmix/

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

data.init_dataset.__doc__ = """ Init dataset by name.
Args:
  data_name: the registered dataset name from `tensorflow_datasets`.
  input_shape: input shape.
  batch_size: batch size.
  buffer_size: dataset shuffle buffer size.
  info_only: boolean value if returns dataset info only.
  mixup_alpha: mixup applying probability.
  cutmix_alpha: cutmix applying probability.
  rescale_mode: one of ["tf", "torch", "raw01", "raw"]. Detail in `data.init_mean_std_by_rescale_mode`. Or specific `(mean, std)` like `(128.0, 128.0)`.
  eval_central_crop: central crop fraction for evaluation. Default `1.0` for disabling, < 0 values set a crop fraction.
  random_crop_min: min scale for `random_crop_fraction`. Max scale is `1.0`, Ratio is `(0.75, 1.3333333)`.
  resize_method: one of ["nearest", "bilinear", "bicubic"]. Resize method for `tf.image.resize`.
  resize_antialias: boolean value if using antialias for `tf.image.resize`.
  random_erasing_prob: if applying random erasing. Default 0 for disabling, > 0 values set probability.
  magnitude: randaug magnitude.
  num_layers: randaug num_layers.
  augment_kwargs: randaug kwargs. Too many to list them all.

Returns: train_dataset, test_dataset, total_images, num_classes, steps_per_epoch
"""
