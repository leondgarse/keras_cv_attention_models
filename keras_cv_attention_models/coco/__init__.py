from keras_cv_attention_models import backend

from keras_cv_attention_models.coco import eval_func, anchors_func
from keras_cv_attention_models.coco.eval_func import DecodePredictions, COCOEvalCallback
from keras_cv_attention_models.coco.anchors_func import (
    get_anchors_mode_parameters,
    get_anchors,
    get_anchor_free_anchors,
    get_yolor_anchors,
    get_anchors_mode_by_anchors,
    get_pyramid_levels_by_anchors,
    decode_bboxes,
)

if backend.is_tensorflow_backend:
    from keras_cv_attention_models.coco import data, losses
    from keras_cv_attention_models.coco.data import (
        COCO_80_LABEL_DICT,
        COCO_90_LABEL_DICT,
        COCO_80_to_90_LABEL_DICT,
        aspect_aware_resize_and_crop_image,
        init_mean_std_by_rescale_mode,
        init_dataset,
        show_image_with_bboxes,
        show_batch_sample
    )

    data.init_dataset.__doc__ = """ Init dataset by name.
    Args:
      data_name: the registered dataset name from `tensorflow_datasets`.
      input_shape: input shape.
      batch_size: batch size.
      buffer_size: dataset shuffle buffer size.
      info_only: boolean value if returns dataset info only.
      max_labels_per_image: .
      anchors_mode: .
      anchor_pyramid_levels: .
      anchor_aspect_ratios: .
      anchor_num_scales: .
      anchor_scale: .
      anchor_scale: .
      cutmix_alpha: cutmix applying probability.
      rescale_mode: one of ["tf", "torch", "raw01", "raw"]. Detail in `data.init_mean_std_by_rescale_mode`. Or specific `(mean, std)` like `(128.0, 128.0)`.
      random_crop_mode: .
      mosaic_mix_prob: .
      resize_method: one of ["nearest", "bilinear", "bicubic"]. Resize method for `tf.image.resize`.
      resize_antialias: boolean value if using antialias for `tf.image.resize`.
      magnitude: randaug magnitude.
      num_layers: randaug num_layers.
      augment_kwargs: randaug kwargs. Too many to list them all.

    Returns: train_dataset, test_dataset, total_images, num_classes, steps_per_epoch
    """
