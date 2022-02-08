import tensorflow_datasets as tfds
ds, info = tfds.load('coco/2017', with_info=True)
aa = ds['validation']
print(aa.element_spec.keys())
datapoint = aa.as_numpy_iterator().next()

# from keras_cv_attention_models import efficientdet
# model = efficientdet.EfficientDetD0()
import hparams_config
import inference
from tf2 import efficientdet_keras

config = hparams_config.get_efficientdet_config('efficientdet-d0')
# config.nms_configs.score_thresh = 0.4
# model = efficientdet_keras.EfficientDetNet(config=config)
model = efficientdet_keras.EfficientDetModel(config=config)
model.build((None, 512, 512, 3))
model.load_weights(tf.train.latest_checkpoint("efficientdet-d0/"))
model.summary(expand_nested=True)

# rescale_mode, target_shape, resize_method, resize_antialias = "torch", model.input_shape[1:-1], "bilinear", False
rescale_mode, resize_method, resize_antialias, score_threshold = "torch", "bilinear", False, 0.001

def resize_and_pad_image(image, target_shape, method="bilinear", antialias=False):
    scale_hh, scale_ww = target_shape[0] / image.shape[0], target_shape[1] / image.shape[1]
    scale = tf.minimum(scale_hh, scale_ww)
    scaled_hh, scaled_ww = int(image.shape[0] * scale), int(image.shape[1] * scale)
    scaled_image = tf.image.resize(image, [scaled_hh, scaled_ww], method=method, antialias=antialias)
    output_image = tf.image.pad_to_bounding_box(scaled_image, 0, 0, target_shape[0], target_shape[1])
    return output_image, scale

def model_interf_1(model, input_image, rescale_mode="torch", resize_method="bilinear", resize_antialias=False, score_threshold=0.001):
    from keras_cv_attention_models.coco import data
    mean, std = data.init_mean_std_by_rescale_mode(rescale_mode)

    target_shape = model.input_shape[1:-1]
    # input_image = tf.image.resize(input_image, target_shape, method=resize_method, antialias=resize_antialias)
    processed_image, scale = resize_and_pad_image(input_image, target_shape, method=resize_method, antialias=resize_antialias)
    processed_image = (tf.cast(processed_image, tf.float32) - mean) / std
    preds = model(tf.expand_dims(processed_image, 0))
    bboxes, labels, confidences = model.decode_predictions(preds, score_threshold=score_threshold)[0]
    num_classes = preds.shape[-1] - 4

    # height, width = input_image.shape[:2]
    # bboxes *= [height, width, height, width]
    height, width = target_shape[0] / scale, target_shape[1] / scale
    bboxes *= [height, width, height, width]
    # [top, left, bottom, right] -> [top, left, height, width]
    bboxes = np.hstack([bboxes[:, :2], bboxes[:, 2:] - bboxes[:, :2]])
    # [top, left, height, width] -> [left, top, width, height]
    bboxes = bboxes[:, [1, 0, 3, 2]]
    labels = (labels + 1) if num_classes == 90 else [data.COCO_80_to_90_LABEL_DICT[ii] + 1 for ii in labels]
    return bboxes, labels, confidences

def model_interf(model, input_image, rescale_mode="torch", resize_method="bilinear", resize_antialias=False, score_threshold=0.001):
    bboxes, confidences, labels, valid_len = model(tf.expand_dims(input_image, 0))
    bboxes, confidences, labels = bboxes[0][:valid_len[0]].numpy(), confidences[0][:valid_len[0]].numpy(), labels[0][:valid_len[0]].numpy()
    bboxes = np.hstack([bboxes[:, :2], bboxes[:, 2:] - bboxes[:, :2]])  # [top, left, bottom, right] -> [top, left, height, width]
    bboxes = bboxes[:, [1, 0, 3, 2]]    # [top, left, height, width] -> [left, top, width, height]
    return bboxes, labels, confidences

from tqdm import tqdm
results = []
# Format: [image_id, x, y, width, height, score, class]
to_coco_eval_single = lambda image_id, bbox, label, score: [image_id, *bbox.tolist(), score, label]
for datapoint in tqdm(aa):
    image_id = int(datapoint['image/id'])
    bboxes, labels, confidences = model_interf(model, datapoint["image"], rescale_mode, resize_method, resize_antialias, score_threshold)
    results.extend([to_coco_eval_single(image_id, bbox, label, score) for bbox, label, score in zip(bboxes, labels, confidences)])

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

url = "https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/coco_annotations_instances_val2017.json"
annotation_file = keras.utils.get_file(origin=url)
coco_gt = COCO(annotation_file)
detections = np.array(results)
image_ids = list(set(detections[:, 0]))
coco_dt = coco_gt.loadRes(detections)
coco_eval = COCOeval(coco_gt, coco_dt, iouType='bbox')
coco_eval.params.imgIds = image_ids
coco_eval.evaluate()
coco_eval.accumulate()
coco_eval.summarize()
