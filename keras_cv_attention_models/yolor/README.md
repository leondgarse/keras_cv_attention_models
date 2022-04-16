# ___Keras YOLOR___
***

## Summary
  - Keras implementation of [Github WongKinYiu/yolor](https://github.com/WongKinYiu/yolor). Model weights converted from official publication.
  - [Paper 2105.04206 You Only Learn One Representation: Unified Network for Multiple Tasks](https://arxiv.org/pdf/2105.04206.pdf).
  - Model ouputs are slightly modified for better compiling with already existing implementations.
    - `yolor_head` output changed from PyTorch channel_first compatible format `[batch, 3, height, width, 85 preds]` to `[batch, height, width, 3, 85 preds]`.
    - `85 preds` from `yolor_head` output changed from `[bboxes, object_scores, class_scores]` to `[bboxes, class_scores, object_scores]`.
    - `bboxes` format changed from `[left, top, right, bottom]` to `[top, left, bottom, right]`.
## Models
  - `YOLOR_E6` and `YOLOR_D6` weights are from [Github WongKinYiu/yolor/tree/paper](https://github.com/WongKinYiu/yolor/tree/paper), which is using `batchnorm + activation` after concatenating `short` and `deep` branch in `csp_stack`, different from others.

  | Model      | Params | FLOPs   | Input | COCO val AP | test AP | Download |
  | ---------- | ------ | ------- | ----- | ----------- | ------- | -------- |
  | YOLOR_CSP  | 52.9M  | 60.25G  | 640   | 50.0        | 52.8    | [yolor_csp_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolor/yolor_csp_coco.h5)     |
  | YOLOR_CSPX | 99.8M  | 111.11G | 640   | 51.5        | 54.8    | [yolor_csp_x_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolor/yolor_csp_x_coco.h5) |
  | YOLOR_P6   | 37.3M  | 162.87G | 1280  | 52.5        | 55.7    | [yolor_p6_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolor/yolor_p6_coco.h5)       |
  | YOLOR_W6   | 79.9M  | 226.67G | 1280  | 53.6 ?      | 56.9    | [yolor_w6_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolor/yolor_w6_coco.h5)       |
  | YOLOR_E6   | 115.9M | 341.62G | 1280  | 50.3 ?      | 57.6    | [yolor_e6_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolor/yolor_e6_coco.h5)       |
  | YOLOR_D6   | 151.8M | 467.88G | 1280  | 50.8 ?      | 58.2    | [yolor_d6_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolor/yolor_d6_coco.h5)       |

  **COCO val evaluation results**. More usage info can be found in [COCO Evaluation](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/coco#evaluation).
  ```sh
  CUDA_VISIBLE_DEVICES='0' ./coco_eval_script.py -m yolor.YOLOR_P6 --nms_method hard --nms_iou_or_sigma 0.65 \
  --nms_max_output_size 300 --nms_topk -1 --letterbox_pad 64 --input_shape 1344
  ```
  | Model    | letterbox_pad | input_shape | COCO Val AP 0.50:0.95, area=all |
  | -------- | ------------- | ----------- | ------------------------------- |
  | YOLOR_P6 | 0             | 1280        | 0.521                           |
  | YOLOR_P6 | 64            | 1344        | 0.526                           |
  | YOLOR_W6 | 0             | 1280        | 0.530                           |
  | YOLOR_W6 | 64            | 1344        | 0.536                           |
  | YOLOR_W6 | 128           | 1408        | 0.536                           |
  | YOLOR_E6 | 64            | 1344        | 0.503                           |
  | YOLOR_D6 | 64            | 1344        | 0.508                           |
## Usage
  - **Basic usage**
    ```py
    from keras_cv_attention_models import yolor
    model = yolor.YOLOR_CSP(pretrained="coco")

    # Run prediction
    from keras_cv_attention_models import test_images
    imm = test_images.dog_cat()
    preds = model(model.preprocess_input(imm))
    bboxs, lables, confidences = model.decode_predictions(preds)[0]

    # Show result
    from keras_cv_attention_models.coco import data
    data.show_image_with_bboxes(imm, bboxs, lables, confidences)
    ```
    ![yolor_csp_dog_cat](https://user-images.githubusercontent.com/5744524/158940187-1840ab4f-2f0e-497b-b796-2bdb9f31755a.png)
  - **Use dynamic input resolution** by set `input_shape=(None, None, 3)`. **Note: `YOLO_*6` models using `focus_stem` requires input at least been an even number**.
    ```py
    from keras_cv_attention_models import yolor
    model = yolor.YOLOR_CSP(input_shape=(None, None, 3), pretrained="coco")
    # >>>> Load pretrained from: ~/.keras/models/yolor_csp_coco.h5
    print(model.input_shape, model.output_shape)
    # (None, None, None, 3) (None, None, 85)
    print(model(tf.ones([1, 742, 355, 3])).shape)
    # (1, 16662, 85)
    print(model(tf.ones([1, 188, 276, 3])).shape)
    # (1, 3330, 85)

    from keras_cv_attention_models import test_images
    imm = test_images.dog_cat()
    input_shape = (320, 224, 3)
    preds = model(model.preprocess_input(imm, input_shape=input_shape))
    bboxs, lables, confidences = model.decode_predictions(preds, input_shape=input_shape)[0]

    # Show result
    from keras_cv_attention_models.coco import data
    data.show_image_with_bboxes(imm, bboxs, lables, confidences, num_classes=80)
    ```
    ![yolor_csp_dynamic_dog_cat](https://user-images.githubusercontent.com/5744524/158940195-be958f00-8c6f-4ca9-a5dc-562a9690cad8.png)
## Custom detector using YOLOR header
  - **Backbone** for `YOLOR` can be any model with pyramid stage structure.
    ```py
    from keras_cv_attention_models import efficientnet, yolor
    bb = efficientnet.EfficientNetV2B1(input_shape=(256, 256, 3), num_classes=0)
    mm = yolor.YOLOR(backbone=bb)
    # >>>> features: {'stack_2_block2_output': (None, 32, 32, 48),
    #                 'stack_4_block5_output': (None, 16, 16, 112),
    #                 'stack_5_block8_output': (None, 8, 8, 192)}

    mm.summary()  # Trainable params: 8,710,574
    print(mm.output_shape)
    # (None, 4032, 85)
    ```
  - Currently 3 types anchors supported, parameter **`anchors_mode`** controls which anchor to use, value in `["efficientdet", "anchor_free", "yolor"]`. Default is `"yolor"`.
    ```py
    from keras_cv_attention_models import efficientnet, yolor
    bb = efficientnet.EfficientNetV2B1(input_shape=(256, 256, 3), num_classes=0)

    mm = yolor.YOLOR(backbone=bb, anchors_mode="anchor_free") # Trainable params: 8,617,074
    print(mm.output_shape) # (None, 1344, 85)

    mm = yolor.YOLOR(backbone=bb, anchors_mode="efficientdet") # Trainable params: 8,986,124
    print(mm.output_shape) # (None, 12096, 84)
    ```
    **Default settings for anchors_mode**

    | anchors_mode | use_object_scores | num_anchors | anchor_scale | aspect_ratios | num_scales | grid_zero_start |
    | ------------ | ----------------- | ----------- | ------------ | ------------- | ---------- | --------------- |
    | efficientdet | False             | 9           | 4            | [1, 2, 0.5]   | 3          | False           |
    | anchor_free  | True              | 1           | 1            | [1]           | 1          | True            |
    | yolor        | True              | 3           | None         | presets       | None       | offset=0.5      |
## Verification with PyTorch version
  ```py
  inputs = np.random.uniform(size=(1, 640, 640, 3)).astype("float32")

  """ PyTorch yolor_csp """
  sys.path.append('../yolor/')
  from models import models as torch_yolor
  import torch

  torch_model = torch_yolor.Darknet('../yolor/cfg/yolor_csp.cfg', [640, 640])
  weights = torch.load("../yolor/yolor_csp_star.pt", map_location=torch.device('cpu'))['model']
  torch_model.load_state_dict(weights)
  _ = torch_model.eval()
  torch_decode, torch_out = torch_model.forward_once(torch.from_numpy(inputs).permute([0, 3, 1, 2]))

  """ Keras YOLOR_CSP """
  from keras_cv_attention_models.yolor import yolor
  mm = yolor.YOLOR_CSP(classifier_activation=None)
  keras_out = mm(inputs)

  """ Model outputs verification """
  # keras channel_last compatible format to pytorch channel_first one, torch_out[0].shape == [1, 3, 80, 80, 85]
  keras_out_reorder = tf.transpose(tf.reshape(keras_out[:, :np.prod(torch_out[0].shape[:-1])], [1, -1, 3, 85]), [0, 2, 1, 3])
  keras_out_reorder = tf.reshape(keras_out_reorder, torch_out[0].shape)
  # [top, left, bottom, right, *class_scores, object_score] -> [left, top, right, bottom ,object_score, *class_scores]
  keras_out_reorder = tf.gather(keras_out_reorder, [1, 0, 3, 2, 84, *np.arange(4, 84)], axis=-1)
  print(f"{np.allclose(torch_out[0].detach().numpy(), keras_out_reorder, atol=1e-4) = }")
  # np.allclose(torch_out[0].detach().numpy(), keras_out_reorder, atol=1e-4) = True

  """ Decode """
  from keras_cv_attention_models import coco
  anchors = coco.get_yolor_anchors(input_shape=mm.input_shape[1:-1])
  keras_out_decode = tf.sigmoid(keras_out)
  center_yx = (keras_out_decode[:, :, :2] * 2 * anchors[:, 4:] + anchors[:, :2])
  hhww = ((keras_out_decode[:, :, 2:4] * 2) ** 2 * anchors[:, 2:4])

  # keras channel_last compatible format to pytorch channel_first one
  center_yx = tf.split(center_yx, [80 * 80 * 3, 40 * 40 * 3, 20 * 20 * 3], axis=1)
  center_yx = tf.concat([tf.reshape(tf.transpose(tf.reshape(ii, [1, -1, 3, 2]), [0, 2, 1, 3]), [1, -1, 2]) for ii in center_yx], axis=1)
  hhww = tf.split(hhww, [80 * 80 * 3, 40 * 40 * 3, 20 * 20 * 3], axis=1)
  hhww = tf.concat([tf.reshape(tf.transpose(tf.reshape(ii, [1, -1, 3, 2]), [0, 2, 1, 3]), [1, -1, 2]) for ii in hhww], axis=1)
  labels = tf.split(keras_out_decode[:, :, 4:], [80 * 80 * 3, 40 * 40 * 3, 20 * 20 * 3], axis=1)
  labels = tf.concat([tf.reshape(tf.transpose(tf.reshape(ii, [1, -1, 3, 81]), [0, 2, 1, 3]), [1, -1, 81]) for ii in labels], axis=1)
  # [top, left, bottom, right, *class_scores, object_score] -> [left, top, right, bottom ,object_score, *class_scores]
  keras_out_decode = tf.gather(tf.concat([center_yx, hhww, labels], axis=-1), [1, 0, 3, 2, 84, *np.arange(4, 84)], axis=-1)

  """ Decode verification """
  # Rescale torch bboxes to [0, 1]
  torch_decode = torch.cat([torch_decode[:, :, :4] / 640, torch_decode[:, :, 4:]], axis=-1)
  print(f"{np.allclose(torch_decode.detach().numpy(), keras_out_decode, atol=1e-5) = }")
  # np.allclose(torch_decode.detach().numpy(), keras_out_decode, atol=1e-5) = True
  ```
***
