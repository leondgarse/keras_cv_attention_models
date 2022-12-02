# ___Keras YOLOV7___
***

## Summary
  - Keras implementation of [Github WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7). Model weights converted from official publication.
  - [Paper 2207.02696 YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/pdf/2207.02696.pdf).
***
## Models
  | Model       | Params | FLOPs  | Input | COCO val AP | test AP | Download |
  | ----------- | ------ | ------ | ----- | ----------- | ------- | -------- |
  | YOLOV7_Tiny | 6.23M  | 2.90G  | 416   | 33.3        |         | [yolov7_tiny_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov7/yolov7_tiny_coco.h5) |
  | YOLOV7_CSP  | 37.67M | 53.0G  | 640   | 51.4        |         | [yolov7_csp_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov7/yolov7_csp_coco.h5) |
  | YOLOV7_X    | 71.41M | 95.0G  | 640   | 53.1        |         | [yolov7_x_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov7/yolov7_x_coco.h5) |
  | YOLOV7_W6   | 70.49M | 180.1G | 1280  | 54.9        |         | [yolov7_w6_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov7/yolov7_w6_coco.h5) |
  | YOLOV7_E6   | 97.33M | 257.6G | 1280  | 56.0        |         | [yolov7_e6_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov7/yolov7_e6_coco.h5) |
  | YOLOV7_D6   | 133.9M | 351.4G | 1280  | 56.6        |         | [yolov7_d6_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov7/yolov7_d6_coco.h5) |
  | YOLOV7_E6E  | 151.9M | 421.7G | 1280  | 56.8        |         | [yolov7_e6e_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolov7/yolov7_e6e_coco.h5) |

## Usage
  - **Basic usage**
    ```py
    from keras_cv_attention_models.yolov7 import yolov7
    model = yolov7.YOLOV7_CSP(pretrained="coco")

    # Run prediction
    from keras_cv_attention_models import test_images
    imm = test_images.dog_cat()
    preds = model(model.preprocess_input(imm))
    bboxs, lables, confidences = model.decode_predictions(preds)[0]

    # Show result
    from keras_cv_attention_models.coco import data
    data.show_image_with_bboxes(imm, bboxs, lables, confidences)
    ```
    ![yolov7_csp_dog_cat](https://user-images.githubusercontent.com/5744524/204136183-bc7c46cd-6595-441b-995d-72a8974893a4.png)
  - **Use dynamic input resolution** by set `input_shape=(None, None, 3)`. **Note: `YOLOV7_*6` models using `focus_stem` requires input at least been an even number**.
    ```py
    from keras_cv_attention_models.yolov7 import yolov7
    model = yolov7.YOLOV7_CSP(input_shape=(None, None, 3), pretrained="coco")
    # >>>> Load pretrained from: ~/.keras/models/yolov7_csp_coco.h5
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
    ![yolov7_csp_dynamic_dog_cat](https://user-images.githubusercontent.com/5744524/204529451-25656b67-6e78-4daa-b385-3f48b8c8fb17.png)
## Custom detector using YOLOV7 header
  - **Backbone** for `YOLOV7` can be any model with pyramid stage structure.
    ```py
    from keras_cv_attention_models import efficientnet, yolov7
    bb = efficientnet.EfficientNetV2B1(input_shape=(256, 256, 3), num_classes=0)
    mm = yolov7.YOLOV7(backbone=bb)
    # >>>> features: {'stack_2_block2_output': (None, 32, 32, 48),
    #                 'stack_4_block5_output': (None, 16, 16, 112),
    #                 'stack_5_block8_output': (None, 8, 8, 192)}

    mm.summary()  # Trainable params: 12,268,473
    print(mm.output_shape)
    # (None, 4032, 85)
    ```
  - Currently 3 types anchors supported, parameter **`anchors_mode`** controls which anchor to use, value in `["efficientdet", "anchor_free", "yolor"]`. Default is `"yolor"`.
    ```py
    from keras_cv_attention_models import efficientnet, yolov7
    bb = efficientnet.EfficientNetV2B1(input_shape=(256, 256, 3), num_classes=0)

    mm = yolov7.YOLOV7(backbone=bb, anchors_mode="anchor_free") # Trainable params: 12,213,563
    print(mm.output_shape) # (None, 1344, 85)

    mm = yolov7.YOLOV7(backbone=bb, anchors_mode="efficientdet") # Trainable params: 12,430,296
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

  """ PyTorch yolov7 """
  sys.path.append('../yolov7')
  import torch
  from models.experimental import Ensemble

  model = Ensemble()
  weight = 'yolov7.pt'
  ckpt = torch.load(weight, map_location="cpu")  # load
  model.append(ckpt['ema' if ckpt.get('ema') else 'model'].float().eval())
  for sub in model.modules():
      if type(sub) is torch.nn.Upsample:
          sub.recompute_scale_factor = None  # torch 1.11.0 compatibility
  model[-1].model[-1].export = True
  torch_out = model[-1].forward_once(torch.from_numpy(inputs).permute([0, 3, 1, 2]))

  """ Keras YOLOV7_CSP """
  from keras_cv_attention_models.yolov7 import yolov7
  mm = yolov7.YOLOV7_CSP(pretrained='coco', classifier_activation=None)
  keras_out = mm(inputs)

  """ Model outputs verification """
  # keras channel_last compatible format to pytorch channel_first one, torch_out[0].shape == [1, 3, 80, 80, 85]
  keras_out_reorder = tf.transpose(tf.reshape(keras_out[:, :np.prod(torch_out[0].shape[:-1])], [1, -1, 3, 85]), [0, 2, 1, 3])
  keras_out_reorder = tf.reshape(keras_out_reorder, torch_out[0].shape)
  # [top, left, bottom, right, *class_scores, object_score] -> [left, top, right, bottom ,object_score, *class_scores]
  keras_out_reorder = tf.gather(keras_out_reorder, [1, 0, 3, 2, 84, *np.arange(4, 84)], axis=-1)
  print(f"{np.allclose(torch_out[0].detach().numpy(), keras_out_reorder, atol=1e-4) = }")
  # np.allclose(torch_out[0].detach().numpy(), keras_out_reorder, atol=1e-4) = True
  ```
