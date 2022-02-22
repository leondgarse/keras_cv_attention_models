# ___Keras YOLOX___
***

## Summary
  - Keras implementation of [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX). Model weights converted from official publication.
  - [Paper 2107.08430 YOLOX: Exceeding YOLO Series in 2021](https://arxiv.org/pdf/2107.08430.pdf).
  - Model ouputs are slightly modified for better compiling with already existing implementations. That `YOLOXHeader` output changed from `[bboxes, object_scores, class_scores]` to `[bboxes, class_scores, object_scores]`, and `bboxes` format changed from `[left, top, right, bottom]` to `[top, left, bottom, right]`.
## Models

  | Model     | Params | Image resolution | COCO test AP | Download |
  | --------- | ------ | ---------------- | ------------ | -------- |
  | YOLOXNano | 0.91M  | 416              | 25.8         | [yolox_nano_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolox/yolox_nano_coco.h5) |
  | YOLOXTiny | 5.06M  | 416              | 32.8         | [yolox_tiny_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolox/yolox_tiny_coco.h5) |
  | YOLOXS    | 9.0M   | 640              | 40.5         | [yolox_s_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolox/yolox_s_coco.h5) |
  | YOLOXM    | 25.3M  | 640              | 47.2         | [yolox_m_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolox/yolox_m_coco.h5) |
  | YOLOXL    | 54.2M  | 640              | 50.1         | [yolox_l_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolox/yolox_l_coco.h5) |
  | YOLOXX    | 99.1M  | 640              | 51.5         | [yolox_x_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolox/yolox_x_coco.h5) |
## Usage
  - **Basic usage**
    ```py
    from keras_cv_attention_models import yolox
    model = yolox.YOLOXS(pretrained="coco")

    # Run prediction
    from keras_cv_attention_models import test_images
    imm = test_images.dog_cat()
    preds = model(model.preprocess_input(imm))
    bboxs, lables, confidences = model.decode_predictions(preds)[0]

    # Show result
    from keras_cv_attention_models.coco import data
    data.show_image_with_bboxes(imm, bboxs, lables, confidences, num_classes=80)
    ```
    ![yoloxs_dog_cat](https://user-images.githubusercontent.com/5744524/154924798-79a69e1b-40f3-4ac9-895f-0a15dd4ca9b3.png)
  - **Use dynamic input resolution** by set `input_shape=(None, None, 3)`. Currently using `keras.layers.UpSampling2D` for upsampling, thus actual `input_shape` should be dividable by `32`.
    ```py
    from keras_cv_attention_models import yolox
    model = yolox.YOLOXTiny(input_shape=(None, None, 3), pretrained="coco")
    # >>>> Load pretrained from: ~/.keras/models/yolox_tiny_coco.h5
    print(model.input_shape, model.output_shape)
    # (None, None, None, 3) (None, None, 85)
    print(model(tf.ones([1, 768, 768, 3])).shape)
    # (1, 12096, 85)
    print(model(tf.ones([1, 160, 256, 3])).shape)
    # (1, 840, 85)

    from keras_cv_attention_models import test_images
    imm = test_images.dog_cat()
    input_shape = (320, 224, 3)
    preds = model(model.preprocess_input(imm, input_shape=input_shape))
    bboxs, lables, confidences = model.decode_predictions(preds, input_shape=input_shape)[0]

    # Show result
    from keras_cv_attention_models.coco import data
    data.show_image_with_bboxes(imm, bboxs, lables, confidences, num_classes=80)
    ```
    ![yoloxtiny_dynamic_dog_cat](https://user-images.githubusercontent.com/5744524/154925006-0c5e1034-5d34-4762-937f-a914b4810a77.png)
## Custom detector using YOLOX header
  - `Backbone` for `YOLOX` can be any model with pyramid stage structure. Default `width_mul=-1` means using `min([ii.shape[-1] for ii in features]) / 256`.
    ```py
    from keras_cv_attention_models import efficientnet, yolox
    bb = efficientnet.EfficientNetV2B1(input_shape=(256, 256, 3), num_classes=0)
    mm = yolox.YOLOX(backbone=bb)
    # >>>> features: {'stack_2_block2_output': (None, 32, 32, 48),
    #                 'stack_4_block5_output': (None, 16, 16, 112),
    #                 'stack_5_block8_output': (None, 8, 8, 192)}
    # >>>> width_mul: 0.1875

    mm.summary()  # Trainable params: 7,762,115
    ```
## Verification with PyTorch version
  ```py
  inputs = np.random.uniform(size=(1, 640, 640, 3)).astype("float32")

  """ PyTorch yolox_s """
  sys.path.append('../YOLOX/')
  from exps.default import yolox_s as torch_yolox
  import torch
  yolo_model = torch_yolox.Exp()
  torch_model = yolo_model.get_model()
  _ = torch_model.eval()
  weight = torch.load('yolox_s.pth', map_location=torch.device('cpu'))["model"]
  torch_model.load_state_dict(weight)
  torch_model.head.decode_in_inference = False
  torch_out = torch_model(torch.from_numpy(inputs).permute(0, 3, 1, 2))

  """ Keras YOLOXS """
  from keras_cv_attention_models.yolox import yolox
  mm = yolox.YOLOXS(pretrained="coco")
  keras_out = mm(inputs).numpy()
  # [top, left, bottom, right, *class_scores, object_score] -> [left, top, right, bottom ,object_score, *class_scores]
  keras_out_reorder = np.concatenate([keras_out[:, :, [1, 0, 3, 2, -1]], keras_out[:, :, 4:-1]], axis=-1)

  """ Model outputs verification """
  print(f"{np.allclose(torch_out.detach().numpy(), keras_out_reorder, atol=1e-4) = }")
  # np.allclose(torch_out.detach().numpy(), keras_out, atol=1e-4) = True

  """ Torch decode """
  torch_decode = torch_model.head.decode_outputs(torch_out, torch_out.dtype).detach().numpy()[0]
  hh, ww = inputs.shape[1], inputs.shape[2]
  torch_center, torch_wh = torch_decode[:, :2] / [hh, ww], torch_decode[:, 2:4] / [hh, ww]
  torch_decode = np.concatenate([torch_center - 0.5 * torch_wh, torch_center + 0.5 * torch_wh, torch_decode[:, 4:]], axis=-1)

  """ Keras decode """
  from keras_cv_attention_models import coco
  anchors = coco.get_anchors(mm.input_shape[1:-1], pyramid_levels=[3, 5], aspect_ratios=[1], num_scales=1, anchor_scale=1, grid_zero_start=True)
  keras_decode = coco.decode_bboxes(keras_out[0], anchors=anchors).numpy()

  # [top, left, bottom, right, *class_scores, object_score] -> [left, top, right, bottom ,object_score, *class_scores]
  keras_decode = np.concatenate([keras_decode[:, [1, 0, 3, 2, -1]], keras_decode[:, 4:-1]], axis=-1)

  """ Decode verification """
  print(f"{np.allclose(torch_decode, keras_decode, atol=1e-4) = }")
  # np.allclose(torch_decode, keras_decode, atol=1e-4) = True
  ```
