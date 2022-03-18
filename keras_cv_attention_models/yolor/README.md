# ___Keras YOLOR___
***

## Summary
  - Keras implementation of [Github WongKinYiu/yolor](https://github.com/WongKinYiu/yolor). Model weights converted from official publication.
  - [Paper 2105.04206 You Only Learn One Representation: Unified Network for Multiple Tasks](https://arxiv.org/pdf/2105.04206.pdf).
  - Model ouputs are slightly modified for better compiling with already existing implementations.
    - `yolor_head` output changed from PyTorch channel_first compatible format `[batch, 3, height, width, 85 preds]` to `[batch, height, width, 3, 85 preds]`.
    - `85 preds` from `yolor_head` output changed from `[bboxes, object_scores, class_scores]` to `[bboxes, class_scores, object_scores]`.
    - `bboxes` format changed from `[left, top, right, bottom]` to `[top, left, bottom, right]`.
  - **Currently, training only supporting `anchor_free_mode`: `CUDA_VISIBLE_DEVICES='0' ./coco_train_script.py --det_header yolor.YOLOR --use_anchor_free_mode`. Yolor training strategy is under working**.
## Models
  | Model      | Params | Image resolution | COCO val AP | Download |
  | ---------- | ------ | ---------------- | ----------- | -------- |
  | YOLOR_CSP  | 52.9M  | 640              | 50.0        | [yolor_csp_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolor/yolor_csp_coco.h5) |
  | YOLOR_CSPX | 99.8M  | 640              | 51.5        | [yolor_csp_x_coco.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/yolor/yolor_csp_x_coco.h5) |
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
  - **Use dynamic input resolution** by set `input_shape=(None, None, 3)`.
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
  # keras channel_last compatible format to pytorch channel_first one
  keras_out_reorder = tf.transpose(tf.reshape(keras_out[:, :np.prod(torch_out[0].shape[:-1])], [1, -1, 3, 85]), [0, 2, 1, 3])
  keras_out_reorder = tf.reshape(keras_out_reorder, torch_out[0].shape)
  # [top, left, bottom, right, *class_scores, object_score] -> [left, top, right, bottom ,object_score, *class_scores]
  keras_out_reorder = tf.gather(keras_out_reorder, [1, 0, 3, 2, 84, *np.arange(4, 84)], axis=-1)
  print(f"{np.allclose(torch_out[0].detach().numpy(), keras_out_reorder, atol=1e-4) = }")
  # np.allclose(torch_out[0].detach().numpy(), keras_out_reorder, atol=1e-4) = True

  """ Decode """
  from keras_cv_attention_models import coco
  anchors = coco.get_yolor_anchors()
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
  print(f"{np.allclose(torch_decode.detach().numpy(), keras_out_decode, atol=1e-6) = }")
  # np.allclose(torch_decode.detach().numpy(), keras_out_decode, atol=1e-6) = True
  ```
***
