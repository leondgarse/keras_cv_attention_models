# ___Keras EfficientDet___
***

## Summary
  - Keras implementation of [google/automl/efficientdet](https://github.com/google/automl/tree/master/efficientdet). Model weights converted from official publication. It's the `h5` ones for `EfficientDetD*` models, not `ckpt` ones, as their accuracy higher.
  - [Paper 1911.09070 EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/pdf/1911.09070.pdf).
  - `Det-AdvProp + AutoAugment` [Paper 2103.13886 Robust and Accurate Object Detection via Adversarial Learning](https://arxiv.org/pdf/2103.13886.pdf).

  ![](https://user-images.githubusercontent.com/5744524/151656702-9fb68cf6-e4ce-42b5-a488-80807cc66e56.png)
***

## Models
  | Model              | Params | FLOPs   | Input | COCO val AP | test AP | Download |
  | ------------------ | ------ | ------- | ----- | ----------- | ------- | -------- |
  | EfficientDetD0     | 3.9M   | 2.55G   | 512   | 34.3        | 34.6    | [efficientdet_d0.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d0_512_coco.h5)         |
  | - Det-AdvProp      | 3.9M   | 2.55G   | 512   | 35.1        | 35.3    |          |
  | EfficientDetD1     | 6.6M   | 6.13G   | 640   | 40.2        | 40.5    | [efficientdet_d1.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d1_640_coco.h5)         |
  | - Det-AdvProp      | 6.6M   | 6.13G   | 640   | 40.8        | 40.9    |          |
  | EfficientDetD2     | 8.1M   | 11.03G  | 768   | 43.5        | 43.9    | [efficientdet_d2.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d2_768_coco.h5)         |
  | - Det-AdvProp      | 8.1M   | 11.03G  | 768   | 44.3        | 44.3    |          |
  | EfficientDetD3     | 12.0M  | 24.95G  | 896   | 46.8        | 47.2    | [efficientdet_d3.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d3_896_coco.h5)         |
  | - Det-AdvProp      | 12.0M  | 24.95G  | 896   | 47.7        | 48.0    |          |
  | EfficientDetD4     | 20.7M  | 55.29G  | 1024  | 49.3        | 49.7    | [efficientdet_d4.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d4_1024_coco.h5)        |
  | - Det-AdvProp      | 20.7M  | 55.29G  | 1024  | 50.4        | 50.4    |          |
  | EfficientDetD5     | 33.7M  | 135.62G | 1280  | 51.2        | 51.5    | [efficientdet_d5.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d5_1280_coco.h5)        |
  | - Det-AdvProp      | 33.7M  | 135.62G | 1280  | 52.2        | 52.5    |          |
  | EfficientDetD6     | 51.9M  | 225.93G | 1280  | 52.1        | 52.6    | [efficientdet_d6.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d6_1280_coco.h5)        |
  | EfficientDetD7     | 51.9M  | 325.34G | 1536  | 53.4        | 53.7    | [efficientdet_d7.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d7_1536_coco.h5)        |
  | EfficientDetD7X    | 77.0M  | 410.87G | 1536  | 54.4        | 55.1    | [efficientdet_d7x.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d7x_1536_coco.h5)      |
  | EfficientDetLite0  | 3.2M   | 0.98G   | 320   | 27.5        | 26.41   | [efficientdet_lite0.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_lite0_320_coco.h5)   |
  | EfficientDetLite1  | 4.2M   | 1.97G   | 384   | 32.6        | 31.50   | [efficientdet_lite1.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_lite1_384_coco.h5)   |
  | EfficientDetLite2  | 5.3M   | 3.38G   | 448   | 36.2        | 35.06   | [efficientdet_lite2.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_lite2_448_coco.h5)   |
  | EfficientDetLite3  | 8.4M   | 7.50G   | 512   | 39.9        | 38.77   | [efficientdet_lite3.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_lite3_512_coco.h5)   |
  | EfficientDetLite3X | 9.3M   | 14.01G  | 640   | 44.0        | 42.64   | [efficientdet_lite3x.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_lite3x_640_coco.h5) |
  | EfficientDetLite4  | 15.1M  | 20.20G  | 640   | 44.4        | 43.18   | [efficientdet_lite4.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_lite4_640_coco.h5)   |

  **COCO val evaluation**. More usage info can be found in [COCO Evaluation](https://github.com/leondgarse/keras_cv_attention_models/tree/main/keras_cv_attention_models/coco#evaluation).
  ```py
  # EfficientDet using resize method bilinear w/o antialias
  CUDA_VISIBLE_DEVICES='0' ./coco_eval_script.py -m efficientdet.EfficientDetD0 --resize_method bilinear --disable_antialias
  # >>>> [COCOEvalCallback] input_shape: (512, 512), pyramid_levels: [3, 7], anchors_mode: efficientdet

  # Specify anchor_scale if not 4
  CUDA_VISIBLE_DEVICES='0' ./coco_eval_script.py -m efficientdet.EfficientDetLite0 --anchor_scale 3 --resize_method bilinear --disable_antialias
  # >>>> model_kwargs: {'anchor_scale': 3}
  # >>>> [COCOEvalCallback] input_shape: (320, 320), pyramid_levels: [3, 7], anchors_mode: efficientdet
  ```
## Usage
  - **Basic usage**
    ```py
    from keras_cv_attention_models import efficientdet
    model = efficientdet.EfficientDetD0(pretrained="coco")

    # Run prediction
    from keras_cv_attention_models import test_images
    imm = test_images.dog_cat()
    preds = model(model.preprocess_input(imm))
    bboxs, lables, confidences = model.decode_predictions(preds)[0]

    # Show result
    from keras_cv_attention_models.coco import data
    data.show_image_with_bboxes(imm, bboxs, lables, confidences, num_classes=90)
    ```
    ![effdetd0_dog_cat](https://user-images.githubusercontent.com/5744524/151114104-b8e0d625-66b5-4ccd-89cb-fe3c47dbf1a7.png)
  - **Use dynamic input resolution** by set `input_shape=(None, None, 3)`.
    ```py
    from keras_cv_attention_models import efficientdet
    model = efficientdet.EfficientDetD1(input_shape=(None, None, 3), pretrained="coco")
    # >>>> Load pretrained from: ~/.keras/models/efficientdet_d1_640_coco.h5
    print(model.input_shape, model.output_shape)
    # (None, None, None, 3) (None, None, 94)
    print(model(tf.ones([1, 768, 768, 3])).shape)
    # (1, 110484, 94)
    print(model(tf.ones([1, 176, 228, 3])).shape)
    # (1, 7803, 94)

    from keras_cv_attention_models import test_images
    imm = test_images.dog_cat()
    input_shape = (376, 227, 3)
    preds = model(model.preprocess_input(imm, input_shape=input_shape))
    bboxs, lables, confidences = model.decode_predictions(preds, input_shape=input_shape)[0]

    # Show result
    from keras_cv_attention_models.coco import data
    data.show_image_with_bboxes(imm, bboxs, lables, confidences, num_classes=90)
    ```
    ![effdetd1_dynamic_dog_cat](https://user-images.githubusercontent.com/5744524/153983911-2299efad-3b42-46b9-88c8-92c3b6e4e091.png)
## Custom detector using efficientdet header
  - **Backbone** for `EfficientDet` can be any model with pyramid stage structure.
    ```py
    from keras_cv_attention_models import efficientdet, coatnet
    bb = coatnet.CoAtNet0(input_shape=(256, 256, 3), num_classes=0)
    mm = efficientdet.EfficientDet(backbone=bb, num_classes=80)
    # >>>> features: {'stack_2_block_3_output': (None, 32, 32, 192),
    #                 'stack_3_block_5_ffn_output': (None, 16, 16, 384),
    #                 'stack_4_block_2_ffn_output': (None, 8, 8, 768)}

    mm.summary()  # Trainable params: 23,487,463
    print(mm.output_shape)
    # (None, 12276, 84)
    ```
    Each `EfficientDetD*` / `EfficientDetLite*` can also set with `backbone=xxx` for using their presets.
    ```py
    from keras_cv_attention_models import efficientdet, resnest
    backbone = resnest.ResNest50(input_shape=(384, 384, 3), num_classes=0)
    mm = efficientdet.EfficientDetD2(backbone=backbone, pretrained=None, num_classes=80)

    mm.summary()  # Trainable params: 27,142,867
    ```
  - Currently 3 types anchors supported, parameter **`anchors_mode`** controls which anchor to use, value in `["efficientdet", "anchor_free", "yolor"]`. Default is `"efficientdet"`.
    ```py
    from keras_cv_attention_models import efficientdet, coatnet
    bb = coatnet.CoAtNet0(input_shape=(256, 256, 3), num_classes=0)

    mm = efficientdet.EfficientDet(backbone=bb, anchors_mode="anchor_free", num_classes=80) # Trainable params: 23,444,424
    print(mm.output_shape) # (None, 1364, 85)

    mm = efficientdet.EfficientDet(backbone=bb, anchors_mode="yolor", num_classes=80) # Trainable params: 23,455,474
    print(mm.output_shape) # (None, 4096, 85)
    ```
    **Default settings for anchors_mode**

    | anchors_mode | use_object_scores | num_anchors | anchor_scale | aspect_ratios | num_scales | grid_zero_start |
    | ------------ | ----------------- | ----------- | ------------ | ------------- | ---------- | --------------- |
    | efficientdet | False             | 9           | 4            | [1, 2, 0.5]   | 3          | False           |
    | anchor_free  | True              | 1           | 1            | [1]           | 1          | True            |
    | yolor        | True              | 3           | None         | presets       | None       | offset=0.5      |
***
