# ___Keras EfficientDet___
***

## Summary
  - Keras implementation of [google/automl/efficientdet](https://github.com/google/automl/tree/master/efficientdet). Model weights converted from official publication. It's the `h5` ones for `EfficientDetD*` models, not `ckpt` ones, as their accuracy higher.
  - [Paper 1911.09070 EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/pdf/1911.09070.pdf).

## Models
  ![](https://user-images.githubusercontent.com/5744524/151656702-9fb68cf6-e4ce-42b5-a488-80807cc66e56.png)

  | Model              | Params | Image resolution | COCO test AP | Download |
  | ------------------ | ------ | ---------------- | ------------ | -------- |
  | EfficientDetD0     | 3.9M   | 512              | 34.6         | [efficientdet_d0.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d0_512_coco.h5) |
  | EfficientDetD1     | 6.6M   | 640              | 40.5         | [efficientdet_d1.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d1_640_coco.h5) |
  | EfficientDetD2     | 8.1M   | 768              | 43.9         | [efficientdet_d2.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d2_768_coco.h5) |
  | EfficientDetD3     | 12.0M  | 896              | 47.2         | [efficientdet_d3.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d3_896_coco.h5) |
  | EfficientDetD4     | 20.7M  | 1024             | 49.7         | [efficientdet_d4.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d4_1024_coco.h5) |
  | EfficientDetD5     | 33.7M  | 1280             | 51.5         | [efficientdet_d5.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d5_1280_coco.h5) |
  | EfficientDetD6     | 51.9M  | 1280             | 52.6         | [efficientdet_d6.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d6_1280_coco.h5) |
  | EfficientDetD7     | 51.9M  | 1536             | 53.7         | [efficientdet_d7.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d7_1536_coco.h5) |
  | EfficientDetD7X    | 77.0M  | 1536             | 55.1         | [efficientdet_d7x.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d7x_1536_coco.h5) |
  | EfficientDetLite0  | 3.2M   | 320              | 26.41        | [efficientdet_lite0.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_lite0_320_coco.h5) |
  | EfficientDetLite1  | 4.2M   | 384              | 31.50        | [efficientdet_lite1.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_lite1_384_coco.h5) |
  | EfficientDetLite2  | 5.3M   | 448              | 35.06        | [efficientdet_lite2.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_lite2_448_coco.h5) |
  | EfficientDetLite3  | 8.4M   | 512              | 38.77        | [efficientdet_lite3.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_lite3_512_coco.h5) |
  | EfficientDetLite3X | 9.3M   | 640              | 42.64        | [efficientdet_lite3x.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_lite3x_640_coco.h5) |
  | EfficientDetLite4  | 15.1M  | 640              | 43.18        | [efficientdet_lite4.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_lite4_640_coco.h5) |
## Usage
  - **Basic usage**
    ```py
    from keras_cv_attention_models import efficientdet
    model = efficientdet.EfficientDetD0(pretrained="coco")

    # Run prediction
    from keras_cv_attention_models import test_images
    imm = test_images.dog_cat()
    bboxs, lables, confidences = model.decode_predictions(model(model.preprocess_input(imm)))[0]

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
  - `Backbone` for `EfficientDet` can be any model with pyramid stage structure.
    ```py
    from keras_cv_attention_models import efficientdet, coatnet
    bb = coatnet.CoAtNet0(input_shape=(256, 256, 3), num_classes=0)
    mm = efficientdet.EfficientDet(backbone=bb)
    # >>>> features: {'stack_2_block_3_output': (None, 32, 32, 192),
    #                 'stack_3_block_5_ffn_output': (None, 16, 16, 384),
    #                 'stack_4_block_2_ffn_output': (None, 8, 8, 768)}

    mm.summary()  # Trainable params: 18,773,185
    ```
  - Each `EfficientDetD*` / `EfficientDetLite*` can also set with `backbone=xxx` for using their pre-settings.
    ```py
    from keras_cv_attention_models import efficientdet, resnest
    mm = efficientdet.EfficientDetD2(backbone=resnest.ResNest50(input_shape=(384, 384, 3), num_classes=0), pretrained=None)

    mm.summary()  # Trainable params: 27,153,037
    ```
***
