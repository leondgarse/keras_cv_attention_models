# ___Keras EfficientDet___
***

## Summary
  - Keras implementation of [google/automl/efficientdet](https://github.com/google/automl/tree/master/efficientdet). Model weights converted from official publication.
  - [Paper 1911.09070 EfficientDet: Scalable and Efficient Object Detection](https://arxiv.org/pdf/1911.09070.pdf).

  | Model            | Params | Image resolution | COCO test AP | Download |
  | ---------------- | ------ | ---------------- | ------------ | -------- |
  | EfficientDet-D0  | 3.9M   | 512              | 34.6         | [efficientdet_d0.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d0_512_coco.h5) |
  | EfficientDet-D1  | 6.6M   | 640              | 40.5         | [efficientdet_d1.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d1_640_coco.h5) |
  | EfficientDet-D2  | 8.1M   | 768              | 43.9         | [efficientdet_d2.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d2_768_coco.h5) |
  | EfficientDet-D3  | 12.0M  | 896              | 47.2         | [efficientdet_d3.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d3_896_coco.h5) |
  | EfficientDet-D4  | 20.7M  | 1024             | 49.7         | [efficientdet_d4.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d4_1024_coco.h5) |
  | EfficientDet-D5  | 33.7M  | 1280             | 51.5         | [efficientdet_d5.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d5_1280_coco.h5) |
  | EfficientDet-D6  | 51.9M  | 1280             | 52.6         | [efficientdet_d6.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d6_1280_coco.h5) |
  | EfficientDet-D7  | 51.9M  | 1536             | 53.7         | [efficientdet_d7.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d7_1536_coco.h5) |
  | EfficientDet-D7x | 77.0M  | 1536             | 55.1         | [efficientdet_d7x.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/efficientdet/efficientdet_d7x_1536_coco.h5) |
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
  - **Use dynamic input resolution** by set `input_shape=(None, None, 3)`. Actually used `input_shape` should be dividable by `64`, and min size is `128`.
    ```py
    from keras_cv_attention_models import efficientdet
    model = efficientdet.EfficientDetD1(input_shape=(None, None, 3), pretrained="coco")
    # >>>> Load pretrained from: /home/leondgarse/.keras/models/efficientdet_d1_640_coco.h5
    print(model.input_shape, model.output_shape)
    # (None, None, None, 3) (None, None, 94)
    print(model(tf.ones([1, 256, 256, 3])).shape)
    # (1, 12276, 94)
    print(model(tf.ones([1, 768, 768, 3])).shape)
    # (1, 110484, 94)

    from keras_cv_attention_models import test_images
    imm = test_images.dog_cat()
    input_shape = (256, 256, 3)
    preds = model(model.preprocess_input(imm, input_shape=input_shape))
    bboxs, lables, confidences = model.decode_predictions(preds, input_shape=input_shape)[0]

    # Show result
    from keras_cv_attention_models.coco import data
    data.show_image_with_bboxes(imm, bboxs, lables, confidences, num_classes=90)
    ```
    ![effdetd1_dynamic_dog_cat](https://user-images.githubusercontent.com/5744524/151114148-3e6e6988-54ca-413c-8fab-1a9700149114.png)
  - **Custom detector using efficientdet header**. `Backbone` for `EfficientDet` can be any model with pyramid stage structure.
    ```py
    from keras_cv_attention_models import efficientdet, coatnet
    bb = coatnet.CoAtNet0(input_shape=(256, 256, 3), num_classes=0)
    mm = efficientdet.EfficientDet(backbone=bb)
    # >>>> features: {'stack_2_block_3_output': (None, 32, 32, 192),
    #                 'stack_3_block_5_output': (None, 16, 16, 384),
    #                 'stack_4_block_2_output': (None, 8, 8, 768)}

    mm.summary()  # Trainable params: 18,773,185
    ```
***
