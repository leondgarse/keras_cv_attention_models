# ___Keras YOLOV7___
***

## Summary
  - **Currently architecture only**
  - Keras implementation of [Github WongKinYiu/yolov7](https://github.com/WongKinYiu/yolov7). Model weights converted from official publication.
  - [Paper 2207.02696 YOLOv7: Trainable bag-of-freebies sets new state-of-the-art for real-time object detectors](https://arxiv.org/pdf/2207.02696.pdf).
***

## Usage
  - **Basic usage**
    ```py
    from keras_cv_attention_models import yolov7
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
