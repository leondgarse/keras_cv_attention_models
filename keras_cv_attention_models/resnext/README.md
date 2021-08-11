# Keras_ResNeXt
***

## Summary
  - Keras implementation of [Github facebookresearch/ResNeXt](https://github.com/facebookresearch/ResNeXt). Paper [PDF 1611.05431 Aggregated Residual Transformations for Deep Neural Networks](https://arxiv.org/pdf/1611.05431.pdf).
  - Model weights reloaded from [Tensorflow keras/applications](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/applications/resnet.py).
***

## Models
  | Model          | Params | Image  resolution | Top1 Acc | Download            |
  | -------------- | ------ | ----------------- | -------- | ------------------- |
  | resnext50      | 25M    | 224               | 77.8     | [resnext50.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnext/resnext50.h5)  |
  | resnext101     | 42M    | 224               | 80.9     | [resnext101.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/resnext/resnext101.h5)  |
## Usage
  ```py
  from keras_cv_attention_models import resnext

  # Will download and load pretrained imagenet weights.
  mm = resnext.ResNeXt50(pretrained="imagenet")

  # Run prediction
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='tf') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.98292357),
  #  ('n02123045', 'tabby', 0.009655442),
  #  ('n02123159', 'tiger_cat', 0.0057404325),
  #  ('n02127052', 'lynx', 0.00089362176),
  #  ('n04209239', 'shower_curtain', 0.00013918217)]
  ```
  **Set new input resolution**
  ```py
  from keras_cv_attention_models import resnext
  mm = resnext.ResNeXt101(input_shape=(320, 320, 3), num_classes=0)
  print(mm(np.ones([1, 320, 320, 3])).shape)
  # (1, 10, 10, 2048)

  mm = resnext.ResNeXt101(input_shape=(512, 512, 3), num_classes=0)
  print(mm(np.ones([1, 512, 512, 3])).shape)
  # (1, 16, 16, 2048)
  ```
***
