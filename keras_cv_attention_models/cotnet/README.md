# Keras_CoTNet
***

## Summary
  - [PDF 2107.12292 Contextual Transformer Networks for Visual Recognition](https://arxiv.org/pdf/2107.12292.pdf)
  - [Github JDAI-CV/CoTNet](https://github.com/JDAI-CV/CoTNet)
***

## Models
  | Model          | params | Image resolution | FLOPs | Top1 Acc | Top5 Acc | Download            |
  | -------------- |:------:| ---------------- | ----- |:--------:|:--------:| ------------------- |
  | CoTNet-50      | 22.2M  | 224              | 3.3   |   81.3   |   95.6   |[cotnet50_224.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/cotnet/cotnet50_224.h5)  |
  | CoTNeXt-50     | 30.1M  | 224              | 4.3   |   82.1   |   95.9   |  |
  | SE-CoTNetD-50  | 23.1M  | 224              | 4.1   |   81.6   |   95.8   |  |
  | CoTNet-101     | 38.3M  | 224              | 6.1   |   82.8   |   96.2   |  |
  | CoTNeXt-101    | 53.4M  | 224              | 8.2   |   83.2   |   96.4   |  |
  | SE-CoTNetD-101 | 40.9M  | 224              | 8.5   |   83.2   |   96.5   |  |
  | SE-CoTNetD-152 | 55.8M  | 224              | 17.0  |   84.0   |   97.0   |  |
  | SE-CoTNetD-152 | 55.8M  | 320              | 26.5  |   84.6   |   97.1   |  |
## Usage
  ```py
  from keras_cv_attention_models import cotnet

  # Will download and load pretrained imagenet weights.
  mm = cotnet.CotNet50(pretrained="imagenet")

  # Run prediction
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.72987473),
  #  ('n02123159', 'tiger_cat', 0.073337175),
  #  ('n02123045', 'tabby', 0.03993373),
  #  ('n02127052', 'lynx', 0.0032631743),
  #  ('n03720891', 'maraca', 0.0021165807)]
  ```
  **Change input resolution**
  ```py
  from keras_cv_attention_models import cotnet
  mm = cotnet.CotNet50(input_shape=(480, 480, 3), pretrained="imagenet")

  # Run prediction on Chelsea with (480, 480) resolution
  from skimage.data import chelsea
  imm = keras.applications.imagenet_utils.preprocess_input(chelsea(), mode='torch') # Chelsea the cat
  pred = mm(tf.expand_dims(tf.image.resize(imm, mm.input_shape[1:3]), 0)).numpy()
  print(keras.applications.imagenet_utils.decode_predictions(pred)[0])
  # [('n02124075', 'Egyptian_cat', 0.5839457), ('n02123159', 'tiger_cat', 0.15395148), ...]
  ```
***
