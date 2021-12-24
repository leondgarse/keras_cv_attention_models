## Model Surgery
***

## Summary
  - Functions used to change model parameters after built.
  - `SAMModel`: SAMModel definition.
  - `add_l2_regularizer_2_model`: add `l2` weight decay to `Dense` / `Conv2D` / `DepthwiseConv2D` / `SeparableConv2D` layers.
  - `convert_to_mixed_float16`: convert `float32` model to `mixed_float16`.
  - `convert_mixed_float16_to_float32`: convert `mixed_float16` model to `float32`.
  - `convert_groups_conv2d_2_split_conv2d`: convert `Conv2D groups != 1` to `SplitConv2D` using `split -> conv -> concat`.
  - `convert_gelu_and_extract_patches_for_tflite`: convert model `gelu` activation to `gelu approximate=True`, and `tf.image.extract_patches` to a `Conv2D` version.
  - `convert_to_fused_conv_bn_model`: fuse convolution and batchnorm layers for inference.
  - `prepare_for_tflite`: a combination of `convert_groups_conv2d_2_split_conv2d` and `convert_gelu_and_extract_patches_for_tflite`.
  - `replace_ReLU`: replace all `ReLU` with other activations, default target is `PReLU`.
  - `replace_add_with_stochastic_depth`: replace all `Add` layers with `StochasticDepth`.
  - `replace_stochastic_depth_with_add`: replace all `StochasticDepth` layers with `add` + `multiply`.
## Usage
  - **Convert add layers to stochastic depth**
    ```py
    from keras_cv_attention_models import model_surgery
    mm = keras.applications.ResNet50()
    mm = model_surgery.replace_add_with_drop_connect(mm, drop_rate=(0, 0.2))
    print(model_surgery.get_actual_drop_connect_rates(mm))
    # [0.0, 0.0125, 0.025, 0.0375, 0.05, 0.0625, 0.075, 0.0875, 0.1, 0.1125, 0.125, 0.1375, 0.15, 0.1625, 0.175, 0.1875]
    ```
  - **Convert model between float16 and float32**
    ```py
    from keras_cv_attention_models import model_surgery
    mm = keras.applications.ResNet50()
    print(mm.layers[-1].compute_dtype)
    # float32
    mm = model_surgery.convert_to_mixed_float16(mm)
    print(mm.layers[-1].compute_dtype)
    # float16
    mm = model_surgery.convert_mixed_float16_to_float32(mm)
    print(mm.layers[-1].compute_dtype)
    # float32
    ```
  - **Convert groups conv2d to split conv2d for TFLite usage**
    ```py
    from keras_cv_attention_models import model_surgery, regnet
    mm = regnet.RegNetZD32()
    print([ii.groups for ii in mm.layers if isinstance(ii, keras.layers.Conv2D) and ii.groups != 1])
    # [8, 8, 8, 8, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 48, 48]
    mm = model_surgery.convert_groups_conv2d_2_split_conv2d(mm)
    print([ii.groups for ii in mm.layers if isinstance(ii, model_surgery.model_surgery.SplitConv2D)])
    # [8, 8, 8, 8, 16, 16, 16, 16, 16, 16, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 32, 48, 48]

    converter = tf.lite.TFLiteConverter.from_keras_model(mm)
    open(mm.name + ".tflite", "wb").write(converter.convert())
    ```
    ![](https://user-images.githubusercontent.com/5744524/147234593-0323b99b-7dcd-4b75-b8ed-94060346aabb.png)
  - **Change model input_shape after built**
    ```py
    from keras_cv_attention_models import model_surgery
    mm = keras.applications.ResNet50()
    print(mm.input_shape)
    # (None, 224, 224, 3)
    mm = model_surgery.change_model_input_shape(mm, (320, 320))
    print(mm.input_shape)
    # (None, 320, 320, 3)
    ```
  - **Replace ReLU activation layers**
    ```py
    from keras_cv_attention_models import model_surgery
    mm = keras.applications.ResNet50()
    print(mm.layers[-3].activation.__name__)
    # relu
    mm = model_surgery.replace_ReLU(mm, "PReLU")
    print(mm.layers[-3].__class__.__name__)
    # PReLU
    ```
  - **Fuse convolution and batchnorm layers for inference**
    ```py
    from keras_cv_attention_models import model_surgery
    mm = keras.applications.ResNet50()
    mm.summary()
    # Trainable params: 25,583,592
    mm = model_surgery.convert_to_fused_conv_bn_model(mm)
    mm.summary()
    # Trainable params: 25,530,472
    ```
***
