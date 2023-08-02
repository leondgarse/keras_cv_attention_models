from keras_cv_attention_models.vanillanet.vanillanet import (
    VanillaNet,
    VanillaNet5,
    VanillaNet6,
    VanillaNet7,
    VanillaNet8,
    VanillaNet9,
    VanillaNet10,
    VanillaNet11,
    VanillaNet12,
    VanillaNet13,
    set_leaky_relu_alpha,
    switch_to_deploy,
)
__head_doc__ = """
Keras implementation of [Github huawei-noah/VanillaNet](https://github.com/huawei-noah/VanillaNet).
Paper [PDF 2305.12972 VanillaNet: the Power of Minimalism in Deep Learning](https://arxiv.org/pdf/2305.12972.pdf).
"""

__tail_doc__ = """  input_shape: it should have exactly 3 inputs channels, like `(224, 224, 3)`.
  deploy: boolean value if build a fused model. **Evaluation only, not good for training**.
  num_classes: number of classes to classify images into. Set `0` to exclude top layers.
  activation: activation used in whole model, default `relu`.
  classifier_activation: A `str` or callable. The activation function to use on the "top" layer if `num_classes > 0`.
      Set `classifier_activation=None` to return the logits of the "top" layer.
      Default is `None`.
  pretrained: one of `None` (random initialization) or 'imagenet'.
      Will try to download and load pre-trained model weights if not None.

Returns:
    A `keras.Model` instance.

Train and Deploy:
- **Currently only works for Tensorflow**
- **`set_leaky_relu_alpha`** is introduced in the paper, used for changing activation alpha while training.
  Say `epochs=300, decay_epochs=100`, then `alpha = (current_epoch / decay_epochs) if current_epoch < decay_epochs else 1`.
- **`switch_to_deploy`** is used for converting a custom pretrained model to `deploy` version. The process is
  `fuse conv bn` -> `remove leaky_relu with alpha=1` -> `fuse 2 sequential conv`.
- Training process could be:
  >>> from keras_cv_attention_models import vanillanet
  >>> # Create a `deploy=False` for custom training
  >>> model = vanillanet.VanillaNet5(deploy=False, pretrained="imagenet")
  >>> print(f"{model.count_params() = }")
  >>> # model.count_params() = 22369712
  >>> # ...

  >>> # Run some training step with changing leaky_relu alpha
  >>> decay_epochs = 100
  >>> current_epoch = 10
  >>> model.set_leaky_relu_alpha(alpha=(current_epoch / decay_epochs) if current_epoch < decay_epochs else 1)
  >>> # Or set in actual callbacks
  >>> # on_epoch_end = lambda epoch, logs: model.set_leaky_relu_alpha(alpha=(epoch / decay_epochs) if epoch < decay_epochs else 1)
  >>> # act_learn = keras.callbacks.LambdaCallback(on_epoch_end=on_epoch_end)
  >>> # model.fit(..., callbacks=[..., act_learn])
  >>> # ...

  >>> # Fuse to deploy model
  >>> model.set_leaky_relu_alpha(alpha=1)  # This should already set when training reach `decay_epochs`
  >>> tt = model.switch_to_deploy()
  >>> deploy_save_name = model.name + "_deploy.h5"
  >>> tt.save(deploy_save_name)

  >>> # Further evaluation with deploy=True
  >>> bb = vanillanet.VanillaNet5(deploy=True, pretrained=deploy_save_name)
  >>> print(f"{bb.count_params() = }")
  >>> # bb.count_params() = 15523304
  >>> print(f"{np.allclose(model(tf.ones([1, 224, 224, 3])), bb(tf.ones([1, 224, 224, 3])), atol=1e-5) = }")
  >>> # np.allclose(model(tf.ones([1, 224, 224, 3])), bb(tf.ones([1, 224, 224, 3])), atol=1e-5) = True
"""

VanillaNet.__doc__ = __head_doc__ + """
Args:
  out_channels: output channels for each stack.
  strides: list value for strides value in each stack. Should be same length with `out_channels`.
  stem_width: stem output channels, default `-1` for using `out_channels[0]`.
  leaky_relu_alpha: float value in `[0, 1]` for initializing `LeakyReLU` alpha value, default `1.0` for identical, and `0` for `ReLU`.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model         | Params | FLOPs  | Input | Top1 Acc |
  | ------------- | ------ | ------ | ----- | -------- |
  | VanillaNet5   | 22.33M | 8.46G  | 224   | 72.49    |
  | - deploy=True | 15.52M | 5.17G  | 224   | 72.49    |
  | VanillaNet6   | 56.12M | 10.11G | 224   | 76.36    |
  | - deploy=True | 32.51M | 6.00G  | 224   | 76.36    |
  | VanillaNet7   | 56.67M | 11.84G | 224   | 77.98    |
  | - deploy=True | 32.80M | 6.90G  | 224   | 77.98    |
  | VanillaNet8   | 65.18M | 13.50G | 224   | 79.13    |
  | - deploy=True | 37.10M | 7.75G  | 224   | 79.13    |
  | VanillaNet9   | 73.68M | 15.17G | 224   | 79.87    |
  | - deploy=True | 41.40M | 8.59G  | 224   | 79.87    |
  | VanillaNet10  | 82.19M | 16.83G | 224   | 80.57    |
  | - deploy=True | 45.69M | 9.43G  | 224   | 80.57    |
  | VanillaNet11  | 90.69M | 18.49G | 224   | 81.08    |
  | - deploy=True | 50.00M | 10.27G | 224   | 81.08    |
  | VanillaNet12  | 99.20M | 20.16G | 224   | 81.55    |
  | - deploy=True | 54.29M | 11.11G | 224   | 81.55    |
  | VanillaNet13  | 107.7M | 21.82G | 224   | 82.05    |
  | - deploy=True | 58.59M | 11.96G | 224   | 82.05    |
"""

VanillaNet5.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

VanillaNet6.__doc__ = VanillaNet5.__doc__
VanillaNet7.__doc__ = VanillaNet5.__doc__
VanillaNet8.__doc__ = VanillaNet5.__doc__
VanillaNet9.__doc__ = VanillaNet5.__doc__
VanillaNet10.__doc__ = VanillaNet5.__doc__
VanillaNet11.__doc__ = VanillaNet5.__doc__
VanillaNet12.__doc__ = VanillaNet5.__doc__
VanillaNet13.__doc__ = VanillaNet5.__doc__
