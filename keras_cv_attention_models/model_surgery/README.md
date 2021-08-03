## Model Surgery
***

## Summary
  - Functions used to change model parameters after built.
  - `SAMModel`: SAMModel definition.
  - `add_l2_regularizer_2_model`: add `l2` weight decay to `Dense` / `Conv2D` / `DepthwiseConv2D` / `SeparableConv2D` layers.
  - `convert_to_mixed_float16`: convert `float32` model to `mixed_float16`.
  - `convert_mixed_float16_to_float32`: convert `mixed_float16` model to `float32`.
  - `convert_to_fused_conv_bn_model`: fuse convolution and batchnorm layers for inference.
  - `replace_ReLU`: replace all `ReLU` with other activations, default target is `PReLU`.
  - `replace_add_with_stochastic_depth`: replace all `Add` layers with `StochasticDepth`.
  - `replace_stochastic_depth_with_add`: replace all `StochasticDepth` layers with `add` + `multiply`.
***
