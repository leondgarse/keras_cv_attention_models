from keras_cv_attention_models.gpt2.gpt2 import (
    GPT2,
    GPT2_Base,
    GPT2_Medium,
    GPT2_Large,
    GPT2_XLarge,
    RunPrediction,
    PositionalIndex,
    CausalMask,
    load_weights_from_huggingface,
)

__head_doc__ = """
Keras implementation of [Github openai/gpt-2](https://github.com/openai/gpt-2).
Paper [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf).
"""

__tail_doc__ = """  vocab_size: model vocab size.
  max_block_size: number of tokens generated in each sample.
  include_top: boolena value if including output Dense head layer. Set false to exclude the head layer.
  dropout: float value for drop out rate for Embedding layer and attention blocks.
  activation: activation used in whole model, default `gelu/app`.
  pretrained: None or one of ["webtext", "huggingface"].
      - if "webtext", will try to download and load ported weights if available.
      - if "huggingface", will try converting and loading weights from huggingface `transformers` pacakge.
      - if None, will initialize model with ranbdom weights.

Returns:
    A `keras.Model` instance.
"""

GPT2.__doc__ = __head_doc__ + """
Args:
  num_blocks: num of `attention_mlp_block`s.
  embedding_size: `attention_mlp_block` block embedding size.
  num_heads: num of heads.
  block_use_bias: boolean value if using bias for `attention_mlp_block` Dense layers.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model       | Params  | FLOPs   | vocab_size | LAMBADA PPL |
  | ------------| ------- | ------- | ---------- | ----------- |
  | GPT2_Base   | 163.04M | 146.42G | 50257      | 35.13       |
  | GPT2_Medium | 406.29M | 415.07G | 50257      | 15.60       |
  | GPT2_Large  | 838.36M | 890.28G | 50257      | 10.87       |
  | GPT2_XLarge | 1.638B  | 1758.3G | 50257      | 8.63        |
"""

GPT2_Base.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

GPT2_Medium.__doc__ = GPT2_Base.__doc__
GPT2_Large.__doc__ = GPT2_Base.__doc__
GPT2_XLarge.__doc__ = GPT2_Base.__doc__
