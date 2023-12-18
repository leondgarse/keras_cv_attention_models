from keras_cv_attention_models.llama2.llama2 import (
    LLaMA2,
    LLaMA2_15M,
    LLaMA2_42M,
    LLaMA2_110M,
    LLaMA2_1B,
    LLaMA2_7B,
    RunPrediction,
    PositionalEncodingFourierRot1D,
    RMSNorm,
    convert_huggingface_weights_to_h5,
)

__head_doc__ = """
Keras implementation of [Github facebookresearch/llama](https://github.com/facebookresearch/llama).
Paper [PDF 2307.09288 Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/pdf/2307.09288.pdf).
"""

__tail_doc__ = """  vocab_size: model vocab size.
  max_block_size: number of tokens generated in each sample.
  include_top: boolena value if including output Dense head layer. Set false to exclude the head layer.
  dropout: float value for drop out rate for Embedding layer and attention blocks.
  activation: activation used in whole model, default `swish`.
  pretrained: None or "tiny_stories", or specific ".pt" or ".h5" file.
      - if "tiny_stories" or "tiny_llama_1.1B_chat_v0.4", will try to download and load ported weights if available.
      - if "xxx.pt", will try converting and loading weights from .pt file.
      - if "xxx.h5", will just load weights.
      - if None, will initialize model with ranbdom weights.

Returns:
    A `keras.Model` instance.
"""

LLaMA2.__doc__ = __head_doc__ + """
Args:
  num_blocks: num of `attention_fft_block`s.
  embedding_size: `attention_fft_block` block embedding size.
  hidden_divisible: int value making fft block hidden layer size multiple of large power of 2.
  num_heads: num of heads.
  num_kv_heads: int value specific key value heads, num_heads should be divisible by num_kv_heads. Default -1 for equal with num_heads.
  block_use_bias: boolean value if using bias for `attention_fft_block` Dense layers.
  model_name: string, model name.
""" + __tail_doc__ + """
Model architectures:
  | Model       | Params | FLOPs  | vocab_size | Val loss |
  | ----------- | ------ | ------ | ---------- | -------- |
  | LLaMA2_15M  | 24.41M | 4.06G  | 32000      | 1.072    |
  | LLaMA2_42M  | 58.17M | 50.7G  | 32000      | 0.847    |
  | LLaMA2_110M | 134.1M | 130.2G | 32000      | 0.760    |
  | LLaMA2_1B   | 1.10B  | 2.50T  | 32003      |          |
  | LLaMA2_7B   | 6.74B  | 14.54T | 32000      |          |
"""

LLaMA2_15M.__doc__ = __head_doc__ + """
Args:
""" + __tail_doc__

LLaMA2_42M.__doc__ = LLaMA2_15M.__doc__
LLaMA2_110M.__doc__ = LLaMA2_15M.__doc__
LLaMA2_1B.__doc__ = LLaMA2_15M.__doc__
LLaMA2_7B.__doc__ = LLaMA2_15M.__doc__
