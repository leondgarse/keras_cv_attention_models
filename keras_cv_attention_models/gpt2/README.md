# ___Keras GPT2___
***

## Summary
  - Keras implementation of [Github openai/gpt-2](https://github.com/openai/gpt-2). Paper [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language-models.pdf).
  - Model ported from [huggingface/gpt2](https://huggingface.co/gpt2).
  - References [Github karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) and [Github jaymody/picoGPT](https://github.com/jaymody/picoGPT).
## Models
  - For `GPT2_XLarge`, needs to download 2 file parts `gpt2_xlarge_webtext.1.h5` and `gpt2_xlarge_webtext.2.h5`.

  | Model            | Params  | FLOPs   | vocab_size | LAMBADA PPL |
  | ---------------- | ------- | ------- | ---------- | ----------- |
  | [GPT2_Base](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gpt2/gpt2_base_webtext.h5)        | 163.04M | 146.42G | 50257      | 35.13       |
  | [GPT2_Medium](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gpt2/gpt2_medium_webtext.h5)      | 406.29M | 415.07G | 50257      | 15.60       |
  | [GPT2_Large](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gpt2/gpt2_large_webtext.h5)       | 838.36M | 890.28G | 50257      | 10.87       |
  | [GPT2_XLarge](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gpt2/gpt2_xlarge_webtext.1.h5), [+.2](https://github.com/leondgarse/keras_cv_attention_models/releases/download/gpt2/gpt2_xlarge_webtext.2.h5) | 1.638B  | 1758.3G | 50257      | 8.63        |
## Usage
  ```py
  from keras_cv_attention_models import gpt2

  mm = gpt2.GPT2_Base()
  mm.run_prediction("hello world", num_samples=1, max_new_tokens=100)
  # hello world. I mean, just because we call ourselves anorexic, with a very strong genetic, doesn't mean we are human.
  #
  # And so there we have it. And we've just got to get through going through the rest of our lives.
  #
  #
  # I mean, it's a real challenge right now. And we know, we've already talked about the ethical issues. And so, I think, you know, the human body is a very dangerous thing, and the ethical issues
  # ---------------
  ```
  **Set `include_top=False`** to exclude model head layer.
  ```py
  from keras_cv_attention_models import gpt2

  mm = gpt2.GPT2_Base(include_top=False)
  # >>>> Load pretrained from: ~/.keras/models/gpt2_base_webtext.h5
  print(f"{mm.output_shape = }")
  # mm.output_shape = (None, 1024, 768)
  ```
  **Set `pretrained="huggingface"`** for converting and loading weights from huggingface `transformers` pacakge.
  ```py
  from keras_cv_attention_models import gpt2

  mm = gpt2.GPT2_Medium(pretrained="huggingface")
  # Load and convert weights from huggingface
  # >>>> Save to: ~/.keras/models/gpt2_medium_huggingface.h5
  mm.run_prediction("hello world", num_samples=1, max_new_tokens=100)
  # hello world, and he'll meet you in the afternoon and ask you to think about your career, and then I'll return. I'll write something up, and after that I'll have you come over."<|endoftext|>BALTIMORE -- The Baltimore Sun has been the one to expose the violence and destruction of the Baltimore riots that led to the death of Freddie Gray, and it's not your typical public servant.
  #
  # The Sun, which is owned by the Baltimore-based News Corp, went public with
  # ---------------
  ```
***
