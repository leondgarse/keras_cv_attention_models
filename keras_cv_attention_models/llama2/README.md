# ___Keras LLaMA2___
***

## Summary
  - Keras implementation of [Github facebookresearch/llama](https://github.com/facebookresearch/llama). Paper [PDF 2307.09288 Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/pdf/2307.09288.pdf).
  - Model weights ported from [Github karpathy/llama2.c](https://github.com/karpathy/llama2.c).
## Models
  - `Params` is counted with `include_top=True`, will match the name if set `include_top=False`.

  | Model       | Params | FLOPs  | vocab_size | Val loss |
  | ----------- | ------ | ------ | ---------- | -------- |
  | [LLaMA2_15M](https://github.com/leondgarse/keras_cv_attention_models/releases/download/llama2/llama2_15m_tiny_stories.h5)  | 24.41M | 4.06G  | 32000      | 1.072    |
  | [LLaMA2_42M](https://github.com/leondgarse/keras_cv_attention_models/releases/download/llama2/llama2_42m_tiny_stories.h5)  | 58.17M | 50.7G  | 32000      | 0.847    |
  | [LLaMA2_110M](https://github.com/leondgarse/keras_cv_attention_models/releases/download/llama2/llama2_110m_tiny_stories.h5) | 134.1M | 130.2G | 32000      | 0.760    |
## Usage
  ```py
  from keras_cv_attention_models import llama2

  mm = llama2.LLaMA2_42M()
  # >>>> Load pretrained from: ~/.keras/models/llama2_42m_tiny_stories.h5
  mm.run_prediction("As evening fell, a maiden stood at the edge of a wood. In her hands,")
  # >>>> Load tokenizer from file: ~/.keras/datasets/llama_tokenizer.model
  # <s>
  # As evening fell, a maiden stood at the edge of a wood. In her hands, she held a beautiful diamond. Everyone was surprised to see it.
  # "What is it?" one of the kids asked.
  # "It's a diamond," the maiden said.
  # ...
  ```
  **Set `include_top=False`** to exclude model head layer.
  ```py
  from keras_cv_attention_models import llama2

  mm = llama2.LLaMA2_42M(include_top=False)
  # >>>> Load pretrained from: ~/.keras/models/llama2_42m_tiny_stories.h5
  print(f"{mm.output_shape = }")
  # mm.output_shape = (None, 1024, 512)
  ```
  **Set `pretrained="xxx.pt"`** for converting and loading weights from saved specific `xxx.pt` file.
  ```py
  from keras_cv_attention_models import llama2

  mm = llama2.LLaMA2_42M(pretrained="stories42M.pt")
  # Load and convert weights from huggingface
  # >>>> Save to: ~/.keras/models/llama2_42m_stories42M.h5
  mm.run_prediction("hello world", num_samples=1, max_new_tokens=100)
  # hello world was excited for the day. It was a bright day with lots of fun things to do.
  # ...
  ```
***
