# ___Keras LLaMA2___
***

## Summary
  - Keras implementation of [Github facebookresearch/llama](https://github.com/facebookresearch/llama). Paper [PDF 2307.09288 Llama 2: Open Foundation and Fine-Tuned Chat Models](https://arxiv.org/pdf/2307.09288.pdf).
  - `LLaMA2_15M` / `LLaMA2_42M`, `LLaMA2_110M` model weights ported from [Github karpathy/llama2.c](https://github.com/karpathy/llama2.c).
  - `LLaMA2_1B` model weights ported from [Github jzhang38/TinyLlama](https://githubfast.com/jzhang38/TinyLlama) `TinyLlama-1.1B-Chat-V0.4` one.
## Models
  - `Params` is counted with `include_top=True`, will match the name if set `include_top=False`.

  | Model       | Params | FLOPs  | vocab_size | Val loss |
  | ----------- | ------ | ------ | ---------- | -------- |
  | [LLaMA2_15M](https://github.com/leondgarse/keras_cv_attention_models/releases/download/llama2/llama2_15m_tiny_stories.h5)  | 24.41M | 4.06G  | 32000      | 1.072    |
  | [LLaMA2_42M](https://github.com/leondgarse/keras_cv_attention_models/releases/download/llama2/llama2_42m_tiny_stories.h5)  | 58.17M | 50.7G  | 32000      | 0.847    |
  | [LLaMA2_110M](https://github.com/leondgarse/keras_cv_attention_models/releases/download/llama2/llama2_110m_tiny_stories.h5) | 134.1M | 130.2G | 32000      | 0.760    |
  | [LLaMA2_1B](https://github.com/leondgarse/keras_cv_attention_models/releases/download/llama2/llama2_1b_tiny_llama_1.1B_chat_v0.4.h5) | 1.10B  | 2.50T  | 32003      |          |
  | LLaMA2_7B   | 6.74B  | 14.54T | 32000      |          |
## Usage
  ```py
  from keras_cv_attention_models import llama2

  mm = llama2.LLaMA2_42M()
  # >>>> Load pretrained from: ~/.keras/models/llama2_42m_tiny_stories.h5
  _ = mm.run_prediction("As evening fell, a maiden stood at the edge of a wood. In her hands,")
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
## Convert weights
- Manually downloading weights from [Huggingface meta-llama/Llama-2-7b-chat-hf](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf) or [Huggingface LinkSoul/Chinese-Llama-2-7b](https://huggingface.co/LinkSoul/Chinese-Llama-2-7b), and convert to `h5` format. The benefit of saving h5 is that, it's like `npz` or `tfrecord`, weights can be loaded layer by layer without reading the entire file into memory.
  ```py
  # Set to build model using pure float16 if using Tensorflow
  policy = keras.mixed_precision.Policy("float16")
  keras.mixed_precision.set_global_policy(policy)

  from keras_cv_attention_models import llama2
  _ = llama2.convert_huggingface_weights_to_h5("pytorch_model-00001-of-00002.bin", to_fp16=True)
  # >>>> Save to: pytorch_model-00001-of-00002.h5
  _ = llama2.convert_huggingface_weights_to_h5("pytorch_model-00002-of-00002.bin", to_fp16=True)
  # >>>> Save to: pytorch_model-00002-of-00002.h5
  ```
  Then load back into model.
  ```py
  policy = keras.mixed_precision.Policy("float16")
  keras.mixed_precision.set_global_policy(policy)

  from keras_cv_attention_models import llama2
  mm = llama2.LLaMA2_7B(pretrained=["pytorch_model-00001-of-00002.h5", "pytorch_model-00002-of-00002.h5"])
  # >>>> Load pretrained from: pytorch_model-00001-of-00002.h5
  # >>>> Load pretrained from: pytorch_model-00002-of-00002.h5
  mm.save(mm.name + ".h5")  # mm.half().save(mm.name + ".h5") if using PyTorch backend

  _ = mm.run_prediction("Who's there?")
  ```
***
