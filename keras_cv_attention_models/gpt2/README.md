# ___Keras GPT2___
***

## Summary
  - Reference [Github karpathy/nanoGPT](https://github.com/karpathy/nanoGPT) and [Github jaymody/picoGPT](https://github.com/jaymody/picoGPT).
  - Will dowload and convert weights from `huggingface`.
## Model load statedict from huggingface and run prediction
  - Runing prediction requires `pip install tiktoken`.
  ```py
  from keras_cv_attention_models import gpt2

  mm = gpt2.GPT2_Base()
  gpt2.run_prediction(mm, "hello world")
  ```
***
