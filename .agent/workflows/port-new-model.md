---
description: How to port a new model to keras_cv_attention_models
---

# Porting a New Model

This workflow covers adding a new model architecture (vision or LLM) to `keras_cv_attention_models`.

## 1. Research the Source Model

1. Read the paper and reference implementation (usually PyTorch from GitHub).
2. Identify which **existing model family** it's closest to:
   - **Vision**: ViT-5 → `Beit`, EVA02 → `Beit`, DINOv2 → `Beit`, etc.
   - **LLM**: LLaMA3 → `LLaMA2`, etc.
3. Note key architectural differences: new layers, attention mechanisms, normalization, positional encoding.
4. Check if required building blocks already exist in `attention_layers/` (e.g., `RMSNorm`, `PositionalEncodingFourierRot`, `ClassToken`, `CausalMask`).

## 2. Implement New Components

1. If new layers are needed, add them to the appropriate file in `attention_layers/`.
2. **Reuse existing layers** whenever possible — check `attention_layers/__init__.py` for available components.
3. If modifying existing layers (e.g., adding parameters to `PositionalEncodingFourierRot`):
   - Ensure backward compatibility — existing models using the layer must still work.
   - Add new parameters with defaults matching the old behavior.
   - Update `get_config()` to include new parameters.

> [!IMPORTANT]
> **PyTorch backend compatibility**: Any layer that stores tensors as plain attributes (not via `add_weight`) must call `self.register_buffer()` in `__init__` for PyTorch backend support. Without this, the layer becomes a no-op identity in the PyTorch computation graph. See the PyTorch backend workflow for details.

## 3. Create the Model Variant File

Models that share a base architecture should be **thin wrappers**:

**Vision model** (see `beit/vit5.py`, `beit/eva02.py`):
```python
from keras_cv_attention_models.beit.beit import Beit
from keras_cv_attention_models.models import register_model

def ModelFamily(model_name="model_family", **kwargs):
    kwargs.pop("kwargs", None)
    return Beit(**locals(), **kwargs)

@register_model
def ModelFamily_Small(input_shape=(224, 224, 3), num_classes=1000, activation="gelu",
                      classifier_activation="softmax", pretrained="imagenet", **kwargs):
    embed_dim = 384
    num_heads = 6
    return ModelFamily(**locals(), model_name="model_family_small", **kwargs)
```

**LLM model** (see `llama2/llama2.py`):
```python
from keras_cv_attention_models.llama2.llama2 import LLaMA2
from keras_cv_attention_models.models import register_model

@register_model
def LLaMA3_8B(max_block_size=8192, vocab_size=128256, include_top=True,
              activation="swish", pretrained=None, **kwargs):
    num_blocks = 32
    embedding_size = 4096
    num_heads = 32
    num_kv_heads = 8
    return LLaMA2(**locals(), model_name="llama3_8b", **kwargs)
```

Key rules:
- Use `@register_model` decorator for each variant.
- Set `pretrained` default appropriately once weights are available.
- Use `kwargs.pop("kwargs", None)` to handle nested kwargs.
- Model names use **snake_case** (e.g., `vit5_small_patch16`, `llama2_42m`).

## 4. Register in `__init__.py`

1. Import the new model and variant functions.
2. Add a `__head_doc__` string with paper link and GitHub link.
3. Add a `__tail_doc__` for common parameters:
   - **Vision**: `input_shape`, `num_classes`, `classifier_activation`, `pretrained`, plus model-specific params.
   - **LLM**: `vocab_size`, `max_block_size`, `include_top`, `dropout`, `activation`, `pretrained`.
4. Add model architecture table (Params, FLOPs, and model-specific columns).

## 5. Weight Conversion

1. Add entries to `PRETRAINED_DICT` in the base model file:
   ```python
   # Vision: {model_name: {pretrained_tag: {resolution: md5}}}
   "vit5_small_patch16": {"imagenet": {224: "md5_hash_here"}},
   # LLM: {model_name: {pretrained_tag: md5}} (no resolution key)
   "llama2_42m": {"tiny_stories": "md5_hash_here"},
   ```
2. Convert weights using the appropriate method:
   - **Vision**: `download_and_load.keras_reload_from_torch_model()` or a model-specific helper.
   - **LLM**: `convert_huggingface_weights_to_h5()` or direct state_dict loading.
3. After conversion, verify:
   ```python
   # Vision
   mm = kecam.beit.ViT5_Small_Patch16(pretrained="path/to/converted.h5")
   pred = mm(mm.preprocess_input(kecam.test_images.cat()))
   print(mm.decode_predictions(pred)[0][0])  # ('n02124075', 'Egyptian_cat', 0.84...)

   # LLM
   mm = kecam.llama2.LLaMA2_42M(pretrained="path/to/converted.h5")
   print(mm.run_prediction("A long time ago,", top_k=1, max_new_tokens=5))
   ```
4. Compute the md5 hash: `md5sum converted_model.h5`
5. Upload to GitHub releases under the appropriate sub-release tag. Notice user if cannot upload directly.

## 6. Documentation

1. **Model-specific `README.md`** (e.g., `beit/README.md`, `llama2/README.md`):
   - Add paper reference in the Summary section.
   - Add model table with Params, FLOPs, and model-specific metrics.
   - Download links: `[filename.h5](https://github.com/leondgarse/keras_cv_attention_models/releases/download/{sub_release}/{filename}.h5)`
2. **Root `README.md`**: Add entry to Table of Contents in alphabetical order.
3. **`__init__.py`**: Ensure all parameters are documented with model-specific defaults noted.

## 7. Testing

Add tests in `tests/test_models.py`:

**Vision model**:
```python
def test_ModelFamily_defination():
    mm = keras_cv_attention_models.beit.ModelFamily_Small(pretrained=None)
    assert isinstance(mm, models.Model)
    mm = keras_cv_attention_models.beit.ModelFamily_Small(pretrained=None, num_classes=0)
    assert isinstance(mm, models.Model)

def test_ModelFamily_predict():
    mm = keras_cv_attention_models.beit.ModelFamily_Small(pretrained="imagenet")
    pred = mm(mm.preprocess_input(cat()))
    out = mm.decode_predictions(pred)[0][0]
    assert out[1] == "Egyptian_cat"
```

**LLM model**:
```python
def test_LLM_run_prediction():
    mm = keras_cv_attention_models.llama2.LLaMA2_42M(pretrained="tiny_stories")
    generated = mm.run_prediction("A long time ago,", top_k=1, max_new_tokens=5)
    assert generated == " there was a little girl"
```

If new layers were added, add layer tests in `tests/test_layers.py`.

## 8. Format and Verify

// turbo-all

1. Run black formatter:
   ```sh
   find ./* -name "*.py" | grep -v __init__ | xargs -I {} black -l 160 {}
   ```
2. Run TF test suite:
   ```sh
   CUDA_VISIBLE_DEVICES='-1' pytest -vv --durations=0 ./tests
   ```
3. Run PyTorch test suite:
   ```sh
   CUDA_VISIBLE_DEVICES='-1' KECAM_BACKEND='torch' pytest -vv --durations=0 ./tests/test_models.py
   ```