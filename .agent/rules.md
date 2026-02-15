# Project Rules for keras_cv_attention_models

## Code Style
- Line length: 160 characters (`black -l 160`)
- `__init__.py` files are excluded from black formatting
- Use single-line multi-assignment for related vars: `self.a, self.b = a, b`

## Model Architecture Patterns
- Model variant files (e.g., `vit5.py`) should be **thin wrappers** calling the base function (e.g., `Beit`)
- Use `@register_model` decorator for every model variant function
- Use `kwargs.pop("kwargs", None)` at the start of base model functions to handle nested kwargs
- Model names in code use **snake_case** (e.g., `vit5_small_patch16`), class-style names for functions (e.g., `ViT5_Small_Patch16`)
- Default `pretrained="imagenet"` (or appropriate tag) once weights are available

## PyTorch Backend (CRITICAL)
- Any layer storing tensors as attributes (not via `add_weight`) **must** call `self.register_buffer()` in `__init__` with `if hasattr(self, "register_buffer")` guard
- Prefer NumPy operations in `build()` for one-time tensor computation, then convert with `functional.convert_to_tensor()`
- Always verify predictions match between TF and PyTorch backends after changes

## Documentation
- Every model family needs: paper link, GitHub link, model table with Params/FLOPs/Input/Top1 Acc/Download
- Document all parameters in `__class_tail_doc__` with model-specific defaults noted (e.g., "Default True for ViT-5, False for others")
- Download links: `https://github.com/leondgarse/keras_cv_attention_models/releases/download/{sub_release}/{filename}.h5`

## Testing
- Every model needs both `test_ModelFamily_defination` and `test_ModelFamily_predict` in `tests/test_models.py`

## Modifying Existing Code
- **Preserve existing patterns**: When refactoring a layer, study the original implementation's design choices (e.g., `blocks_shape` for dimension-agnostic reshaping) before replacing them with new approaches. Reuse proven patterns rather than inventing alternatives.
- **Use proper APIs, not workarounds**: For PyTorch backend, use `register_buffer()` — don't set internal flags like `use_layer_as_module` directly. Always prefer the idiomatic solution over hacking internals.
- **Run existing tests first**: Before modifying a layer, run its existing test (`test_layers.py`) to understand the input/output contract. Don't change behavior without checking what the tests expect.
- **`build()` should use numpy**: Computations in `build()` are one-time setup. Use `np.tile`, `np.sin`, `np.concatenate` etc., then convert the final result with `functional.convert_to_tensor()`. Don't use `functional.tile` or other framework ops for build-time constants.
- **Consolidate related operations**: When a helper function (e.g., `_get_sin_cos`) always has the same post-processing (e.g., tiling by `num_heads`), fold that into the helper rather than leaving it in the caller.
