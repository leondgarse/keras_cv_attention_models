---
description: PyTorch backend compatibility rules and gotchas for keras_cv_attention_models
---

# PyTorch Backend Compatibility

The `pytorch_backend/` provides a custom Keras-like API on top of PyTorch. It uses a graph-based execution model where layers build a `GraphNode` computation graph during model construction, then execute it in `Model.forward()`. **This has several critical gotchas.**

## Critical Rule: Layers with Non-Parameter Tensors

**Problem**: In the PyTorch backend's `Layer.forward()` (`pytorch_backend/layers.py`), when a `GraphNode` input is encountered, the node's `callable` is set based on `self.use_layer_as_module`:

```python
if self.use_layer_as_module:  # Layer has weights → callable = self (invokes call())
    cur_node.callable = self
else:  # No weights → callable = self.module (identity lambda!)
    cur_node.callable = self.module  # ← self.module = lambda xx: xx
```

`use_layer_as_module` is set to `True` only when `add_weight()` is called. **If a layer stores tensors as plain attributes without `add_weight()`, it becomes a silent no-op (identity function) in the PyTorch computation graph.**

**Solution**: Use `register_buffer()` in `__init__` for any tensors that are not trainable parameters:

```python
def __init__(self, **kwargs):
    super().__init__(**kwargs)
    if hasattr(self, "register_buffer"):  # PyTorch backend only
        self.register_buffer("my_tensor", None, persistent=False)
        # Initialize as None, assign actual value in build()
```

This ensures:
1. The layer is registered as a PyTorch submodule → `use_layer_as_module = True`
2. Tensors are automatically moved to the correct device with `.to(device)`
3. The layer's `call()` method is actually invoked during forward pass

> [!CAUTION]
> This is a **silent** failure — the model will run without errors but produce wrong predictions. The only symptom is degraded accuracy.

## Diagnosing PyTorch Backend Issues

### Symptom: Prediction mismatch between TF and PyTorch backends

1. **Compare intermediate outputs** layer by layer:
   ```python
   # Enable debug mode to see layer-by-layer execution
   mm.set_debug(True)
   ```

2. **Check if a layer is identity** by inspecting node callables:
   ```python
   for node in mm.forward_pipeline:
       if 'layer_name' in node.name:
           print(type(node.callable).__name__)  # 'function' = identity lambda
           test_input = torch.randn(1, 197, 384)
           print(torch.allclose(test_input, node.callable(test_input)))  # True = no-op!
   ```

3. **Write a comparison script** that runs both backends on the same input and compares named layer outputs.

### Common Sources of Mismatch

| Issue | Cause | Fix |
|-------|-------|-----|
| Layer is no-op | No `add_weight()` call → identity callable | Use `register_buffer()` in `__init__` |
| Device mismatch | `convert_to_tensor()` creates CPU tensors | Use `register_buffer()` for auto device placement |
| Shape mismatch | `functional.reshape` with hardcoded dims | Use `functional.shape(x)` or `blocks_shape` pattern for dimension-agnostic code |
| Split behavior | `math.ceil` in integer splits | Check `pytorch_backend/functional.py:split()` |

## Dimension-Agnostic Layer Design

Layers should support both 3D `[batch, tokens, channels]` and 4D `[batch, heads, tokens, channels]` inputs. Use the `blocks_shape` pattern:

```python
def build(self, input_shape):
    num_tokens = input_shape[-2]
    self.blocks_shape = [*input_shape[1:-2], num_tokens]  # Captures intermediate dims

def _process(self, inputs):
    # Use blocks_shape for reshape — preserves batch and any intermediate dims
    x = functional.reshape(inputs, [-1, *self.blocks_shape, self.channels // 2, 2])
    ...
    return functional.reshape(result, (-1, *self.blocks_shape, self.channels))
```

## Testing Both Backends

Always run tests on both backends:

```sh
# TF backend (default)
CUDA_VISIBLE_DEVICES='-1' pytest -vv --durations=0 ./tests

# PyTorch backend
CUDA_VISIBLE_DEVICES='-1' KECAM_BACKEND='torch' pytest -vv --durations=0 ./tests/test_models.py
```

## NumPy vs functional Operations in build()

Prefer **NumPy** operations in `build()` for tensors that are computed once and stored (e.g., positional encodings). This avoids backend-specific issues with `functional.tile()`, `functional.convert_to_tensor()`, etc.:

```python
# Good: compute in numpy, convert once
pos_sin = np.tile(np.sin(grid), [1, 1, num_heads])
self.pos_sin = functional.convert_to_tensor(pos_sin, dtype=self.compute_dtype)

# Risky: functional.tile() may behave differently across backends
self.pos_sin = functional.tile(functional.convert_to_tensor(np.sin(grid)), [1, 1, num_heads])
```
