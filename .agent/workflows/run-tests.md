---
description: How to run tests and formatting for keras_cv_attention_models
---

# Running Tests and Formatting

// turbo-all

## 1. Format Code with Black

```sh
find ./* -name "*.py" | grep -v __init__ | xargs -I {} black -l 160 {}
```

Note: `__init__.py` files are excluded from formatting.

## 2. Run TF Backend Tests

```sh
CUDA_VISIBLE_DEVICES='-1' pytest -vv --durations=0 ./tests
```

This runs all tests: `test_models.py`, `test_layers.py`, `test_models_tf.py`, `test_switch_to_deploy_tf.py`.

## 3. Run PyTorch Backend Tests

```sh
CUDA_VISIBLE_DEVICES='-1' KECAM_BACKEND='torch' pytest -vv --durations=0 ./tests/test_models.py
```

Only `test_models.py` is run for PyTorch backend.

## Notes

- These tests are also executed in GitHub Actions (`.github/workflows/publish-to-test-pypi.yml`), so running locally is not compulsory but recommended before pushing.
- TF tests take ~14 minutes on CPU, PyTorch tests take ~2.5 minutes.
- Use `CUDA_VISIBLE_DEVICES='-1'` to force CPU execution for reproducible results.
- To run a single test: `pytest tests/test_models.py::test_ViT5_predict -xvs`
