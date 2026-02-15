---
name: Weight Conversion
description: Converting PyTorch model weights to Keras h5 format for keras_cv_attention_models
---

# Weight Conversion Skill

Convert pretrained PyTorch weights to Keras h5 format.

The core task is **aligning the weight name order** between torch and keras models. `download_and_load.keras_reload_from_torch_model` is a convenience helper that automates this, but direct manual conversion is also fine.

## Pipeline Overview

1. **Weight collection** (`state_dict_stack_by_layer`): Groups torch `state_dict` entries by layer name (splitting on `.`), filtering via `skip_weights` and `unstack_weights`.
2. **Name alignment** (`align_layer_names_multi_stage`): Reorders keras layer names to match torch weight order.
3. **Weight transfer** (`keras_reload_stacked_state_dict`): Applies standard transforms (Conv2D/Dense transpose, etc.) plus custom `additional_transfer` overrides, then saves.

## Parameter Reference

### Weight Collection
| Parameter | Purpose |
|-----------|---------|
| `skip_weights` | Weight name suffixes to drop (e.g., `["num_batches_tracked", "relative_position_index"]`) |
| `unstack_weights` | Weights kept as individual entries instead of grouped with their layer (e.g., `["cls_token", "pos_embed", "gamma_1"]`) |

### Name Alignment (order matching)
| Parameter | Purpose |
|-----------|---------|
| `tail_align_dict` | Reposition layers by tail name: `{tail_name: offset}`. Negative offset moves layer earlier. Can be scoped by stack: `{"stack3": {"attn_gamma": -6}}` |
| `full_name_align_dict` | Reposition by exact name: value can be negative offset, absolute position, or another layer's name string |
| `tail_split_position` | Where to split name into head/tail (default `2`). E.g., `1` → head=`stack1`, tail=`attn_gamma` |
| `specific_match_func` | Function returning the complete ordered name list, bypassing all alignment logic. Use for complex cases where dicts can't express the mapping |

### Weight Transfer
| Parameter | Purpose |
|-----------|---------|
| `additional_transfer` | Custom transforms: `{LayerClass: lambda ww: [...]}` or `{"name_suffix": lambda ww: [...]}`. Applied after default Conv2D/Dense transposes |

## Workflow

1. Create keras model with `pretrained=None, classifier_activation=None`
2. Run with `do_convert=False` first to inspect both name lists
3. Compare printed torch/keras weight lists — find misalignments
4. Configure alignment parameters to fix ordering
5. Run with `do_convert=True` — it predicts with both models and prints results
6. Verify top prediction matches (usually `Egyptian_cat` for the cat test image)
7. `md5sum output.h5` → add hash to `PRETRAINED_DICT`
8. Upload to GitHub releases. Notify user if cannot upload directly.

## Troubleshooting

- **Shape mismatch**: Dense/Conv transposes are automatic; check if combined QKV needs `unstack_weights`
- **Name ordering wrong**: Use `do_convert=False` to see lists side-by-side; adjust offsets or use `specific_match_func` for full control
- **Predictions don't match**: Check `rescale_mode` in `add_pre_post_process()`, `classifier_activation`, or intermediate layer outputs
