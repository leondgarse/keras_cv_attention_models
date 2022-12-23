import numpy as np
import tensorflow as tf
from keras_cv_attention_models.visualizing.plot_func import get_plot_cols_rows, stack_and_plot_images


def matmul_prod(aa):
    vv = aa[0]
    for ii in aa[1:]:
        vv = np.matmul(vv, ii)
    return vv


def apply_mask_2_image(image, mask):
    if len(mask.shape) == 1:
        width = height = int(np.sqrt(mask.shape[0]))
        mask = mask[-width * height :]
    else:
        height, width = mask.shape[:2]
    mask = mask.reshape(width, height, 1)
    mask = tf.image.resize(mask / mask.max(), image.shape[:2], method="bilinear").numpy()
    return (mask * image).astype("uint8")


def clip_max_value_matrix(dd, axis=0):
    # print("Before:", dd.max())
    for ii in range(dd.shape[axis]):
        if axis == 0:
            max_idx = np.argmax(dd[ii])
            dd[ii, max_idx] = dd[ii].min()
            dd[ii, max_idx] = dd[ii].max()
        else:
            max_idx = np.argmax(dd[:, ii])
            dd[max_idx, ii] = dd[:, ii].min()
            dd[max_idx, ii] = dd[:, ii].max()
    # print("After:", dd.max())
    return dd


def down_sample_matrix_axis_0(dd, target, method="avg"):
    if dd.shape[0] == target:
        return dd
    rate = int(np.sqrt(dd.shape[0] // target))
    hh = ww = int(np.sqrt(dd.shape[0]))
    dd = dd.reshape(1, hh, ww, -1)
    if rate == 0:  # Upsample
        hh = ww = int(np.sqrt(target))
        dd = tf.image.resize(dd, [hh, ww]).numpy()
    elif "avg" in method.lower():
        dd = tf.nn.avg_pool(dd, rate, rate, "VALID").numpy()
    elif "swin" in method.lower():
        dd = dd.reshape(1, hh // 2, 2, ww // 2, 2, -1).transpose(0, 1, 3, 4, 2, 5)
    else:
        dd = tf.nn.max_pool(dd, rate, rate, "VALID").numpy()
    dd = dd.reshape(-1, dd.shape[-1])
    return dd


def plot_attention_score_maps(model, image, rescale_mode="auto", attn_type="auto", rows=-1, base_size=3):
    import matplotlib.pyplot as plt

    if rescale_mode.lower() == "auto":
        rescale_mode = getattr(model, "rescale_mode", "torch")
        print(">>>> rescale_mode:", rescale_mode)

    if isinstance(model, tf.keras.models.Model):
        imm_inputs = tf.keras.applications.imagenet_utils.preprocess_input(image, mode=rescale_mode)
        imm_inputs = tf.expand_dims(tf.image.resize(imm_inputs, model.input_shape[1:3]), 0)
        try:
            pred = model(imm_inputs).numpy()
            if model.layers[-1].activation.__name__ != "softmax":
                pred = tf.nn.softmax(pred).numpy()  # If classifier activation is not softmax
            print(">>>> Prediction:", tf.keras.applications.imagenet_utils.decode_predictions(pred)[0])
        except:
            pass
        bb = tf.keras.models.Model(model.inputs[0], [ii.output for ii in model.layers if ii.name.endswith("attention_scores")])
        attn_scores = bb(imm_inputs)
        layer_name_title = "\nLayer name: {} --> {}".format(bb.output_names[-1], bb.output_names[0])
    else:
        attn_scores = model
        layer_name_title = ""
        assert attn_type != "auto"

    attn_type = attn_type.lower()
    check_type_is = lambda tt: (tt in model.name.lower()) if attn_type == "auto" else (attn_type.startswith(tt))
    if check_type_is("beit"):
        # beit attn_score [batch, num_heads, cls_token + hh * ww, cls_token + hh * ww]
        print(">>>> Attention type: beit")
        mask = [np.array(ii)[0].mean(0) + np.eye(ii.shape[-1]) for ii in attn_scores][::-1]
        mask = [(ii / ii.sum()) for ii in mask]
        cum_mask = [matmul_prod(mask[: ii + 1])[0] for ii in range(len(mask))]
        mask = [ii[0] for ii in mask]
    elif check_type_is("levit"):
        # levit attn_score [batch, num_heads, q_blocks, k_blocks]
        print(">>>> Attention type: levit")
        mask = [np.array(ii)[0].mean(0) for ii in attn_scores][::-1]
        cum_mask = [matmul_prod(mask[: ii + 1]).mean(0) for ii in range(len(mask))]
        mask = [ii.mean(0) for ii in mask]
    elif check_type_is("bot"):
        # bot attn_score [batch, num_heads, hh * ww, hh * ww]
        print(">>>> Attention type: bot")
        mask = [np.array(ii)[0].mean((0)) for ii in attn_scores if len(ii.shape) == 4][::-1]
        mask = [clip_max_value_matrix(ii) for ii in mask]  # Or it will be too dark.
        cum_mask = [mask[0]] + [down_sample_matrix_axis_0(mask[ii], mask[ii - 1].shape[1], "avg") for ii in range(1, len(mask))]
        cum_mask = [matmul_prod(cum_mask[: ii + 1]).mean(0) for ii in range(len(cum_mask))]
        mask = [ii.mean(0) for ii in mask]
    elif check_type_is("coatnet") or check_type_is("cmt") or check_type_is("uniformer") or check_type_is("swin"):
        # bot attn_score [batch, num_heads, hh * ww, hh * ww]
        print(">>>> Attention type: coatnet / cmt / uniformer / swin")
        mask = [np.array(ii)[0].mean((0)) for ii in attn_scores if len(ii.shape) == 4][::-1]
        downsample_method = "swin" if check_type_is("swin") else "max"
        cum_mask = [mask[0]] + [down_sample_matrix_axis_0(mask[ii], mask[ii - 1].shape[1], downsample_method) for ii in range(1, len(mask))]
        cum_mask = [matmul_prod(cum_mask[: ii + 1]).mean(0) for ii in range(len(cum_mask))]
        mask = [ii.mean(0) for ii in mask]
    elif check_type_is("coat"):
        # coat attn_score [batch, num_heads, cls_token + hh * ww, key_dim]
        print(">>>> Attention type: coat")
        mask = [np.array(ii)[0].mean((0))[1:] for ii in attn_scores if len(ii.shape) == 4][::-1]
        mask = [ii.max(-1, keepdims=True) for ii in mask]
        target_shape = np.min([ii.shape[0] for ii in mask])
        cum_mask = [down_sample_matrix_axis_0(mask[ii], target_shape, "max") for ii in range(len(mask))]
        cum_mask = [ii[:, 0] for ii in cum_mask]
        mask = [ii[:, 0] for ii in mask]
    elif check_type_is("halo"):
        # halo attn_score [batch, num_heads, hh, ww, query_block * query_block, kv_kernel * kv_kernel]
        print(">>>> Attention type: halo")
        from einops import rearrange
        from keras_cv_attention_models.attention_layers import CompatibleExtractPatches

        mask = [np.array(ii)[0].mean(0) for ii in attn_scores if len(ii.shape) == 6][::-1]

        qqs = [int(np.sqrt(ii.shape[2])) for ii in mask]  # query_kernel
        vvs = [int(np.sqrt(ii.shape[3])) for ii in mask]  # kv_kernel
        hhs = [(jj - ii) // 2 for ii, jj in zip(qqs, vvs)]  # halo_size
        tt = [rearrange(ii, "hh ww (hb wb) cc -> (hh hb) (ww wb) cc", hb=qq, wb=qq) for ii, qq in zip(mask, qqs)]
        tt = [tf.expand_dims(tf.pad(ii, [[hh, hh], [hh, hh], [0, 0]]), 0) for ii, hh in zip(tt, hhs)]
        tt = [CompatibleExtractPatches(vv, qq, padding="VALID", compressed=False)(ii).numpy()[0] for ii, vv, qq in zip(tt, vvs, qqs)]
        tt = [rearrange(ii, "hh ww hb wb cc -> hh ww (hb wb) cc").mean((0, 1)) for ii in tt]
        # tt = [tf.reduce_max(rearrange(ii, "hh ww hb wb cc -> hh ww (hb wb) cc"), axis=(0, 1)).numpy() for ii in tt]
        cum_mask = [matmul_prod(tt[: ii + 1]).mean(0) for ii in range(len(tt))]
        mask = [ii.mean((0, 1, 2)) for ii in mask]
    else:
        print(">>>> Attention type: cot / volo / unknown")
        # cot attn_score [batch, 1, 1, filters, randix]
        # volo attn_score [batch, hh, ww, num_heads, kernel_size * kernel_size, kernel_size * kernel_size]
        print("[{}] still don't know how...".format(attn_type))
        return None, None, None

    masked_image = [apply_mask_2_image(image, ii) for ii in mask]
    cum_masked_image = [apply_mask_2_image(image, ii) for ii in cum_mask]

    cols, rows = get_plot_cols_rows(len(mask), rows)
    fig, axes = plt.subplots(2, 1, figsize=(base_size * cols, base_size * rows * 2))
    stack_and_plot_images(masked_image, margin=5, rows=rows, ax=axes[0])
    axes[0].set_title("Attention scores: attn_scores[{}] --> attn_scores[0]".format(len(mask)) + layer_name_title)
    stack_and_plot_images(cum_masked_image, margin=5, rows=rows, ax=axes[1])
    axes[1].set_title("Accumulated attention scores: attn_scores[{}:] --> attn_scores[0:]".format(len(mask) - 1) + layer_name_title)
    fig.tight_layout()
    return mask, cum_mask, fig
