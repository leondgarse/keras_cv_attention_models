import numpy as np
import tensorflow as tf
from tqdm import tqdm


def __gradient_ascent_step__(feature_extractor, image_var, filter_index, optimizer):
    with tf.GradientTape() as tape:
        tape.watch(image_var)
        activation = feature_extractor(tf.expand_dims(image_var[0], 0))
        # We avoid border artifacts by only involving non-border pixels in the loss.
        # filter_activation = activation[:, 2:-2, 2:-2, filter_index]
        filter_activation = activation[..., filter_index]
        loss = tf.reduce_mean(filter_activation)
    # Compute gradients.
    grads = tape.gradient(loss, image_var)
    # Normalize gradients.
    grads = tf.math.l2_normalize(grads)
    # image_var += 10.0 * grads # 10.0 is learning_rate
    optimizer.apply_gradients(zip(grads * -1, image_var))  # For SGD, w = w - learning_rate * g
    return loss, image_var


def __initialize_image__(img_width, img_height, rescale_mode="tf", value_range=0.125):
    # We start from a gray image with some random noise. Pick image vvalue range in middle
    min_val, max_val = int(128 - 128 * value_range), int(128 + 128 * value_range)
    img = tf.random.uniform((img_width, img_height, 3), min_val, max_val)
    # For ResNet50V2 expects inputs in the range [-1, +1], results in `[-0.25, 0.25]`
    return tf.keras.applications.imagenet_utils.preprocess_input(img, mode=rescale_mode).numpy()


def __deprocess_image__(img, crop_border=0.1):
    # Normalize array: center on 0., ensure variance is 0.15
    img -= img.mean(axis=(0, 1))
    img /= img.std(axis=(0, 1)) + 1e-5
    img *= 0.15

    # Center crop
    hh_crop, ww_crop = int(img.shape[0] * crop_border), int(img.shape[1] * crop_border)
    img = img[hh_crop:-hh_crop, ww_crop:-ww_crop]

    # Clip to [0, 1]
    img += 0.5
    img = np.clip(img, 0, 1)

    # Convert to RGB array
    img *= 255
    img = np.clip(img, 0, 255).astype("uint8")
    return img


def get_plot_cols_rows(total, rows=-1, ceil_mode=False):
    if rows == -1 and total < 8:
        rows = 1  # for total in [1, 7], plot 1 row only
    elif rows == -1:
        rr = int(np.floor(np.sqrt(total)))
        for ii in range(1, rr + 1)[::-1]:
            if total % ii == 0:
                rows = ii
                break
    if ceil_mode:
        cols = int(np.ceil(total / rows))
    else:
        cols = total // rows
    return cols, rows


def put_text_on_image(image, text, coord=(5, 5), color=(255, 0, 0)):
    from PIL import Image
    from PIL import ImageDraw

    # from PIL import ImageFont

    image = image * 255 if image.max() < 2 else image
    img = Image.fromarray(image.astype("uint8"))
    draw = ImageDraw.Draw(img)
    draw.text(coord, str(text), color)
    return np.array(img)


def stack_and_plot_images(images, texts=None, margin=5, margin_value=0, rows=-1, ax=None, base_size=3):
    """ Stack and plot a list of images. Returns ax, stacked_images """
    import matplotlib.pyplot as plt

    cols, rows = get_plot_cols_rows(len(images), rows)
    images = images[: rows * cols]
    # print(">>>> rows:", rows, ", cols:", cols, ", total:", len(images))

    if texts is not None:
        images = [put_text_on_image(imm, itt) for imm, itt in zip(images, texts)] + list(images[len(texts) :])
        images = np.array(images)

    if ax is None:
        _, ax = plt.subplots(1, 1, figsize=(base_size * cols, base_size * rows))

    if margin > 0:
        ww_margin = np.zeros_like(images[0][:, :margin]) + margin_value
        ww_margined_images = [np.hstack([ii, ww_margin]) for ii in images]
        hstacked_images = [np.hstack(ww_margined_images[ii : ii + cols]) for ii in range(0, len(ww_margined_images), cols)]

        hh_margin = np.zeros_like(hstacked_images[0][:margin]) + margin_value
        hh_margined_images = [np.vstack([ii, hh_margin]) for ii in hstacked_images]
        vstacked_images = np.vstack(hh_margined_images)

        stacked_images = vstacked_images[:-margin, :-margin]
    else:
        stacked_images = np.vstack([np.hstack(images[ii * cols : (rr + 1) * cols]) for ii in range(rows)])

    ax.imshow(stacked_images)
    ax.set_axis_off()
    ax.grid(False)
    plt.tight_layout()
    plt.show()
    return ax, stacked_images


def visualize_filters(
    model,
    layer_name="auto",
    filter_index_list=[0],
    input_shape=None,
    rescale_mode="auto",
    iterations=30,
    optimizer="SGD",  # "SGD" / "RMSprop" / "Adam"
    learning_rate="auto",
    value_range=0.125,
    random_magnitude=1.0,  # basic random value for `tf.roll` and `random_rotation` is `4` and `1`.
    crop_border=0.1,
    base_size=3,
):
    from tensorflow.keras.preprocessing.image import random_rotation

    # Set up a model that returns the activation values for our target layer
    # model = tf.keras.models.clone_model(model)
    if layer_name == "auto":
        layer = model.layers[-1]
        layer_name = layer.name
    else:
        layer = model.get_layer(name=layer_name)
    feature_extractor = tf.keras.Model(inputs=model.inputs[0], outputs=layer.output)
    input_shape = model.input_shape[1:-1] if input_shape is None else input_shape[:2]
    assert input_shape[0] is not None and input_shape[1] is not None
    print(">>>> Total filters for layer {}: {}, input_shape: {}".format(layer_name, layer.output_shape[-1], input_shape))

    auto_lr = {"ada": 0.1, "rms": 1.0, "sgd": 10.0}
    if isinstance(optimizer, str):
        optimizer = optimizer.lower()
        learning_rate = auto_lr.get(optimizer[:3], 1.0) if learning_rate == "auto" else learning_rate
        if optimizer.startswith("rms"):
            optimizer = tf.optimizers.RMSprop(learning_rate, rho=0.999)
        elif optimizer == "adam":
            optimizer = tf.optimizers.Adam(learning_rate)
        elif optimizer == "sgd":
            optimizer = tf.optimizers.SGD(learning_rate)

    if rescale_mode.lower() == "auto":
        rescale_mode = getattr(model, "rescale_mode", "torch")
        print(">>>> rescale_mode:", rescale_mode)

    # We run gradient ascent for [iterations] steps
    losses, filter_images = [], []
    for filter_index in filter_index_list:
        image = __initialize_image__(input_shape[0], input_shape[1], rescale_mode, value_range=value_range)
        for iteration in tqdm(range(iterations), desc="Processing filter %d" % (filter_index,)):
            image = random_rotation(image, 1 * random_magnitude, row_axis=0, col_axis=1, channel_axis=2, fill_mode="reflect")
            image = tf.roll(image, shift=np.random.randint(-4 * random_magnitude, 4 * random_magnitude, size=2), axis=[0, 1])
            image_var = [tf.Variable(image)]
            loss, image_var = __gradient_ascent_step__(feature_extractor, image_var, filter_index, optimizer)
            image = image_var[0].numpy()
        # Decode the resulting input image
        image = __deprocess_image__(image, crop_border)
        losses.append(loss.numpy())
        filter_images.append(image)

    ax, _ = stack_and_plot_images(filter_images, base_size=base_size)
    return losses, np.stack(filter_images), ax


def make_gradcam_heatmap(model, processed_image, layer_name="auto", pred_index=None, use_v2=True):
    # First, we create a model that maps the input image to the activations
    # of the last conv layer as well as the output predictions
    if layer_name == "auto":
        for ii in model.layers[::-1]:
            if len(ii.output_shape) == 4:
                # if isinstance(ii, tf.keras.layers.Conv2D):
                layer_name = ii.name
                print("Using layer_name:", layer_name)
                break
    grad_model = tf.keras.models.Model(model.inputs[0], [model.get_layer(layer_name).output, model.output])

    # Then, we compute the gradient of the top predicted class for our input image
    # with respect to the activations of the last conv layer
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(processed_image)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)

    if use_v2:
        # gradcam_plus_plus from: https://github.com/keisen/tf-keras-vis/blob/master/tf_keras_vis/gradcam_plus_plus.py
        score_values = tf.reduce_sum(tf.math.exp(class_channel))
        first_derivative = score_values * grads
        second_derivative = first_derivative * grads
        third_derivative = second_derivative * grads

        reduction_axis = list(range(1, len(last_conv_layer_output.shape) - 1))
        global_sum = tf.reduce_sum(last_conv_layer_output, axis=reduction_axis, keepdims=True)
        alpha_denom = second_derivative * 2.0 + third_derivative * global_sum
        alpha_denom = tf.where(second_derivative == 0.0, tf.ones_like(alpha_denom), alpha_denom)
        alphas = second_derivative / alpha_denom

        alpha_norm_constant = tf.reduce_sum(alphas, axis=reduction_axis, keepdims=True)
        alpha_norm_constant = tf.where(alpha_norm_constant == 0.0, tf.ones_like(alpha_norm_constant), alpha_norm_constant)
        alphas = alphas / alpha_norm_constant

        deep_linearization_weights = first_derivative * alphas
        deep_linearization_weights = tf.reduce_sum(deep_linearization_weights, axis=reduction_axis)
    else:
        # This is a vector where each entry is the mean intensity of the gradient over a specific feature map channel
        deep_linearization_weights = tf.reduce_mean(grads, axis=list(range(0, len(grads.shape) - 1)))

    # We multiply each channel in the feature map array
    # by "how important this channel is" with regard to the top predicted class
    # then sum all the channels to obtain the heatmap class activation
    heatmap = last_conv_layer_output @ deep_linearization_weights[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    if len(heatmap.shape) > 2:
        heatmap = tf.reduce_mean(heatmap, list(range(0, len(heatmap.shape) - 2)))

    # For visualization purpose, we will also normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy().astype("float32"), preds.numpy()


def make_and_apply_gradcam_heatmap(model, image, layer_name="auto", rescale_mode="auto", pred_index=None, alpha=0.8, use_v2=True, plot=True):
    import matplotlib.cm as cm
    import matplotlib.pyplot as plt

    if rescale_mode.lower() == "auto":
        rescale_mode = getattr(model, "rescale_mode", "torch")
        print(">>>> rescale_mode:", rescale_mode)

    image = np.array(image)
    image = image * 255 if image.max() < 2 else image  # Makse sure it's [0, 255]
    processed_image = tf.expand_dims(tf.image.resize(image, model.input_shape[1:-1]), 0)
    processed_image = tf.keras.applications.imagenet_utils.preprocess_input(processed_image, mode=rescale_mode)
    heatmap, preds = make_gradcam_heatmap(model, processed_image, layer_name, pred_index=pred_index, use_v2=use_v2)

    # Use jet colormap to colorize heatmap. Use RGB values of the colormap
    jet = cm.get_cmap("jet")
    jet_colors = jet(tf.range(256))[:, :3]
    jet_heatmap = jet_colors[tf.cast(heatmap * 255, "uint8").numpy()]

    # Create an image with RGB colorized heatmap
    jet_heatmap = tf.image.resize(jet_heatmap, (image.shape[:2]))  # [0, 1]

    # Superimpose the heatmap on original image
    image = image.astype("float32") / 255
    superimposed_img = (jet_heatmap * alpha + image).numpy()
    superimposed_img /= superimposed_img.max()

    if model.output_shape[-1] == 1000:
        decode_pred = tf.keras.applications.imagenet_utils.decode_predictions(preds, top=5)[0]
        top_5_idxes = np.argsort(preds[0])[-5:][::-1]
        print(">>>> Top5 predictions:", np.array([[ii, *jj] for ii, jj in zip(top_5_idxes, decode_pred)]))
    if plot:
        fig = plt.figure()
        plt.imshow(superimposed_img)
        plt.axis("off")
        plt.tight_layout()
        plt.show()
    return superimposed_img, heatmap, preds


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
    if "avg" in method.lower():
        dd = tf.nn.avg_pool(dd, rate, rate, "VALID").numpy()
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
    elif check_type_is("coatnet") or check_type_is("cmt") or check_type_is("uniformer"):
        # bot attn_score [batch, num_heads, hh * ww, hh * ww]
        print(">>>> Attention type: coatnet / cmt / uniformer")
        mask = [np.array(ii)[0].mean((0)) for ii in attn_scores if len(ii.shape) == 4][::-1]
        cum_mask = [mask[0]] + [down_sample_matrix_axis_0(mask[ii], mask[ii - 1].shape[1], "max") for ii in range(1, len(mask))]
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
        return None, None

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
