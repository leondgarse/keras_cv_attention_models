from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models, functional, image_data_format, initializers
from keras_cv_attention_models.download_and_load import reload_model_weights
from keras_cv_attention_models.attention_layers import ClassToken, layer_norm, conv2d_no_bias, activation_by_name, add_pre_post_process


PRETRAINED_DICT = {
    "coat_lite_tiny": {"imagenet": "e45487e7bfb44faac97b1af51f8bbd01"},
    "coat_lite_mini": {"imagenet": "e5e3f5e4b86765ee75f8bf03973d70a0"},
    "coat_lite_small": {"imagenet": "eddffc46a64eb0a21b7ecc057f231756"},
    "coat_tiny": {"imagenet": "6418d9580ad9ea0a6755c77d8d7bad49"},
    "coat_mini": {"imagenet": "dc284967f6bd32df8e1e03074b2d773d"},
}


# Not a layer, just for reusable
class ConvPositionalEncoding:
    def __init__(self, kernel_size=3, input_height=-1, name=None):
        self.kernel_size, self.input_height, self.name = kernel_size, input_height, name
        if image_data_format() == "channels_last":
            self.pad = [[0, 0], [kernel_size // 2, kernel_size // 2], [kernel_size // 2, kernel_size // 2], [0, 0]]
        else:
            self.pad = [[0, 0], [0, 0], [kernel_size // 2, kernel_size // 2], [kernel_size // 2, kernel_size // 2]]
        self.built = False

    def build(self, input_shape):
        self.height = self.input_height if self.input_height > 0 else int(float(input_shape[1] - 1) ** 0.5)
        self.width = (input_shape[1] - 1) // self.height

        self.channel = input_shape[-1]
        self.dconv = layers.DepthwiseConv2D(self.kernel_size, strides=1, padding="VALID", name=self.name and self.name + "depth_conv")

    def __call__(self, inputs, **kwargs):
        if not self.built:
            self.build(inputs.shape)
            self.built = True

        cls_token, img_token = inputs[:, :1], inputs[:, 1:]
        img_token = functional.reshape(img_token, [-1, self.height, self.width, self.channel])
        nn = img_token if image_data_format() == "channels_last" else layers.Permute([3, 1, 2])(img_token)
        # print(f"{nn.shape = }")
        nn = self.dconv(functional.pad(nn, self.pad))
        nn = nn if image_data_format() == "channels_last" else layers.Permute([2, 3, 1])(nn)
        nn = layers.Add()([nn, img_token])
        nn = functional.reshape(nn, [-1, self.height * self.width, self.channel])
        return functional.concat([cls_token, nn], axis=1)


# Not a layer, just for reusable
class ConvRelativePositionalEncoding:
    def __init__(self, head_splits=[2, 3, 3], head_kernel_size=[3, 5, 7], input_height=-1, name=None):
        self.head_splits, self.head_kernel_size, self.input_height, self.name = head_splits, head_kernel_size, input_height, name
        self.built = False

    def build(self, query_shape):
        # print(query_shape)
        self.height = self.input_height if self.input_height > 0 else int(float(query_shape[2] - 1) ** 0.5)
        self.width = (query_shape[2] - 1) // self.height
        self.num_heads, self.query_dim = query_shape[1], query_shape[-1]
        self.channel_splits = [ii * self.query_dim for ii in self.head_splits]

        self.dconvs = []
        self.pads = []
        for id, (head_split, kernel_size) in enumerate(zip(self.head_splits, self.head_kernel_size)):
            name = self.name and self.name + "depth_conv_" + str(id + 1)
            dconv = layers.DepthwiseConv2D(kernel_size, strides=1, padding="VALID", name=name)
            if image_data_format() == "channels_last":
                pad = [[0, 0], [kernel_size // 2, kernel_size // 2], [kernel_size // 2, kernel_size // 2], [0, 0]]
            else:
                pad = [[0, 0], [0, 0], [kernel_size // 2, kernel_size // 2], [kernel_size // 2, kernel_size // 2]]
            self.dconvs.append(dconv)
            self.pads.append(pad)

    def __call__(self, query, value, **kwargs):
        if not self.built:
            self.build(query.shape)
            self.built = True
        img_token_q, img_token_v = query[:, :, 1:, :], value[:, :, 1:, :]

        if image_data_format() == "channels_last":
            img_token_v = functional.transpose(img_token_v, [0, 2, 1, 3])  # [batch, blocks, num_heads, query_dim]
            img_token_v = functional.reshape(img_token_v, [-1, self.height, self.width, self.num_heads * self.query_dim])
            split_values = functional.split(img_token_v, self.channel_splits, axis=-1)
        else:
            img_token_v = functional.transpose(img_token_v, [0, 1, 3, 2])  # [batch, num_heads, query_dim, blocks]
            img_token_v = functional.reshape(img_token_v, [-1, self.num_heads * self.query_dim, self.height, self.width])
            split_values = functional.split(img_token_v, self.channel_splits, axis=1)
        nn = [dconv(functional.pad(split_value, pad)) for split_value, dconv, pad in zip(split_values, self.dconvs, self.pads)]

        if image_data_format() == "channels_last":
            nn = functional.concat(nn, axis=-1)
            conv_v_img = functional.reshape(nn, [-1, self.height * self.width, self.num_heads, self.query_dim])
            conv_v_img = functional.transpose(conv_v_img, [0, 2, 1, 3])
        else:
            nn = functional.concat(nn, axis=1)
            conv_v_img = functional.reshape(nn, [-1, self.num_heads, self.query_dim, self.height * self.width])
            conv_v_img = functional.transpose(conv_v_img, [0, 1, 3, 2])

        EV_hat_img = img_token_q * conv_v_img
        return functional.pad(EV_hat_img, [[0, 0], [0, 0], [1, 0], [0, 0]])


def factor_attention_conv_relative_positional_encoding(inputs, shared_crpe=None, num_heads=8, qkv_bias=True, name=""):
    blocks, dim = inputs.shape[1], inputs.shape[-1]
    key_dim = dim // num_heads
    qk_scale = 1.0 / (float(key_dim) ** 0.5)

    qkv = layers.Dense(dim * 3, use_bias=qkv_bias, name=name + "qkv")(inputs)
    qkv = layers.Reshape([blocks, 3, num_heads, key_dim])(qkv)
    qq, kk, vv = functional.transpose(qkv, [2, 0, 3, 1, 4])  # [qkv, batch, num_heads, blocks, key_dim]
    # print(f">>>> {qkv.shape = }, {qq.shape = }, {kk.shape = }, {vv.shape = }")

    # Factorized attention.
    # kk = tf.nn.softmax(kk, axis=2)  # On `blocks` dimension
    kk = layers.Softmax(axis=2, name=name and name + "attention_scores")(kk)  # On `blocks` dimension
    kk = functional.transpose(kk, [0, 1, 3, 2])
    factor_att = qq @ (kk @ vv)

    # Convolutional relative position encoding.
    crpe_out = shared_crpe(qq, vv) if shared_crpe is not None else ConvRelativePositionalEncoding(name=name + "crpe_")(qq, vv)

    # Merge and reshape.
    nn = layers.Add()([factor_att * qk_scale, crpe_out])
    nn = layers.Permute([2, 1, 3])(nn)
    nn = layers.Reshape([blocks, dim])(nn)
    nn = layers.Dense(dim, name=name + "out")(nn)
    return nn


def cpe_norm_crpe(inputs, shared_cpe=None, shared_crpe=None, num_heads=8, name=""):
    cpe_out = shared_cpe(inputs) if shared_cpe is not None else ConvPositionalEncoding(name=name + "cpe_")(inputs)  # shared
    nn = layer_norm(cpe_out, axis=-1, name=name + "norm1")
    crpe_out = factor_attention_conv_relative_positional_encoding(nn, shared_crpe=shared_crpe, num_heads=num_heads, name=name + "factoratt_crpe_")
    return cpe_out, crpe_out


def res_mlp_block(cpe_out, crpe_out, mlp_ratio=4, drop_rate=0, activation="gelu", name=""):
    if drop_rate > 0:
        crpe_out = layers.Dropout(drop_rate, noise_shape=(None, 1, 1), name=name + "drop_1")(crpe_out)
    cpe_crpe = layers.Add()([cpe_out, crpe_out])

    # MLP
    pre_mlp = layer_norm(cpe_crpe, axis=-1, name=name + "norm2")
    nn = layers.Dense(pre_mlp.shape[-1] * mlp_ratio, name=name + "mlp_dense_0")(pre_mlp)
    nn = activation_by_name(nn, activation, name=name + "mlp_")
    nn = layers.Dense(pre_mlp.shape[-1], name=name + "mlp_dense_1")(nn)

    if drop_rate > 0:
        nn = layers.Dropout(drop_rate, noise_shape=(None, 1, 1), name=name + "drop_2")(nn)
    return layers.Add(name=name + "output")([cpe_crpe, nn])


def serial_block(inputs, embed_dim, shared_cpe=None, shared_crpe=None, num_heads=8, mlp_ratio=4, drop_rate=0, activation="gelu", name=""):
    cpe_out, crpe_out = cpe_norm_crpe(inputs, shared_cpe, shared_crpe, num_heads, name=name)
    out = res_mlp_block(cpe_out, crpe_out, mlp_ratio, drop_rate, activation=activation, name=name)
    return out


def resample(image, target_shape, class_token=None):
    # print(f"{image.shape = }, {target_shape = }")
    image = image if image_data_format() == "channels_last" else layers.Permute([3, 1, 2])(image)
    # out_image = functional.cast(functional.resize(image, target_shape, method="bilinear"), image.dtype)
    out_image = functional.resize(image, target_shape, method="bilinear")
    out_image = out_image if image_data_format() == "channels_last" else layers.Permute([2, 3, 1])(out_image)

    if class_token is not None:
        out_image = functional.reshape(out_image, [-1, out_image.shape[1] * out_image.shape[2], out_image.shape[-1]])
        return functional.concat([class_token, out_image], axis=1)
    else:
        return out_image


def parallel_block(inputs, shared_cpes=None, shared_crpes=None, block_heights=[], num_heads=8, mlp_ratios=[], drop_rate=0, activation="gelu", name=""):
    # Conv-Attention.
    # print(f">>>> {block_heights = }")
    cpe_outs, crpe_outs, crpe_images, resample_shapes = [], [], [], []
    block_heights = block_heights[1:]
    for id, (xx, shared_cpe, shared_crpe) in enumerate(zip(inputs[1:], shared_cpes[1:], shared_crpes[1:])):
        cur_name = name + "{}_".format(id + 2)
        cpe_out, crpe_out = cpe_norm_crpe(xx, shared_cpe, shared_crpe, num_heads, name=cur_name)
        cpe_outs.append(cpe_out)
        crpe_outs.append(crpe_out)
        height = block_heights[id] if len(block_heights) > id else int(float(crpe_out.shape[1] - 1) ** 0.5)
        width = (crpe_out.shape[1] - 1) // height
        crpe_images.append(functional.reshape(crpe_out[:, 1:, :], [-1, height, width, crpe_out.shape[-1]]))
        resample_shapes.append([height, width])
        # print(f">>>> {crpe_out.shape = }, {crpe_images[-1].shape = }")
    crpe_stack = [  # [[None, 28, 28, 152], [None, 14, 14, 152], [None, 7, 7, 152]]
        crpe_outs[0] + resample(crpe_images[1], resample_shapes[0], crpe_outs[1][:, :1]) + resample(crpe_images[2], resample_shapes[0], crpe_outs[2][:, :1]),
        crpe_outs[1] + resample(crpe_images[2], resample_shapes[1], crpe_outs[2][:, :1]) + resample(crpe_images[0], resample_shapes[1], crpe_outs[0][:, :1]),
        crpe_outs[2] + resample(crpe_images[1], resample_shapes[2], crpe_outs[1][:, :1]) + resample(crpe_images[0], resample_shapes[2], crpe_outs[0][:, :1]),
    ]

    # MLP
    outs = []
    for id, (cpe_out, crpe_out, mlp_ratio) in enumerate(zip(cpe_outs, crpe_stack, mlp_ratios[1:])):
        cur_name = name + "{}_".format(id + 2)
        out = res_mlp_block(cpe_out, crpe_out, mlp_ratio, drop_rate, activation=activation, name=cur_name)
        outs.append(out)
    return inputs[:1] + outs  # inputs[0] directly out


def patch_embed(inputs, embed_dim, patch_size=2, input_height=-1, name=""):
    if len(inputs.shape) == 3:
        input_height = input_height if input_height > 0 else int(float(inputs.shape[1]) ** 0.5)
        input_width = inputs.shape[1] // input_height
        inputs = layers.Reshape([input_height, input_width, inputs.shape[-1]])(inputs)
        inputs = inputs if image_data_format() == "channels_last" else layers.Permute([3, 1, 2])(inputs)
    nn = conv2d_no_bias(inputs, embed_dim, kernel_size=patch_size, strides=patch_size, use_bias=True, name=name)
    nn = nn if image_data_format() == "channels_last" else layers.Permute([2, 3, 1])(nn)
    block_height = nn.shape[1]
    nn = layers.Reshape([nn.shape[1] * nn.shape[2], nn.shape[3]])(nn)  # flatten(2)
    nn = layer_norm(nn, axis=-1, name=name)
    return nn, block_height


def CoaT(
    serial_depths=[2, 2, 2, 2],
    embed_dims=[64, 128, 256, 320],
    mlp_ratios=[8, 8, 4, 4],
    parallel_depth=0,
    patch_size=4,
    num_heads=8,
    head_splits=[2, 3, 3],
    head_kernel_size=[3, 5, 7],
    use_shared_cpe=True,  # For checking model architecture only, keep input_shape height == width if set False
    use_shared_crpe=True,  # For checking model architecture only, keep input_shape height == width if set False
    out_features=None,
    input_shape=(224, 224, 3),
    num_classes=1000,
    activation="gelu",
    drop_connect_rate=0,
    classifier_activation="softmax",
    pretrained="imagenet",
    model_name="coat",
    kwargs=None,
):
    # Regard input_shape as force using original shape if len(input_shape) == 4,
    # else assume channel dimention is the one with min value in input_shape, and put it first or last regarding image_data_format
    input_shape = backend.align_input_shape_by_image_data_format(input_shape)
    inputs = layers.Input(input_shape)

    # serial blocks
    nn = inputs
    classfier_outs = []
    shared_cpes = []
    shared_crpes = []
    block_heights = []
    for sid, (depth, embed_dim, mlp_ratio) in enumerate(zip(serial_depths, embed_dims, mlp_ratios)):
        name = "serial{}_".format(sid + 1)
        patch_size = patch_size if sid == 0 else 2
        patch_input_height = -1 if sid == 0 else block_heights[-1]
        # print(f">>>> {nn.shape = }")
        nn, block_height = patch_embed(nn, embed_dim, patch_size=patch_size, input_height=patch_input_height, name=name + "patch_")
        block_heights.append(block_height)
        # print(f">>>> {nn.shape = }, {block_height = }")
        nn = ClassToken(name=name + "class_token")(nn)
        shared_cpe = ConvPositionalEncoding(kernel_size=3, input_height=block_height, name="cpe{}_".format(sid + 1)) if use_shared_cpe else None
        shared_crpe = ConvRelativePositionalEncoding(head_splits, head_kernel_size, block_height, name="crpe{}_".format(sid + 1)) if use_shared_crpe else None
        for bid in range(depth):
            block_name = name + "block{}_".format(bid + 1)
            nn = serial_block(nn, embed_dim, shared_cpe, shared_crpe, num_heads, mlp_ratio, activation=activation, name=block_name)
        classfier_outs.append(nn)
        shared_cpes.append(shared_cpe)
        shared_crpes.append(shared_crpe)
        nn = nn[:, 1:, :]  # remove class token

    # Parallel blocks.
    for pid in range(parallel_depth):
        name = "parallel{}_".format(pid + 1)
        classfier_outs = parallel_block(classfier_outs, shared_cpes, shared_crpes, block_heights, num_heads, mlp_ratios, activation=activation, name=name)

    if out_features is not None:  # Return intermediate features (for down-stream tasks).
        nn = [classfier_outs[id][:, 1:, :] for id in out_features]
    elif parallel_depth == 0:  # Lite model, only serial blocks, Early return.
        nn = layer_norm(classfier_outs[-1], axis=-1, name="out_")[:, 0]
    else:
        nn = [layer_norm(xx, axis=-1, name="out_{}_".format(id + 1))[:, :1, :] for id, xx in enumerate(classfier_outs[1:])]
        nn = layers.Concatenate(axis=1)(nn)
        nn = layers.Permute([2, 1])(nn) if image_data_format() == "channels_last" else nn
        nn = layers.Conv1D(1, 1, name="aggregate")(nn)
        nn = nn[:, :, 0] if image_data_format() == "channels_last" else nn[:, 0]
    if out_features is None and num_classes > 0:
        nn = layers.Dense(num_classes, dtype="float32", activation=classifier_activation, name="predictions")(nn)

    model = models.Model(inputs, nn, name=model_name)
    add_pre_post_process(model, rescale_mode="torch")
    reload_model_weights(model, pretrained_dict=PRETRAINED_DICT, sub_release="coat", pretrained=pretrained)
    return model


def CoaTLiteTiny(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    return CoaT(**locals(), model_name="coat_lite_tiny", **kwargs)


def CoaTLiteMini(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    embed_dims = [64, 128, 320, 512]
    return CoaT(**locals(), model_name="coat_lite_mini", **kwargs)


def CoaTLiteSmall(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    serial_depths = [3, 4, 6, 3]
    embed_dims = [64, 128, 320, 512]
    return CoaT(**locals(), model_name="coat_lite_small", **kwargs)


def CoaTTiny(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    embed_dims = [152, 152, 152, 152]
    mlp_ratios = [4, 4, 4, 4]
    parallel_depth = 6
    return CoaT(**locals(), model_name="coat_tiny", **kwargs)


def CoaTMini(input_shape=(224, 224, 3), num_classes=1000, activation="gelu", classifier_activation="softmax", pretrained="imagenet", **kwargs):
    embed_dims = [152, 216, 216, 216]
    mlp_ratios = [4, 4, 4, 4]
    parallel_depth = 6
    return CoaT(**locals(), model_name="coat_mini", **kwargs)
