import numpy as np
from PIL import Image
from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models, functional, image_data_format, initializers
from keras_cv_attention_models.attention_layers import BiasLayer, PureWeigths, batchnorm_with_activation, conv2d_no_bias, layer_norm
from keras_cv_attention_models.models import register_model, FakeModelWrapper
from keras_cv_attention_models.download_and_load import reload_model_weights
from keras_cv_attention_models.segment_anything import image_encoders, mask_decoder

LAYER_NORM_EPSILON = 1e-6

PRETRAINED_DICT = {
    "mobile_sam_5m_bboxes_encoder": {"sam": "19e3fcdcdd927ecc67943a3062569d5f"},
    "mobile_sam_5m_empty_mask": {"sam": "304d75d6f1b2c68e8010192086c3b38d"},
    "mobile_sam_5m_mask_encoder": {"sam": "2b20607797a03734b2002ae2e5256f62"},
    "mobile_sam_5m_points_encoder": {"sam": "bbc785ad50937da738259d3ddf64b1f3"},
    "mobile_sam_5m_positional_embedding": {"sam": "2d97b3faee52a82551e6d2d6562b394e"},
    "efficientvit_sam_l0_bboxes_encoder": {"sam": "a305c26af66a2c07052545341dcd8163"},
    "efficientvit_sam_l0_empty_mask": {"sam": "f654dc9d837e6c69902ae744ace6f779"},
    "efficientvit_sam_l0_mask_encoder": {"sam": "8c9528749eb302b9f157e2eecc19807d"},
    "efficientvit_sam_l0_points_encoder": {"sam": "a5c1cd344447c43203ee4fbd812eda64"},
    "efficientvit_sam_l0_positional_embedding": {"sam": "a57e860ed79b0c18bb9975eed2951ccb"},
}


""" PromptEncoder """


def PointsEncoder(embed_dims=256, pretrained="mobile_sam_5m", name="points_encoder"):
    points = layers.Input([None, embed_dims], name="points")
    labels = layers.Input([None], dtype="int64", name="labels")
    nn = layers.Embedding(3, embed_dims, name="points_embedding")(labels + 1)  # labels [-1, 0, 1] -> [0, 1, 2]
    nn = points + nn

    model = models.Model([points, labels], nn, name=name)
    reload_model_weights(model, PRETRAINED_DICT, "segment_anything", pretrained)
    return model


def BboxesEncoder(embed_dims=256, pretrained="mobile_sam_5m", name="bboxes_encoder"):
    embeddings_initializer = initializers.RandomNormal(mean=0, stddev=1)
    model = models.Sequential([layers.Input([2, embed_dims]), BiasLayer(axis=[-2, -1], initializer=embeddings_initializer, name="bboxes_bias")], name=name)
    reload_model_weights(model, PRETRAINED_DICT, "segment_anything", pretrained)
    return model


def MaskEncoder(embed_dims=256, mask_hidden_dims=16, activation="gelu", pretrained="mobile_sam_5m", name="mask_encoder"):
    model = models.Sequential(
        [
            layers.Input(backend.align_input_shape_by_image_data_format([None, None, 1])),
            layers.Conv2D(mask_hidden_dims // 4, kernel_size=2, strides=2, name="1_conv"),
            layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON, name="1_ln"),
            layers.Activation(activation=activation),
            layers.Conv2D(mask_hidden_dims, kernel_size=2, strides=2, name="2_conv"),
            layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON, name="2_ln"),
            layers.Activation(activation=activation),
            layers.Conv2D(embed_dims, kernel_size=1, name="output_conv"),
        ],
        name=name,
    )
    reload_model_weights(model, PRETRAINED_DICT, "segment_anything", pretrained)
    return model


def EmptyMask(embed_dims=256, pretrained="mobile_sam_5m", name="empty_mask"):
    model = models.Sequential([layers.Input([]), PureWeigths(shape=[1, 256], name="empty_masks")], name=name)
    reload_model_weights(model, PRETRAINED_DICT, "segment_anything", pretrained)
    return model


def PositionEmbeddingRandom(embed_dims=256, scale=1.0, pretrained="sam", name="positional_embedding"):
    scale = 1.0 if scale <= 0.0 else scale
    positional_initializer = initializers.RandomNormal(mean=0, stddev=scale)
    model = models.Sequential(
        [layers.Input([]), PureWeigths(shape=[2, embed_dims // 2], initializer=positional_initializer, name="positional_embedding")], name=name
    )
    reload_model_weights(model, PRETRAINED_DICT, "segment_anything", pretrained)
    return model


""" SAM """


@register_model
class SAM(FakeModelWrapper):  # FakeModelWrapper providing save / load / cuda class methods
    def __init__(
        self, image_encoder="TinyViT_5M", image_shape=(1024, 1024), embed_dims=256, mask_hidden_dims=16, pretrained="sam", name="mobile_sam_5m", kwargs=None
    ):
        self.image_shape = image_shape[:2] if isinstance(image_shape, (list, tuple)) else [image_shape, image_shape]
        self.embed_dims = embed_dims

        self.image_encoder = image_encoders.get_image_encoder(
            image_encoder, input_shape=[*self.image_shape, 3], embed_dims=embed_dims, pretrained=pretrained, name=name + "_image_encoder"
        )
        self.image_embedding_shape = self.image_encoder.output_shape[1:-1] if image_data_format() == "channels_last" else self.image_encoder.output_shape[2:]
        self.prompt_mask_shape = [int(self.image_embedding_shape[0] * 16), int(self.image_embedding_shape[1] * 16)]  # [64, 64] -> [1024, 1024]

        self.points_encoder = PointsEncoder(embed_dims=embed_dims, pretrained=pretrained, name=name + "_points_encoder")
        self.bboxes_encoder = BboxesEncoder(embed_dims=embed_dims, pretrained=pretrained, name=name + "_bboxes_encoder")
        self.mask_encoder = MaskEncoder(embed_dims=embed_dims, mask_hidden_dims=mask_hidden_dims, pretrained=pretrained, name=name + "_mask_encoder")
        self.empty_masks_model = EmptyMask(embed_dims=embed_dims, pretrained=pretrained, name=name + "_empty_mask")
        self.positional_embedding = PositionEmbeddingRandom(embed_dims=embed_dims, pretrained=pretrained, name=name + "_positional_embedding")
        self.mask_decoder = mask_decoder.MaskDecoder(input_shape=[*self.image_embedding_shape, embed_dims], pretrained=pretrained, name=name + "_mask_decoder")
        self.models = [self.image_encoder, self.points_encoder, self.bboxes_encoder, self.mask_encoder, self.empty_masks_model, self.positional_embedding]
        super().__init__(self.models, name=name)

        self.positional_encoding_gaussian_matrix = self.positional_embedding.get_layer("positional_embedding").get_weights()[0]
        self.empty_points = functional.convert_to_tensor(np.empty([1, 0, self.embed_dims]).astype("float32"))
        self.empty_bboxes = functional.convert_to_tensor(np.empty([1, 0, self.embed_dims]).astype("float32"))
        self.empty_masks = self.empty_masks_model.get_layer("empty_masks").get_weights()[0]

        grid = np.ones(self.image_embedding_shape, dtype="float32")
        grid = np.stack([grid.cumsum(axis=1) - 0.5, grid.cumsum(axis=0) - 0.5], axis=-1)  # [height, width, 2]
        self.grid_positional_embedding = self.normalize_coords(grid, self.image_embedding_shape[0], self.image_embedding_shape[1])[None]
        self.image_mean, self.image_std = np.array([123.675, 116.28, 103.53]).astype("float32"), np.array([58.395, 57.12, 57.375]).astype("float32")

    def normalize_coords(self, coords, height, width):
        coords = coords / [height, width]
        coords = (2 * coords - 1) * (2 * np.pi)
        coords = coords @ self.positional_encoding_gaussian_matrix  # [1, 1, 2] @ [2, 128] -> [1, 1, 128]
        return np.concatenate([np.sin(coords), np.cos(coords)], axis=-1).astype("float32")  # [1, 1, 256]

    def preprocess_image(self, image):
        """Aspect awared image resize -> Normalize to ~[-2, 2] -> pad to self.image_shape"""
        orign_height, orign_width = image.shape[:2]
        scale = min(self.image_shape[0] / orign_height, self.image_shape[1] / orign_width)
        scaled_height, scaled_width = int(orign_height * scale + 0.5), int(orign_width * scale + 0.5)

        # image = np.clip(backend.numpy_image_resize(image.astype("float32"), [scaled_height, scaled_width], method="bicubic", antialias=True), 0, 255)
        image = np.array(Image.fromarray(image).resize([scaled_width, scaled_height]))
        normed_image = (image - self.image_mean) / self.image_std
        pad_height, pad_width = self.image_shape[0] - normed_image.shape[0], self.image_shape[1] - normed_image.shape[1]
        # print(f"{pad_height = }, {pad_width = }")
        padded_image = np.pad(normed_image, [[0, pad_height], [0, pad_width], [0, 0]])[None] if pad_height > 0 or pad_width > 0 else normed_image[None]
        return padded_image if image_data_format() == "channels_last" else padded_image.transpose([0, 3, 1, 2])

    def coords_scale_and_norm(self, coords, height_scale, width_scale, scaled_height, scaled_width):
        coords = np.array(coords, dtype="float32")
        coords *= [width_scale, height_scale] if coords.max() > 1 else [scaled_width, scaled_height]  # points or bboxes coords is [left, top] format
        coords += 0.5
        return self.normalize_coords(coords, self.prompt_mask_shape[1], self.prompt_mask_shape[0])  # points or bboxes coords is [left, top] format

    def __call__(self, image, points=None, labels=None, boxes=None, masks=None, mask_threshold=0, return_logits=False):
        orign_height, orign_width = image.shape[:2]
        scale = min(self.prompt_mask_shape[0] / orign_height, self.prompt_mask_shape[1] / orign_width)
        scaled_height, scaled_width = int(orign_height * scale + 0.5), int(orign_width * scale + 0.5)
        height_scale, width_scale = scaled_height / orign_height, scaled_width / orign_width
        # print(f"{scaled_height = }, {scaled_width = }, {height_scale = }, {width_scale = }")

        """ image_encoder """
        image_embeddings = self.image_encoder(self.preprocess_image(image))
        image_embeddings = image_embeddings if image_data_format() == "channels_last" else functional.transpose(image_embeddings, [0, 2, 3, 1])

        """ prompt_encoder """
        if boxes is not None:
            boxes = np.array(boxes, dtype="float32").reshape([-1, 2, 2])
            boxes = self.coords_scale_and_norm(boxes, height_scale, width_scale, scaled_height, scaled_width)
            boxes_inputs = self.bboxes_encoder(boxes)
        else:
            boxes_inputs = self.empty_bboxes

        if points is not None and labels is not None:
            points, labels = np.array(points, dtype="float32").reshape([1, -1, 2]), np.array(labels, dtype="float32").reshape([1, -1])
            points = self.coords_scale_and_norm(points, height_scale, width_scale, scaled_height, scaled_width)
            points = np.pad(points, [[0, 0], [0, 1], [0, 0]]) if boxes is None else points
            labels = np.pad(labels, [[0, 0], [0, 1]], constant_values=-1) if boxes is None else labels

            assert points.shape[1] == labels.shape[-1]
            points_inputs = self.points_encoder([points, labels])
        else:
            points_inputs = self.empty_points
        sparse_embeddings = functional.concat([points_inputs, boxes_inputs], axis=1)
        dense_embeddings = self.mask_encoder(masks) if masks is not None else self.empty_masks  # [TODO], channels_first and check masks shape

        """ mask_decoder """
        image_with_masks_inputs = image_embeddings + dense_embeddings
        # print(f"{image_with_masks_inputs.shape = }, {sparse_embeddings.shape = }, {self.grid_positional_embedding.shape = }")
        low_res_masks, iou_predictions = self.mask_decoder([image_with_masks_inputs, sparse_embeddings, self.grid_positional_embedding])
        low_res_masks = low_res_masks.cpu().numpy() if backend.is_torch_backend else low_res_masks.numpy()
        iou_predictions = iou_predictions.cpu().numpy() if backend.is_torch_backend else iou_predictions.numpy()

        """ Remove padding and resize masks to the original image size """
        masks = backend.numpy_image_resize(low_res_masks, self.prompt_mask_shape, method="bilinear")
        masks = masks[:, :scaled_height, :scaled_width] if image_data_format() == "channels_last" else masks[:, :, :scaled_height, :scaled_width]
        masks = backend.numpy_image_resize(masks, [orign_height, orign_width], method="bilinear")
        masks = masks if return_logits else (masks > mask_threshold)

        # For a single image, batch_size is always 1. Converting masks `[1, height, width, 4]` -> `[4, height, width]`
        masks, iou_predictions, low_res_masks = masks[0].transpose([2, 0, 1]), iou_predictions[0], low_res_masks[0].transpose([2, 0, 1])
        return masks, iou_predictions, low_res_masks

    @staticmethod
    def show(image, masks, iou_predictions=None, points=None, labels=None, boxes=None, save_path=None, base_size=10, random_color=False):
        import matplotlib.pyplot as plt

        to_array_reshape = lambda value, shape: np.array(value, dtype="float32").reshape(shape) if value is not None else None
        masks, iou_predictions = to_array_reshape(masks, [masks.shape[-3], masks.shape[-2], masks.shape[-1]]), to_array_reshape(iou_predictions, [-1])
        points, labels, boxes = to_array_reshape(points, [-1, 2]), to_array_reshape(labels, [-1]), to_array_reshape(boxes, [-1, 4])
        height, width = image.shape[:2]

        total = masks.shape[0]
        fig, axes = plt.subplots(1, total, figsize=(base_size * total, base_size))
        marker_size, fontsize = 375 / 10 * base_size, 18 / 10 * base_size
        base_color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        for id, ax in enumerate(axes):
            ax.imshow(image)

            """ show_mask """
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0) if random_color else base_color
            mask_image = np.expand_dims(masks[id], -1) * color.reshape(1, 1, -1)
            ax.imshow(mask_image)

            """ show_points """
            if points is not None and labels is not None:
                points = points if points.max() > 1 else (points * [width, height])
                pos_points, neg_points = points[labels == 1], points[labels == 0]
                ax.scatter(pos_points[:, 0], pos_points[:, 1], color="green", marker="*", s=marker_size, edgecolor="white", linewidth=1.25)
                ax.scatter(neg_points[:, 0], neg_points[:, 1], color="red", marker="*", s=marker_size, edgecolor="white", linewidth=1.25)

            """ show_box """
            if boxes is not None:
                boxes = boxes if boxes.max() > 1 else (boxes * [width, height, width, height])
                for box in boxes:
                    left, top, right, bottom = box
                    ax.add_patch(plt.Rectangle((left, top), right - left, bottom - top, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))

            iou_score_str = " , Score: {:.3f}".format(iou_predictions[id]) if iou_predictions is not None else ""
            ax.set_title("Mask {}".format(id) + iou_score_str, fontsize=fontsize)
            ax.set_axis_off()
        if save_path is not None:
            save_path = save_path if save_path.split(".")[-1].lower() in ["jpg", "png"] else (save_path + ".jpg")
            fig.savefig(save_path, bbox_inches="tight")
        # plt.show()
        return fig


@register_model
def MobileSAM(image_shape=(1024, 1024), pretrained="sam", name="mobile_sam_5m", **kwargs):
    return SAM(image_encoder="TinyViT_5M", **locals(), **kwargs)


@register_model
def EfficientViT_SAM_L0(image_shape=(512, 512), pretrained="sam", name="efficientvit_sam_l0", **kwargs):
    mask_decoder.LAYER_NORM_EPSILON = 1e-6
    model = SAM(image_encoder="EfficientViT_L0", **locals(), **kwargs)
    mask_decoder.LAYER_NORM_EPSILON = 1e-5
    return model
