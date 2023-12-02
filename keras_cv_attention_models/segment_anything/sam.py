import numpy as np
import keras_cv_attention_models
from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models, functional, image_data_format, initializers
from keras_cv_attention_models.attention_layers import BiasLayer, PureWeigths, batchnorm_with_activation, conv2d_no_bias, layer_norm
from keras_cv_attention_models.models import register_model, FakeModelWrapper
from keras_cv_attention_models.download_and_load import reload_model_weights
from keras_cv_attention_models.segment_anything import image_encoders, mask_decoder

LAYER_NORM_EPSILON = 1e-6

PRETRAINED_DICT = {
    "bboxes_encoder": {"mobile_sam_5m": "19e3fcdcdd927ecc67943a3062569d5f", "efficientvit_l0": "a305c26af66a2c07052545341dcd8163"},
    "empty_mask": {"mobile_sam_5m": "304d75d6f1b2c68e8010192086c3b38d", "efficientvit_l0": "f654dc9d837e6c69902ae744ace6f779"},
    "mask_encoder": {"mobile_sam_5m": "2b20607797a03734b2002ae2e5256f62", "efficientvit_l0": "8c9528749eb302b9f157e2eecc19807d"},
    "points_encoder": {"mobile_sam_5m": "bbc785ad50937da738259d3ddf64b1f3", "efficientvit_l0": "a5c1cd344447c43203ee4fbd812eda64"},
    "positional_embedding": {"mobile_sam_5m": "2d97b3faee52a82551e6d2d6562b394e", "efficientvit_l0": "a57e860ed79b0c18bb9975eed2951ccb"},
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
    model = models.Sequential([layers.Input([2, embed_dims]), BiasLayer(axis=[1, 2], initializer=embeddings_initializer, name="bboxes_bias")], name=name)
    reload_model_weights(model, PRETRAINED_DICT, "segment_anything", pretrained)
    return model


def MaskEncoder(embed_dims=256, mask_in_chans=16, activation="gelu", pretrained="mobile_sam_5m", name="mask_encoder"):
    model = models.Sequential(
        [
            layers.Input(backend.align_input_shape_by_image_data_format([None, None, 1])),
            layers.Conv2D(mask_in_chans // 4, kernel_size=2, strides=2, name="1_conv"),
            layers.LayerNormalization(epsilon=LAYER_NORM_EPSILON, name="1_ln"),
            layers.Activation(activation=activation),
            layers.Conv2D(mask_in_chans, kernel_size=2, strides=2, name="2_conv"),
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


def PositionEmbeddingRandom(embed_dims=256, scale=1.0, pretrained="mobile_sam_5m", name="positional_embedding"):
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
    def __init__(self, embed_dims=256, image_input_shape=(1024, 1024), mask_in_chans=16, pretrained="mobile_sam_5m", name="sam"):
        self.image_input_shape = image_input_shape[:2] if isinstance(image_input_shape, (list, tuple)) else image_input_shape
        self.embed_dims = embed_dims

        self.image_encoder = image_encoders.get_image_encoder(pretrained, input_shape=[*image_input_shape, 3], embed_dims=embed_dims)
        self.image_embedding_shape = self.image_encoder.output_shape[1:-1] if image_data_format() == "channels_last" else self.image_encoder.output_shape[2:]

        self.points_encoder = PointsEncoder(embed_dims=embed_dims, pretrained=pretrained)
        self.bboxes_encoder = BboxesEncoder(embed_dims=embed_dims, pretrained=pretrained)
        self.mask_encoder = MaskEncoder(embed_dims=embed_dims, mask_in_chans=mask_in_chans, pretrained=pretrained)
        self.empty_masks_model = EmptyMask(embed_dims=embed_dims, pretrained=pretrained)
        self.positional_embedding = PositionEmbeddingRandom(embed_dims=embed_dims, pretrained=pretrained)
        self.mask_decoder = mask_decoder.MaskDecoder(input_shape=[*self.image_embedding_shape, embed_dims], pretrained=pretrained)
        self.models = [self.image_encoder, self.points_encoder, self.bboxes_encoder, self.mask_encoder, self.empty_masks_model, self.positional_embedding]
        super().__init__(self.models, name=name)

        self.positional_encoding_gaussian_matrix = self.positional_embedding.get_layer("positional_embedding").get_weights()[0]
        self.empty_points = functional.convert_to_tensor(np.empty([1, 0, self.embed_dims]).astype("float32"))
        self.empty_bboxes = functional.convert_to_tensor(np.empty([1, 0, self.embed_dims]).astype("float32"))
        self.empty_masks = self.empty_masks_model.get_layer("empty_masks").get_weights()[0]

        grid = np.ones(self.image_embedding_shape, dtype="float32")
        grid = np.stack([grid.cumsum(axis=1) - 0.5, grid.cumsum(axis=0) - 0.5], axis=-1)  # [height, width, 2]
        self.grid_positional_embedding = self.normalize_coords(grid, self.image_embedding_shape[0], self.image_embedding_shape[1])[None]

    def normalize_coords(self, coords, height, width):
        coords = coords / [height, width]
        coords = (2 * coords - 1) * (2 * np.pi)
        coords = coords @ self.positional_encoding_gaussian_matrix  # [1, 1, 2] @ [2, 128] -> [1, 1, 128]
        return np.concatenate([np.sin(coords), np.cos(coords)], axis=-1).astype("float32")  # [1, 1, 256]

    def preprocess_image(self, image, scaled_height, scaled_width):
        """Aspect awared image resize -> Normalize to ~[-2, 2] -> pad to self.image_input_shape"""
        image = backend.numpy_image_resize(image.astype("float32"), [scaled_height, scaled_width], method="bicubic", antialias=True)
        mean, std = np.array([123.675, 116.28, 103.53]).astype("float32"), np.array([58.395, 57.12, 57.375]).astype("float32")
        normed_image = (image - mean) / std
        pad_height, pad_width = self.image_input_shape[0] - normed_image.shape[0], self.image_input_shape[1] - normed_image.shape[1]
        padded_image = np.pad(normed_image, [[0, pad_height], [0, pad_width], [0, 0]])[None] if pad_height > 0 or pad_width > 0 else normed_image[None]
        return padded_image if image_data_format() == "channels_last" else padded_image.transpose([0, 3, 1, 2])

    def preprocess_points(self, points, height_scale, width_scale, pad=False):
        points = points.reshape([1, -1, 2]) * [height_scale, width_scale] + 0.5  # Shift to center of pixel
        points = self.normalize_coords(points, self.image_input_shape[1], self.image_input_shape[0])  # [TODO] 0, 1 ???
        return np.pad(points, [[0, 0], [0, 1], [0, 0]]) if pad else points

    def preprocess_boxes(self, boxes, height_scale, width_scale):
        boxes = boxes.reshape([1, -1, 2, 2]) * [height_scale, width_scale] + 0.5  # Shift to center of pixel
        return self.normalize_coords(boxes, self.image_input_shape[1], self.image_input_shape[0])  # [TODO] 0, 1 ???

    def __call__(self, image, points=None, labels=None, boxes=None, masks=None, mask_threshold=0, return_logits=False):
        orign_height, orign_width = image.shape[:2]
        scale = min(self.image_input_shape[0] / orign_height, self.image_input_shape[1] / orign_width)
        scaled_height, scaled_width = int(orign_height * scale + 0.5), int(orign_width * scale + 0.5)
        height_scale, width_scale = scaled_height / orign_height, scaled_width / orign_width
        # print(f"{scaled_height = }, {scaled_width = }, {height_scale = }, {width_scale = }")

        """ prompt_encoder """
        image_embeddings = self.image_encoder(self.preprocess_image(image, scaled_height, scaled_width))
        image_embeddings = image_embeddings if image_data_format() == "channels_last" else functional.transpose(image_embeddings, [0, 2, 3, 1])

        boxes_inputs = self.bboxes_encoder(self.preprocess_boxes(boxes, height_scale, width_scale)) if boxes is not None else self.empty_bboxes
        if points is not None and labels is not None:
            pad = boxes is None
            points = self.preprocess_points(points, height_scale, width_scale, pad=pad)
            labels = np.pad(labels.reshape([1, -1]), [[0, 0], [0, 1]], constant_values=-1) if pad else labels
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

        """ Remove padding and resize masks to the original image size. """
        masks = backend.numpy_image_resize(low_res_masks, self.image_input_shape, method="bilinear")
        masks = masks[:, :scaled_height, :scaled_width] if image_data_format() == "channels_last" else masks[:, :, :scaled_height, :scaled_width]
        masks = backend.numpy_image_resize(masks, [orign_height, orign_width], method="bilinear")
        masks = masks if return_logits else (masks > mask_threshold)
        return masks, iou_predictions, low_res_masks

    @staticmethod
    def show(image, masks, iou_predictions=None, points=None, labels=None, boxes=None, save_path=None, base_size=10, random_color=False, marker_size=375):
        import matplotlib.pyplot as plt

        reduce_array_batch_dim = lambda value, ndim: (np.array(value)[0] if np.ndim(value) > ndim else np.array(value)) if value is not None else None
        masks, iou_predictions = reduce_array_batch_dim(masks, ndim=3), reduce_array_batch_dim(iou_predictions, ndim=1)
        points, labels, boxes = reduce_array_batch_dim(points, ndim=2), reduce_array_batch_dim(labels, ndim=1), reduce_array_batch_dim(boxes, ndim=2)

        total = masks.shape[-1]
        fig, axes = plt.subplots(1, total, figsize=(base_size * total, base_size))
        base_color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
        for id, ax in enumerate(axes):
            ax.imshow(image)

            """ show_mask """
            color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0) if random_color else base_color
            mask_image = np.expand_dims(masks[:, :, id], -1) * color.reshape(1, 1, -1)
            ax.imshow(mask_image)

            """ show_points """
            if points is not None and labels is not None:
                pos_points, neg_points = points[labels == 1], points[labels == 0]
                ax.scatter(pos_points[:, 0], pos_points[:, 1], color="green", marker="*", s=marker_size, edgecolor="white", linewidth=1.25)
                ax.scatter(neg_points[:, 0], neg_points[:, 1], color="red", marker="*", s=marker_size, edgecolor="white", linewidth=1.25)

            """ show_box """
            if boxes is not None:
                x0, y0 = boxes[0], boxes[1]
                weight, height = boxes[2] - boxes[0], boxes[3] - boxes[1]
                ax.add_patch(plt.Rectangle((x0, y0), weight, height, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2))

            iou_score_str = " , Score: {:.3f}".format(iou_predictions[id]) if iou_predictions is not None else ""
            ax.set_title("Mask {}".format(id) + iou_score_str, fontsize=18 / 10 * base_size)
            ax.set_axis_off()
        if save_path is not None:
            save_path = save_path if save_path.split(".")[-1].lower() in ["jpg", "png"] else (save_path + ".jpg")
            fig.savefig(save_path, bbox_inches="tight")
        # plt.show()
        return fig
