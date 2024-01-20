import numpy as np
from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models, functional, initializers
from keras_cv_attention_models.attention_layers import BiasLayer, PureWeigths
from keras_cv_attention_models.models import register_model, FakeModelWrapper, no_grad_if_torch
from keras_cv_attention_models.download_and_load import reload_model_weights

LAYER_NORM_EPSILON = 1e-6


PRETRAINED_DICT = {
    "sam_prompt_encoder_bboxes_encoder": {"sam": "d2467e477d9b9040fcf5db189ca7463f"},
    "sam_prompt_encoder_empty_mask": {"sam": "04de4195f5f5c30360d2fcf035c5cad3"},
    "sam_prompt_encoder_mask_encoder": {"sam": "9689653538e63fceac9ff7c20fe7ad99"},
    "sam_prompt_encoder_points_encoder": {"sam": "c2259f9a00b6b833c572d28a30b94938"},
    "sam_prompt_encoder_positional_embedding": {"sam": "19803807663d81412989ad5916e14b81"},
}


class PromptEncoder(FakeModelWrapper):  # FakeModelWrapper providing save / load / cuda class methods
    def __init__(
        self, embed_dims=256, mask_hidden_dims=16, prompt_mask_shape=(1024, 1024), masks_input_shape=(64, 64), pretrained="sam", name="sam_prompt_encoder"
    ):
        self.prompt_mask_shape = prompt_mask_shape[:2] if isinstance(prompt_mask_shape, (list, tuple)) else [prompt_mask_shape, prompt_mask_shape]
        self.masks_input_shape = masks_input_shape[:2] if isinstance(masks_input_shape, (list, tuple)) else [masks_input_shape, masks_input_shape]

        self.points_encoder = self.PointsEncoder(embed_dims=embed_dims, pretrained=pretrained, name=name + "_points_encoder")
        self.bboxes_encoder = self.BboxesEncoder(embed_dims=embed_dims, pretrained=pretrained, name=name + "_bboxes_encoder")
        self.mask_encoder = self.MaskEncoder(embed_dims=embed_dims, mask_hidden_dims=mask_hidden_dims, pretrained=pretrained, name=name + "_mask_encoder")
        self.empty_masks_model = self.EmptyMask(embed_dims=embed_dims, pretrained=pretrained, name=name + "_empty_mask")
        self.positional_embedding = self.PositionEmbeddingRandom(embed_dims=embed_dims, pretrained=pretrained, name=name + "_positional_embedding")
        self.models = [self.points_encoder, self.bboxes_encoder, self.mask_encoder, self.empty_masks_model, self.positional_embedding]
        super().__init__(self.models, name=name)

        self.positional_encoding_gaussian_matrix = self.positional_embedding.get_layer("positional_embedding").get_weights()[0]
        self.empty_points = functional.convert_to_tensor(np.empty([1, 0, embed_dims]).astype("float32"))
        self.empty_bboxes = functional.convert_to_tensor(np.empty([1, 0, embed_dims]).astype("float32"))
        empty_masks = self.empty_masks_model.get_layer("empty_masks").get_weights()[0]
        self.empty_masks = empty_masks[:, None, None, :] if backend.image_data_format() == "channels_last" else empty_masks[:, :, None, None]

    def normalize_coords(self, coords, height, width):
        coords = coords / [height, width]
        coords = (2 * coords - 1) * (2 * np.pi)
        coords = coords @ self.positional_encoding_gaussian_matrix  # [1, 1, 2] @ [2, 128] -> [1, 1, 128]
        return np.concatenate([np.sin(coords), np.cos(coords)], axis=-1).astype("float32")  # [1, 1, 256]

    def get_grid_positional_embedding(self, image_embedding_shape=(64, 64)):
        grid = np.ones(image_embedding_shape, dtype="float32")
        grid = np.stack([grid.cumsum(axis=1) - 0.5, grid.cumsum(axis=0) - 0.5], axis=-1)  # [height, width, 2]
        return self.normalize_coords(grid, image_embedding_shape[0], image_embedding_shape[1])[None]

    def coords_scale_and_norm(self, coords, height_scale, width_scale, scaled_height, scaled_width):
        coords = np.array(coords, dtype="float32")
        coords *= [width_scale, height_scale] if coords.max() > 1 else [scaled_width, scaled_height]  # points or bboxes coords is [left, top] format
        coords += 0.5
        return self.normalize_coords(coords, self.prompt_mask_shape[1], self.prompt_mask_shape[0])  # points or bboxes coords is [left, top] format

    @staticmethod
    def PointsEncoder(embed_dims=256, pretrained="sam", name="sam_prompt_encoder_points_encoder"):
        points = layers.Input([None, embed_dims], name="points")
        labels = layers.Input([None], dtype="int64", name="labels")
        nn = layers.Embedding(3, embed_dims, name="points_embedding")(labels + 1)  # labels [-1, 0, 1] -> [0, 1, 2]
        nn = points + nn

        model = models.Model([points, labels], nn, name=name)
        reload_model_weights(model, PRETRAINED_DICT, "segment_anything", pretrained)
        return model

    @staticmethod
    def BboxesEncoder(embed_dims=256, pretrained="sam", name="sam_prompt_encoder_bboxes_encoder"):
        embeddings_initializer = initializers.RandomNormal(mean=0, stddev=1)
        model = models.Sequential([layers.Input([2, embed_dims]), BiasLayer(axis=[-2, -1], initializer=embeddings_initializer, name="bboxes_bias")], name=name)
        reload_model_weights(model, PRETRAINED_DICT, "segment_anything", pretrained)
        return model

    @staticmethod
    def MaskEncoder(embed_dims=256, mask_hidden_dims=16, activation="gelu", pretrained="sam", name="sam_prompt_encoder_mask_encoder"):
        norm_axis = -1 if backend.image_data_format() == "channels_last" else 1
        model = models.Sequential(
            [
                layers.Input(backend.align_input_shape_by_image_data_format([None, None, 1])),
                layers.Conv2D(mask_hidden_dims // 4, kernel_size=2, strides=2, name="1_conv"),
                layers.LayerNormalization(axis=norm_axis, epsilon=LAYER_NORM_EPSILON, name="1_ln"),
                layers.Activation(activation=activation),
                layers.Conv2D(mask_hidden_dims, kernel_size=2, strides=2, name="2_conv"),
                layers.LayerNormalization(axis=norm_axis, epsilon=LAYER_NORM_EPSILON, name="2_ln"),
                layers.Activation(activation=activation),
                layers.Conv2D(embed_dims, kernel_size=1, name="output_conv"),
            ],
            name=name,
        )
        reload_model_weights(model, PRETRAINED_DICT, "segment_anything", pretrained)
        return model

    @staticmethod
    def EmptyMask(embed_dims=256, pretrained="sam", name="sam_prompt_encoder_empty_mask"):
        model = models.Sequential([layers.Input([]), PureWeigths(shape=[1, 256], name="empty_masks")], name=name)
        reload_model_weights(model, PRETRAINED_DICT, "segment_anything", pretrained)
        return model

    @staticmethod
    def PositionEmbeddingRandom(embed_dims=256, scale=1.0, pretrained="sam", name="sam_prompt_encoder_positional_embedding"):
        scale = 1.0 if scale <= 0.0 else scale
        positional_initializer = initializers.RandomNormal(mean=0, stddev=scale)
        model = models.Sequential(
            [layers.Input([]), PureWeigths(shape=[2, embed_dims // 2], initializer=positional_initializer, name="positional_embedding")], name=name
        )
        reload_model_weights(model, PRETRAINED_DICT, "segment_anything", pretrained)
        return model

    @no_grad_if_torch
    def __call__(self, image_orign_shape, points=None, labels=None, boxes=None, masks=None):
        image_orign_height, image_orign_width = image_orign_shape[:2]
        scale = min(self.prompt_mask_shape[0] / image_orign_height, self.prompt_mask_shape[1] / image_orign_width)
        scaled_height, scaled_width = int(image_orign_height * scale + 0.5), int(image_orign_width * scale + 0.5)
        height_scale, width_scale = scaled_height / image_orign_height, scaled_width / image_orign_width
        # print(f"{scaled_height = }, {scaled_width = }, {height_scale = }, {width_scale = }")

        """ boxes """
        if boxes is not None:
            boxes = np.array(boxes, dtype="float32").reshape([-1, 2, 2])
            assert boxes.shape[0] == 1, "Supports only single boxes as inputs"
            boxes = self.coords_scale_and_norm(boxes, height_scale, width_scale, scaled_height, scaled_width)
            boxes_inputs = self.bboxes_encoder(boxes)
        else:
            boxes_inputs = self.empty_bboxes

        """ points and labels """
        if points is not None and labels is not None:
            points, labels = np.array(points, dtype="float32").reshape([1, -1, 2]), np.array(labels, dtype="float32").reshape([1, -1])
            points = self.coords_scale_and_norm(points, height_scale, width_scale, scaled_height, scaled_width)
            points = np.pad(points, [[0, 0], [0, 1], [0, 0]]) if boxes is None else points
            labels = np.pad(labels, [[0, 0], [0, 1]], constant_values=-1) if boxes is None else labels

            assert points.shape[1] == labels.shape[-1], "Should provide same number of points and labels"
            points_inputs = self.points_encoder([points, labels])
        else:
            points_inputs = self.empty_points
        sparse_embeddings = functional.concat([points_inputs, boxes_inputs], axis=1)

        """ masks """
        if masks is not None:
            masks = np.squeeze(np.array(masks, dtype="float32"))
            assert masks.ndim == 2, "masks should better provided in shape `[height, width]` format"
            if masks.shape[0] != self.masks_input_shape[0] or masks.shape[1] != self.masks_input_shape[1]:
                masks = backend.numpy_image_resize(masks, self.masks_input_shape, method="nearest")  # resize to [256, 256]
            masks = masks[None, :, :, None] if backend.image_data_format() == "channels_last" else masks[None, None]
            dense_embeddings = self.mask_encoder(masks)
        else:
            dense_embeddings = self.empty_masks

        return sparse_embeddings, dense_embeddings
