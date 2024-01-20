import numpy as np
from PIL import Image
from keras_cv_attention_models import backend
from keras_cv_attention_models.backend import layers, models, functional, image_data_format, initializers
from keras_cv_attention_models.attention_layers import BiasLayer, PureWeigths, batchnorm_with_activation, conv2d_no_bias, layer_norm
from keras_cv_attention_models.models import register_model, FakeModelWrapper, no_grad_if_torch
from keras_cv_attention_models.download_and_load import reload_model_weights
from keras_cv_attention_models.segment_anything import image_encoders, mask_decoders, prompt_encoders

LAYER_NORM_EPSILON = 1e-6


@register_model
class SAM(FakeModelWrapper):  # FakeModelWrapper providing save / load / cuda class methods
    def __init__(
        self,
        image_encoder="TinyViT_5M",
        mask_decoder="sam_mask_decoder",  # string or built mask decoder model. Currently string can be one of ["sam_mask_decoder", "tiny_sam_mask_decoder"]
        image_shape=(1024, 1024),
        embed_dims=256,
        mask_hidden_dims=16,
        pretrained="sam",
        name="mobile_sam_5m",
        kwargs=None,  # Not using, just recieving parameter
    ):
        self.image_shape = image_shape[:2] if isinstance(image_shape, (list, tuple)) else [image_shape, image_shape]
        self.embed_dims = embed_dims

        self.image_encoder = image_encoders.get_image_encoder(
            image_encoder, input_shape=[*self.image_shape, 3], embed_dims=embed_dims, pretrained=pretrained, name=name + "_image_encoder"
        )
        self.image_embedding_shape = self.image_encoder.output_shape[1:-1] if image_data_format() == "channels_last" else self.image_encoder.output_shape[2:]
        self.prompt_mask_shape = [int(self.image_embedding_shape[0] * 16), int(self.image_embedding_shape[1] * 16)]  # [64, 64] -> [1024, 1024]
        self.masks_input_shape = [int(self.image_embedding_shape[0] * 4), int(self.image_embedding_shape[1] * 4)]  # [64, 64] -> [256, 256]

        if isinstance(mask_decoder, str):
            self.mask_decoder = mask_decoders.MaskDecoder(input_shape=[*self.image_embedding_shape, embed_dims], name=mask_decoder)
        else:
            self.mask_decoder = mask_decoder

        # prompt_encoder is also a subclass of FakeModelWrapper, and here not passing the `name`
        self.prompt_encoder = prompt_encoders.PromptEncoder(embed_dims, mask_hidden_dims, self.prompt_mask_shape, self.masks_input_shape, pretrained=pretrained)
        self.models = [self.image_encoder, self.mask_decoder] + self.prompt_encoder.models
        super().__init__(self.models, name=name)

        self.grid_positional_embedding = self.prompt_encoder.get_grid_positional_embedding(self.image_embedding_shape)
        self.image_mean, self.image_std = np.array([123.675, 116.28, 103.53]).astype("float32"), np.array([58.395, 57.12, 57.375]).astype("float32")

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

    @no_grad_if_torch
    def __call__(self, image, points=None, labels=None, boxes=None, masks=None, mask_threshold=0, return_logits=False):
        orign_height, orign_width = image.shape[:2]
        scale = min(self.prompt_mask_shape[0] / orign_height, self.prompt_mask_shape[1] / orign_width)
        scaled_height, scaled_width = int(orign_height * scale + 0.5), int(orign_width * scale + 0.5)
        height_scale, width_scale = scaled_height / orign_height, scaled_width / orign_width
        # print(f"{scaled_height = }, {scaled_width = }, {height_scale = }, {width_scale = }")

        """ image_encoder """
        image_embeddings = self.image_encoder(self.preprocess_image(image))

        """ prompt_encoder """
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            image_orign_shape=(orign_height, orign_width), points=points, labels=labels, boxes=boxes, masks=masks
        )

        """ mask_decoder """
        image_with_masks = image_embeddings + dense_embeddings
        image_with_masks = image_with_masks if image_data_format() == "channels_last" else functional.transpose(image_with_masks, [0, 2, 3, 1])
        # print(f"{image_with_masks.shape = }, {sparse_embeddings.shape = }, {self.grid_positional_embedding.shape = }")
        low_res_masks, iou_predictions = self.mask_decoder([image_with_masks, sparse_embeddings, self.grid_positional_embedding])

        """ Remove padding and resize masks to the original image size """
        # [batch, 4, height, width] -> [batch, height, width, 4] if channels_last
        masks = functional.transpose(low_res_masks, [0, 2, 3, 1]) if image_data_format() == "channels_last" else low_res_masks
        masks = functional.resize(masks, self.prompt_mask_shape, method="bilinear")
        masks = masks[:, :scaled_height, :scaled_width] if image_data_format() == "channels_last" else masks[:, :, :scaled_height, :scaled_width]
        masks = functional.resize(masks, [orign_height, orign_width], method="bilinear")
        masks = functional.transpose(masks, [0, 3, 1, 2]) if image_data_format() == "channels_last" else masks  # return [batch, 4, height, width]
        masks = masks if return_logits else (masks > mask_threshold)

        """ Return numpy. For a single image, batch_size is always 1. Converting masks `[1, 4, height, width]` -> `[4, height, width]` """
        if backend.is_torch_backend:
            return masks[0].cpu().numpy(), iou_predictions[0].cpu().numpy(), low_res_masks[0].cpu().numpy()
        else:
            return np.array(masks[0]), np.array(iou_predictions[0]), np.array(low_res_masks[0])

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
def TinySAM(image_shape=(1024, 1024), mask_decoder="tiny_sam_mask_decoder", pretrained="sam", name="tiny_sam_5m", **kwargs):
    return SAM(image_encoder="TinyViT_5M", **locals(), **kwargs)


@register_model
def EfficientViT_SAM_L0(image_shape=(512, 512), pretrained="sam", name="efficientvit_sam_l0", **kwargs):
    mask_decoders.LAYER_NORM_EPSILON = 1e-6
    model = SAM(image_encoder="EfficientViT_L0", **locals(), **kwargs)
    mask_decoders.LAYER_NORM_EPSILON = 1e-5
    return model
