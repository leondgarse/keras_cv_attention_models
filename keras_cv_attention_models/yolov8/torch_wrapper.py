"""
Wrappering a kecam PyTorch YOLOV8 model for training using ultralytics package
"""

import math
import torch
from torch import nn


def dist2bbox(distance, anchor_points, xywh=True, dim=-1):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = distance.chunk(2, dim)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if xywh:
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        return torch.cat((c_xy, wh), dim)  # xywh bbox
    return torch.cat((x1y1, x2y2), dim)  # xyxy bbox


class Detect(nn.Module):
    def __init__(self, model, train_input_shape=640, reg_max=16, num_pyramid_levels=3, export=False):  # detection layer
        super().__init__()
        self.model, self.reg_max, self.num_pyramid_levels, self.export = model, reg_max, num_pyramid_levels, export
        self.device = next(self.model.parameters()).device

        train_input_shape = model.input_shape[2:] if hasattr(model, "input_shape") and model.input_shape[2] is not None else train_input_shape
        train_input_shape = train_input_shape[-2:] if isinstance(train_input_shape, (list, tuple)) else (train_input_shape, train_input_shape)
        self.input_shape, self.input_height, self.input_width = [None, 3, *train_input_shape], train_input_shape[0], train_input_shape[1]
        self.feature_sizes, self.feature_lens = self.get_feature_sizes(train_input_shape)

        self.output_shape = model.output_shape
        self.num_classes = self.output_shape[-1] - self.reg_max * 4
        self.names = {ii: str(ii) for ii in range(self.num_classes)}

        # self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        self.dfl = torch.arange(self.reg_max, device=self.device, dtype=torch.float32).view(1, 1, self.reg_max, 1)
        self.pre_val_input_shape, self.pre_val_feature_sizes, self.pre_val_feature_lens = None, None, None

    def get_feature_sizes(self, input_shape):
        feature_sizes = [(math.ceil(input_shape[-2] / (2**ii)), math.ceil(input_shape[-1] / (2**ii))) for ii in range(3, 3 + self.num_pyramid_levels)]
        feature_lens = [ii[0] * ii[1] for ii in feature_sizes]
        return feature_sizes, feature_lens

    def make_anchors(self, feature_sizes, grid_cell_offset=0.5):
        """Generate anchors from features."""
        anchor_points, stride_tensor = [], []
        strides = [2**ii for ii in range(3, 3 + self.num_pyramid_levels)]
        for (hh, ww), stride in zip(feature_sizes, strides):
            sx = torch.arange(end=ww, device=self.device, dtype=torch.float32) + grid_cell_offset  # shift x
            sy = torch.arange(end=hh, device=self.device, dtype=torch.float32) + grid_cell_offset  # shift y
            sy, sx = torch.meshgrid(sy, sx, indexing="ij")
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((hh * ww, 1), stride, dtype=torch.float32, device=self.device))
        return torch.cat(anchor_points).transpose(0, 1), torch.cat(stride_tensor).transpose(0, 1)

    def forward(self, inputs):
        out = self.model(inputs)
        out = out.permute([0, 2, 1])

        if self.training:
            feature_sizes, feature_lens = self.feature_sizes, self.feature_lens
        elif inputs.shape[1:] == self.pre_val_input_shape:
            feature_sizes, feature_lens = self.pre_val_feature_sizes, self.pre_val_feature_lens
        else:
            feature_sizes, feature_lens = self.get_feature_sizes(inputs.shape[2:])
            self.anchors, self.strides = self.make_anchors(feature_sizes)
            self.pre_val_input_shape, self.pre_val_feature_sizes, self.pre_val_feature_lens = inputs.shape[1:], feature_sizes, feature_lens

        train_out = torch.split(out, feature_lens, dim=-1)
        train_out = [ii.view([-1, ii.shape[1], hh, ww]) for ii, (hh, ww) in zip(train_out, feature_sizes)]

        if self.training:
            return train_out
        else:
            box, cls = torch.split(out, (self.reg_max * 4, self.num_classes), dim=1)
            # box = box.reshape([-1, 4, self.reg_max, box.shape[-1]])[:, [1, 0, 3, 2]].reshape([-1, 64, box.shape[-1]])
            box = (box.view(-1, 4, self.reg_max, box.shape[-1]).softmax(2) * self.dfl).sum(2).view(-1, 4, box.shape[-1])
            box = dist2bbox(box, self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
            # box = box[:, [1, 0, 3, 2]]  # [TODO] w/o ultralytics xyxy -> yxyx
            val_out = torch.cat((box, cls.sigmoid()), 1)
            return val_out if self.export else (val_out, train_out)


if __name__ == "__main__":
    os.environ["KECAM_BACKEND"] = "torch"
    sys.path.append("../ultralytics/")
    import torch
    from keras_cv_attention_models.test_images import dog_cat
    from skimage.transform import resize

    tt = torch.load("yolov8n.pt")["model"]
    _ = tt.eval()
    _ = tt.float()

    imm = resize(dog_cat(), [640, 640])
    preds_torch, torch_out = tt(torch.from_numpy(imm[None]).permute([0, 3, 1, 2]).float())

    import torch
    from keras_cv_attention_models.yolov8 import torch_wrapper, yolov8
    from keras_cv_attention_models.test_images import dog_cat
    from skimage.transform import resize

    mm = yolov8.YOLOV8_N(classifier_activation=None, input_shape=(640, 640, 3))
    tt = torch_wrapper.Detect(mm)
    _ = tt.eval()
    imm = resize(dog_cat(), [640, 640])
    preds_torch, torch_out = tt(torch.from_numpy(imm[None]).permute([0, 3, 1, 2]).float())
