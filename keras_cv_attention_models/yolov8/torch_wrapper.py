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


class DFL(nn.Module):
    # Integral module of Distribution Focal Loss (DFL)
    # Proposed in Generalized Focal Loss https://ieeexplore.ieee.org/document/9792391
    def __init__(self, c1=16):
        super().__init__()
        self.conv = nn.Conv2d(c1, 1, 1, bias=False).requires_grad_(False)
        x = torch.arange(c1, dtype=torch.float)
        self.conv.weight.data[:] = nn.Parameter(x.view(1, c1, 1, 1))
        self.c1 = c1

    def forward(self, x):
        b, c, a = x.shape  # batch, channels, anchors
        return self.conv(x.view(b, 4, self.c1, a).transpose(2, 1).softmax(1)).view(b, 4, a)
        # return self.conv(x.view(b, self.c1, 4, a).softmax(1)).view(b, 4, a)


class Detect(nn.Module):
    def __init__(self, model, reg_max=16, num_pyramid_levels=3, export=False):  # detection layer
        super().__init__()
        self.model, self.reg_max, self.num_pyramid_levels, self.export = model, reg_max, num_pyramid_levels, export
        self.build(model.output_shape)

    def build(self, input_shape):
        self.num_classes = input_shape[-1] - self.reg_max * 4
        pyramid_len = input_shape[1] // sum([4 ** ii for ii in range(self.num_pyramid_levels)])

        self.pyramid_lens = [pyramid_len * (4 ** ii) for ii in range(self.num_pyramid_levels)][::-1]
        self.pyramid_lens_sqrt = [int(math.sqrt(ii)) for ii in self.pyramid_lens]
        self.dfl = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()

        self.device = next(self.model.parameters()).device
        self.anchors, self.strides = self.make_anchors()
        # self.anchors, self.strides = anchors.transpose(0, 1), strides.transpose(0, 1)

    def make_anchors(self, grid_cell_offset=0.5):
        """Generate anchors from features."""
        anchor_points, stride_tensor = [], []
        strides = [2 ** ii for ii in range(3, 3 + self.num_pyramid_levels)]
        for block, stride in zip(self.pyramid_lens_sqrt, strides):
            sx = torch.arange(end=block, device=self.device, dtype=torch.float32) + grid_cell_offset  # shift x
            sy = torch.arange(end=block, device=self.device, dtype=torch.float32) + grid_cell_offset  # shift y
            sy, sx = torch.meshgrid(sy, sx, indexing='ij')
            anchor_points.append(torch.stack((sx, sy), -1).view(-1, 2))
            stride_tensor.append(torch.full((block * block, 1), stride, dtype=torch.float32, device=self.device))
        return torch.cat(anchor_points).transpose(0, 1), torch.cat(stride_tensor).transpose(0, 1)

    def forward(self, inputs):
        out = self.model(inputs)
        out = out.permute([0, 2, 1])

        train_out = torch.split(out, self.pyramid_lens, dim=-1)
        train_out = [ii.view([-1, ii.shape[1], shape, shape]) for ii, shape in zip(train_out, self.pyramid_lens_sqrt)]

        if self.training:
            return train_out
        else:
            box, cls = torch.split(out, (self.reg_max * 4, self.num_classes), dim=1)
            box = box.reshape([-1, 4, 16, box.shape[-1]])[:, [1, 0, 3, 2]].reshape([-1, 64, box.shape[-1]])
            dbox = dist2bbox(self.dfl(box), self.anchors.unsqueeze(0), xywh=True, dim=1) * self.strides
            val_out = torch.cat((dbox, cls.sigmoid()), 1)
            return val_out if self.export else (val_out, train_out)

if __name__ == "__main__":
    from ultralytics import YOLO
    from keras_cv_attention_models.yolov8 import torch_wrapper
    from keras_cv_attention_models import efficientnet, yolov8

    backbone = efficientnet.EfficientNetV2B1(input_shape=(3, 640, 640))
    mm = yolov8.YOLOV8_N(backbone=backbone, pretrained=None, classifier_activation=None)
    _ = mm.train()

    yolo = YOLO('yolov8n.yaml')
    tt = torch_wrapper.Detect(mm)
    tt.yaml = yolo.model.yaml
    yolo.model = tt
    yolo.train(data='coco.yaml', epochs=100)
