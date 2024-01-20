import copy
import math
import torch
import numpy as np
from tqdm import tqdm
from pathlib import Path
from torch import nn
from torch.cuda import amp
from torch.optim import lr_scheduler


class FakeArgs:
    def __init__(self, **kwargs):
        self.update(**kwargs)

    def update(self, **kwargs):
        for kk, vv in kwargs.items():
            setattr(self, kk, vv)


class ModelEMA:
    """Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    To disable EMA set the `enabled` attribute to `False`.
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        # Create EMA
        self.ema = copy.deepcopy(model).eval()  # FP32 EMA
        self.updates = updates  # number of EMA updates
        self.decay = lambda x: decay * (1 - math.exp(-x / tau))  # decay exponential ramp (to help early epochs)
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.enabled = True

    def update(self, model):
        # Update EMA parameters
        if self.enabled:
            self.updates += 1
            d = self.decay(self.updates)

            msd = model.state_dict()  # model state_dict
            for k, v in self.ema.state_dict().items():
                if v.dtype.is_floating_point:  # true for FP16 and FP32
                    v *= d
                    v += (1 - d) * msd[k].detach()
                    # assert v.dtype == msd[k].dtype == torch.float32, f'{k}: EMA {v.dtype},  model {msd[k].dtype}'


def build_optimizer(model, name="sgd", lr=0.01, momentum=0.937, decay=5e-4):
    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        if hasattr(v, "bias") and isinstance(v.bias, nn.Parameter):  # bias (no decay)
            g[2].append(v.bias)
        if isinstance(v, bn):  # weight (no decay)
            g[1].append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, nn.Parameter):  # weight (with decay)
            g[0].append(v.weight)

    name_lower = name.lower()
    if name_lower == "sgd":
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    elif name_lower == "adamw":
        optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
    return optimizer


def train(model, dataset_path="coco.yaml", batch_size=16, epochs=100, initial_epoch=0, optimizer_name="sgd", rect_val=False, overwrite_cfg=None):
    from keras_cv_attention_models.yolov8 import losses
    from keras_cv_attention_models.yolov8 import data, eval
    # from keras_cv_attention_models.coco import torch_data, eval_func  # [TODO] w/o ultralytics

    if torch.cuda.is_available():
        model = model.cuda()
        use_amp = True
    else:
        model = model.cpu()
        use_amp = False

    warmup_epochs = 3
    close_mosaic = 10

    imgsz = (model.input_shape[2] or 640) if hasattr(model, "input_shape") else 640
    print(">>>> imgsz:", imgsz)
    cfg = FakeArgs(data=dataset_path, imgsz=imgsz, iou=0.7, single_cls=False, max_det=300, task="detect", mode="train", split="val", half=False, conf=None)
    cfg.update(augment=False, degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, flipud=0.0, fliplr=0.5)
    cfg.update(mask_ratio=4, overlap_mask=True, project=None, name=None, save_txt=False, save_hybrid=False, save_json=False, plots=False, verbose=True, seed=0)
    if overwrite_cfg is not None:
        cfg.update(**overwrite_cfg)
    # from ultralytics.yolo.cfg import get_cfg
    # from ultralytics.yolo.utils import DEFAULT_CFG
    # cfg = get_cfg(DEFAULT_CFG)
    # cfg.data = dataset_path

    train_loader, val_loader = data.get_data_loader(dataset_path=dataset_path, batch_size=batch_size, imgsz=imgsz, rect_val=rect_val)
    # train_loader, _ = torch_data.init_dataset(data_path=dataset_path, batch_size=batch_size, image_size=imgsz)  # [TODO] w/o ultralytics
    device = next(model.parameters()).device  # get model device
    num_classes = getattr(model, "num_classes", model.output_shape[-1] - 64)
    print(">>>> num_classes =", num_classes)
    compute_loss = losses.Loss(device=device, nc=num_classes)
    optimizer = build_optimizer(model, name=optimizer_name)
    ema = ModelEMA(model)
    # lf = lambda x: (x * (1 - 0.01) / warmup_epochs + 0.01) if x < warmup_epochs else ((1 - x / epochs) * (1.0 - 0.01) + 0.01)  # linear
    lf = lambda x: (1 - x / epochs) * (1.0 - 0.01) + 0.01  # linear
    scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scaler = amp.GradScaler(enabled=use_amp)
    validator = eval.Validator(val_loader=val_loader, model=ema.ema, cfg=cfg)
    # validator = eval_func.COCOEvalCallback(data_name=dataset_path, batch_size=batch_size, nms_mode="global")  # [TODO] w/o ultralytics
    # validator.model = model  # [TODO] w/o ultralytics

    nb = len(train_loader)
    nw = max(round(warmup_epochs * nb), 100)
    nbs = 64
    accumulate = max(round(nbs / batch_size), 1)
    warmup_bias_lr = 0.1
    momentum = 0.937
    warmup_momentum = 0.8
    last_opt_step = -1
    for epoch in range(initial_epoch, epochs):
        # self.run_callbacks('on_train_epoch_start')
        model.train()
        # Update attributes (optional)
        if epoch == (epochs - close_mosaic):
            print("Closing dataloader mosaic")
            if hasattr(train_loader.dataset, "mosaic"):
                train_loader.dataset.mosaic = False
            if hasattr(train_loader.dataset, "close_mosaic"):
                train_loader.dataset.close_mosaic(hyp=cfg)

        tloss = None
        optimizer.zero_grad()
        loss_names = ["box_loss", "cls_loss", "dfl_loss"]
        print(("\n" + "%11s" * (3 + len(loss_names))) % ("Epoch", *loss_names, "Instances", "Size"))
        pbar = tqdm(enumerate(train_loader), total=nb, bar_format="{l_bar}{bar:10}{r_bar}")
        # for i, (images, labels) in pbar:  # [TODO] w/o ultralytics
        for i, batch in pbar:
            # self.run_callbacks('on_train_batch_start')
            ni = i + nb * epoch
            if ni <= nw:
                xi = [0, nw]  # x interp
                accumulate = max(1, np.interp(ni, xi, [1, nbs / batch_size]).round())
                for j, x in enumerate(optimizer.param_groups):
                    # bias lr falls from 0.1 to lr0, all other lrs rise from 0.0 to lr0
                    x["lr"] = np.interp(ni, xi, [warmup_bias_lr if j == 0 else 0.0, x["initial_lr"] * lf(epoch)])
                    if "momentum" in x:
                        x["momentum"] = np.interp(ni, xi, [warmup_momentum, momentum])

            # Forward
            with torch.cuda.amp.autocast(use_amp):
                preds = model(batch["img"].to(device, non_blocking=True).float() / 255)
                loss, loss_items = compute_loss(preds, batch)
                # preds = model(images.to(device, non_blocking=True).float())  # [TODO] w/o ultralytics
                # loss, loss_items = compute_loss(preds, labels)  # [TODO] w/o ultralytics
                tloss = (tloss * i + loss_items) / (i + 1) if tloss is not None else loss_items

            # Backward
            scaler.scale(loss).backward()

            # Optimize - https://pytorch.org/docs/master/notes/amp_examples.html
            if ni - last_opt_step >= accumulate:
                # optimizer_step(model, optimizer, scaler)
                scaler.unscale_(optimizer)  # unscale gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)  # clip gradients
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                ema.update(model)
                last_opt_step = ni

            loss_len = tloss.shape[0] if len(tloss.size()) else 1
            losses = tloss if loss_len > 1 else torch.unsqueeze(tloss, 0)
            pbar.set_description(("%11s" * 1 + "%11.4g" * (2 + loss_len)) % (f"{epoch + 1}/{epochs}", *losses, batch["cls"].shape[0], batch["img"].shape[-1]))
            # pbar.set_description(("%11s" * 1 + "%11.4g" * (2 + loss_len)) % (f"{epoch + 1}/{epochs}", *losses, labels.shape[0], images.shape[-1]))  # [TODO] w/o ultralytics
        scheduler.step()
        model.eval()
        validator()
        # validator.on_epoch_end(epoch=epoch)  # [TODO] w/o ultralytics

        if hasattr(model, "model") and hasattr(model.model, "save_weights"):
            model.model.save_weights(model.model.name + ".h5")
        elif hasattr(model, "model") and hasattr(model.model, "state_dict"):
            torch.save(model.model.state_dict(), model.__class__.__name__ + ".pth")

    # Save ema model
    ema_model = ema.ema.model
    if hasattr(ema_model, "model") and hasattr(ema_model.model, "save_weights"):
        ema_model.model.save_weights(ema_model.model.name + "_ema.h5")
    elif hasattr(ema_model, "model") and hasattr(ema_model.model, "state_dict"):
        torch.save(ema_model.model.state_dict(), ema_model.__class__.__name__ + "_ema.pth")
    return ema


if __name__ == "__main__":
    os.environ["KECAM_BACKEND"] = "torch"
    sys.path.append("../ultralytics/")
    from keras_cv_attention_models.yolov8 import train, yolov8, torch_wrapper

    # from ultralytics import YOLO
    # model = YOLO('../ultralytics/ultralytics/models/v8/yolov8n.yaml').model
    model = yolov8.YOLOV8_N(input_shape=(3, None, None), classifier_activation=None, pretrained=None)
    model = torch_wrapper.Detect(model)
    train.train(model, dataset_path="coco128.yaml")
