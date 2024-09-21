import numpy as np


class ModelEMA:
    """Updated Exponential Moving Average (EMA) from https://github.com/rwightman/pytorch-image-models
    Keeps a moving average of everything in the model state_dict (parameters and buffers)
    For EMA details see https://www.tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    To disable EMA set the `enabled` attribute to `False`.
    """

    def __init__(self, model, decay=0.9999, tau=2000, updates=0):
        from copy import deepcopy

        self.decay, self.tau, self.updates = decay, tau, updates
        self.ema = deepcopy(model).eval()  # FP32 EMA
        for p in self.ema.parameters():
            p.requires_grad_(False)
        self.enabled = True

    def update(self, model):
        # Update EMA parameters
        if not self.enabled:
            return
        self.updates += 1
        cur_decay = self.decay * (1 - np.exp(-self.updates / self.tau))  # decay exponential ramp (to help early epochs)
        model_state_dict = model.state_dict()  # model state_dict
        for name, param in self.ema.state_dict().items():
            if param.dtype.is_floating_point:  # true for FP16 and FP32
                param *= cur_decay  # Ensential in this way, in place operation
                param += (1 - cur_decay) * model_state_dict[name].detach()  # Update EMA parameters


def build_optimizer(model, name="sgd", lr=0.01, momentum=0.937, decay=5e-4):
    import torch

    g = [], [], []  # optimizer parameter groups
    bn = tuple(v for k, v in torch.nn.__dict__.items() if "Norm" in k)  # normalization layers, i.e. BatchNorm2d()
    for v in model.modules():
        if hasattr(v, "bias") and isinstance(v.bias, torch.nn.Parameter):  # bias (no decay)
            g[2].append(v.bias)
        if isinstance(v, bn):  # weight (no decay)
            g[1].append(v.weight)
        elif hasattr(v, "weight") and isinstance(v.weight, torch.nn.Parameter):  # weight (with decay)
            g[0].append(v.weight)

    name_lower = name.lower()
    if name_lower == "sgd":
        optimizer = torch.optim.SGD(g[2], lr=lr, momentum=momentum, nesterov=True)
    elif name_lower == "adamw":
        optimizer = torch.optim.AdamW(g[2], lr=lr, betas=(momentum, 0.999), weight_decay=0.0)
    optimizer.add_param_group({"params": g[0], "weight_decay": decay})  # add g0 with weight_decay
    optimizer.add_param_group({"params": g[1], "weight_decay": 0.0})  # add g1 (BatchNorm2d weights)
    return optimizer


def train(model, dataset_path="coco.json", batch_size=16, epochs=100, initial_epoch=0, optimizer_name="sgd"):
    import torch
    from tqdm import tqdm
    from keras_cv_attention_models.coco import torch_losses, torch_data, eval_func

    if torch.cuda.is_available():
        model = model.cuda()
        use_amp = True
    else:
        model = model.cpu()
        use_amp = False

    warmup_epochs = 3
    close_mosaic = 10

    input_shape = model.input_shape[2:] if hasattr(model, "input_shape") and model.input_shape[2] is not None else (640, 640)
    print(">>>> input_shape:", input_shape)

    train_loader, _ = torch_data.init_dataset(data_path=dataset_path, batch_size=batch_size, image_size=input_shape)
    device = next(model.parameters()).device  # get model device
    num_classes = getattr(model, "num_classes", model.output_shape[-1] - 64)
    print(">>>> num_classes =", num_classes)

    compute_loss = torch_losses.Loss(device=device, nc=num_classes, input_shape=input_shape)
    optimizer = build_optimizer(model, name=optimizer_name)
    ema = ModelEMA(model)
    # lf = lambda x: (x * (1 - 0.01) / warmup_epochs + 0.01) if x < warmup_epochs else ((1 - x / epochs) * (1.0 - 0.01) + 0.01)  # linear
    lf = lambda x: (1 - x / epochs) * (1.0 - 0.01) + 0.01  # linear
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

    validator = eval_func.COCOEvalCallback(
        data_name=dataset_path, batch_size=batch_size, rescale_mode="raw01", nms_method="hard", nms_iou_or_sigma=0.65, nms_max_output_size=300
    )
    validator.model = ema.ema

    nb = len(train_loader)
    nw = max(round(warmup_epochs * nb), 100)
    nbs = 64
    accumulate = max(round(nbs / batch_size), 1)
    warmup_bias_lr = 0.1
    momentum = 0.937
    warmup_momentum = 0.8
    last_opt_step = -1
    for epoch in range(initial_epoch, epochs):
        model.train()
        if epoch == (epochs - close_mosaic):
            print("Closing dataloader mosaic")
            if hasattr(train_loader.dataset, "mosaic"):
                train_loader.dataset.mosaic = 0

        box_loss, cls_loss, dfl_loss = 0, 0, 0
        optimizer.zero_grad()
        loss_names = ["box_loss", "cls_loss", "dfl_loss"]
        print(("\n" + "%11s" * (3 + len(loss_names))) % ("Epoch", *loss_names, "Instances", "Size"))
        pbar = tqdm(enumerate(train_loader), total=nb, bar_format="{l_bar}{bar:10}{r_bar}")
        for batch, (images, labels) in pbar:
            ni = batch + nb * epoch
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
                preds = model(images.to(device, non_blocking=True).float())
                loss = compute_loss(labels, preds)

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

            box_loss = (box_loss * batch + compute_loss.box_loss.detach().item()) / (batch + 1)
            cls_loss = (cls_loss * batch + compute_loss.cls_loss.detach().item()) / (batch + 1)
            dfl_loss = (dfl_loss * batch + compute_loss.dfl_loss.detach().item()) / (batch + 1)
            losses = [box_loss, cls_loss, dfl_loss]
            pbar.set_description(("%11s" * 1 + "%11.4g" * (2 + len(losses))) % (f"{epoch + 1}/{epochs}", *losses, labels.shape[0], images.shape[-1]))
        scheduler.step()
        model.eval()
        validator.on_epoch_end(epoch=epoch)
        model.save(model.name + ".pt")
        ema.ema.save(model.name + "_ema.h5")
    return ema


if __name__ == "__main__":
    import os, sys, torch

    os.environ["KECAM_BACKEND"] = "torch"

    from keras_cv_attention_models.yolov8 import train, yolov8
    from keras_cv_attention_models import efficientnet

    global_device = torch.device("cuda:0") if torch.cuda.is_available() and int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")) >= 0 else torch.device("cpu")
    # model Trainable params: 7,023,904, GFLOPs: 8.1815G
    bb = efficientnet.EfficientNetV2B0(input_shape=(3, 640, 640), num_classes=0)
    model = yolov8.YOLOV8_N(backbone=bb, classifier_activation=None, pretrained=None).to(global_device)  # Note: classifier_activation=None
    # model = yolov8.YOLOV8_N(input_shape=(3, None, None), classifier_activation=None, pretrained=None).to(global_device)
    ema = train.train(model, dataset_path="datasets/coco_dog_cat/detections.json", initial_epoch=0)
