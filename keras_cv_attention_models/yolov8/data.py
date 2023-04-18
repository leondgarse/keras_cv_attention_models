"""
Raw ultralytics data loader
"""
import torch
from torch.utils.data import DataLoader
from ultralytics.yolo.cfg import get_cfg
from ultralytics.yolo.utils import DEFAULT_CFG
from ultralytics.yolo.data.utils import check_det_dataset
from ultralytics.yolo.data.dataset import YOLODataset


def to_data_loader(data, cfg, mode="train", batch_size=16):
    if mode == "train":
        augment, pad, shuffle, rect = True, 0, True, False
    else:
        augment, pad, shuffle, rect = False, 0.5, False, True
    dataset = YOLODataset(
        img_path=data["train"] if mode == "train" else data["val"],
        imgsz=640,
        batch_size=batch_size,
        augment=augment,  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=rect,  # rectangular batches
        cache=None,
        single_cls=False,
        stride=32,
        pad=pad,
        names=data["names"]
        # classes=cfg.classes,
    )

    generator = torch.Generator()
    generator.manual_seed(6148914691236517205)
    collate_fn = getattr(dataset, "collate_fn", None)
    data_loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=16, sampler=None, pin_memory=True, collate_fn=collate_fn, generator=generator
    )
    return data_loader


def get_data_loader(dataset_path="../ultralytics/ultralytics/datasets/coco.yaml", cfg={}):
    cfg = get_cfg(DEFAULT_CFG)
    cfg.data = dataset_path
    data = check_det_dataset(dataset_path)
    train_loader, val_loader = to_data_loader(data, cfg), to_data_loader(data, cfg, "val")
    return train_loader, val_loader


if __name__ == "__main__":
    os.environ["KECAM_BACKEND"] = "torch"
    sys.path.append("../ultralytics/")
    from keras_cv_attention_models.yolov8.data import get_data_loader

    train_loader, val_loader = get_data_loader()
    for aa in train_loader:
        break
    plt.imshow(aa["img"][0].permute(1, 2, 0).numpy())