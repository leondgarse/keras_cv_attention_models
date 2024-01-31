"""
Raw ultralytics data loader
"""
import torch
from torch.utils.data import DataLoader
from ultralytics.cfg import get_cfg
from ultralytics.utils import DEFAULT_CFG
from ultralytics.data.utils import check_det_dataset
from ultralytics.data.dataset import YOLODataset


def to_data_loader(data, cfg, imgsz=640, mode="train", batch_size=16, rect_val=False):
    if mode == "train":
        augment, pad, shuffle, rect, batch_size = True, 0, True, False, batch_size
    else:
        augment, pad, shuffle, rect, batch_size = False, 0.5, False, rect_val, batch_size * 2
    dataset = YOLODataset(
        img_path=data["train"] if mode == "train" else data["val"],
        imgsz=imgsz,
        batch_size=batch_size,
        augment=augment,  # augmentation
        hyp=cfg,  # TODO: probably add a get_hyps_from_cfg function
        rect=rect,  # rectangular batches
        cache=None,
        single_cls=False,
        stride=32,
        pad=pad,
        data=data,
        # names=data["names"],
        # classes=cfg.classes,
    )

    generator = torch.Generator()
    generator.manual_seed(6148914691236517205)
    collate_fn = getattr(dataset, "collate_fn", None)
    data_loader = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=shuffle, num_workers=8, sampler=None, pin_memory=True, collate_fn=collate_fn, generator=generator
    )
    return data_loader


def get_data_loader(dataset_path="coco.yaml", cfg={}, imgsz=640, batch_size=16, rect_val=False):
    cfg = get_cfg(DEFAULT_CFG)
    cfg.data = dataset_path
    cfg.imgsz = imgsz
    data = check_det_dataset(dataset_path)
    train_loader = to_data_loader(data, cfg, imgsz=imgsz, mode="train", batch_size=batch_size)
    val_loader = to_data_loader(data, cfg, imgsz=imgsz, mode="val", batch_size=batch_size, rect_val=rect_val)
    return train_loader, val_loader


if __name__ == "__main__":
    os.environ["KECAM_BACKEND"] = "torch"
    sys.path.append("../ultralytics/")
    from keras_cv_attention_models.yolov8.data import get_data_loader

    train_loader, val_loader = get_data_loader("datasets/coco_dog_cat/ultralytics.yaml")
    aa = next(iter(train_loader))
    plt.imshow(aa["img"][0].permute(1, 2, 0).numpy())
