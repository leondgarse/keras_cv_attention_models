"""
Copied from ultralytics/yolo/engine/validator.py. Original one will fuse bn if use directly on model
"""
import copy
import torch
from tqdm import tqdm
from ultralytics.models import yolo
from ultralytics.utils.ops import Profile
from ultralytics.data.utils import check_det_dataset
from pathlib import Path


class Validator:
    def __init__(self, model, val_loader, save_dir="./test", cfg={}):
        cfg = copy.copy(cfg)
        cfg.rect, cfg.mode = True, "val"
        self.device = next(model.parameters()).device
        cfg.half = self.device.type == "cuda"

        validator = yolo.detect.DetectionValidator(val_loader, save_dir=Path(save_dir), args=cfg.__dict__)
        # def eval(validator, model):
        validator.data = check_det_dataset(validator.args.data)
        validator.device = self.device
        self.n_batches = len(validator.dataloader)
        self.validator, self.model = validator, model

    def __call__(self):
        is_pre_training = self.model.training
        if self.device.type == "cuda":
            self.model.half()
        self.model.eval()
        dt = Profile(), Profile(), Profile()
        desc = self.validator.get_desc()

        bar = tqdm(self.validator.dataloader, desc, self.n_batches, bar_format="{l_bar}{bar:10}{r_bar}")
        self.validator.init_metrics(self.model)
        self.validator.jdict = []  # empty before each val
        with torch.no_grad():
            for batch_i, batch in enumerate(bar):
                self.validator.run_callbacks("on_val_batch_start")
                self.validator.batch_i = batch_i
                # preprocess
                with dt[0]:
                    batch = self.validator.preprocess(batch)

                # inference
                with dt[1]:
                    preds = self.model(batch["img"])

                # postprocess
                with dt[2]:
                    preds = self.validator.postprocess(preds)

                self.validator.update_metrics(preds, batch)
                if self.validator.args.plots and batch_i < 3:
                    self.validator.plot_val_samples(batch, batch_i)
                    self.validator.plot_predictions(batch, preds, batch_i)

            # self.validator.run_callbacks('on_val_batch_end')
        stats = self.validator.get_stats()
        self.validator.check_stats(stats)
        self.validator.print_results()
        # validator.speed = dict(zip(validator.speed.keys(), (x.t / len(validator.dataloader.dataset) * 1E3 for x in dt)))
        self.validator.finalize_metrics()
        # self.validator.run_callbacks('on_val_end')
        stats = self.validator.eval_json(stats)  # update stats
        if is_pre_training:
            self.model.train()
        if self.device.type == "cuda":
            self.model.float()
        return stats


if __name__ == "__main__":
    os.environ["KECAM_BACKEND"] = "torch"
    sys.path.append("../ultralytics/")
    from keras_cv_attention_models.yolov8 import yolov8, torch_wrapper, train, eval, data

    # from ultralytics import YOLO

    dataset_path = "coco128.yaml"
    train_loader, val_loader = data.get_data_loader(dataset_path=dataset_path, rect_val=True)
    cfg = train.FakeArgs(data=dataset_path, imgsz=640, iou=0.7, conf=0.001, single_cls=False, max_det=300, task="detect", mode="train", split="val", half=False)
    cfg.update(degrees=0.0, translate=0.1, scale=0.5, shear=0.0, perspective=0.0, hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, flipud=0.0, fliplr=0.5)
    cfg.update(mask_ratio=4, overlap_mask=True, project=None, name=None, save_txt=False, save_hybrid=False, save_json=False, plots=False, verbose=True)

    # model = YOLO('../ultralytics/ultralytics/models/v8/yolov8n.yaml').model
    model = yolov8.YOLOV8_N(input_shape=(3, None, None), classifier_activation=None, pretrained=None)
    model = torch_wrapper.Detect(model)
    ee = eval.Validator(model, val_loader, cfg=cfg)
    ee()
