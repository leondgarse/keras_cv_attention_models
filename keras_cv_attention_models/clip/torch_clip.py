import os
import math
import torch
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from contextlib import nullcontext

from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor

os.environ["KECAM_BACKEND"] = "torch"
import kecam

# Always 0, no matter CUDA_VISIBLE_DEVICES
global_device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
device_type = global_device.type
if device_type == "cpu":
    global_scaler = torch.cuda.amp.GradScaler(enabled=False)
    global_context = nullcontext()
else:
    global_scaler = torch.cuda.amp.GradScaler(enabled=True)
    global_context = torch.amp.autocast(device_type=device_type, dtype=torch.float16)


class CsvDataset(Dataset):
    def __init__(self, input_filename, tokenizer, image_size=224, sep="\t"):
        df = pd.read_csv(input_filename, header=None, sep=sep, names=["image", "caption"])

        self.images, self.captions, self.base_path = [], [], "."
        for image, caption in zip(df["image"], df["caption"]):
            if image == "TEST":
                break
            if image == "base_path":
                self.base_path = caption
            else:
                self.images.append(image)
                self.captions.append(caption)
        self.images = [os.path.join(self.base_path, ii) for ii in self.images]

        self.mean, self.std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        self.transforms = Compose(
            [
                RandomResizedCrop(image_size, scale=(0.9, 1.0), interpolation=InterpolationMode.BICUBIC),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(mean=self.mean, std=self.std),
            ]
        )
        self.tokenizer = tokenizer
        self.context_length = self.tokenizer.context_length

    def tokenize(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        all_tokens = [[self.tokenizer.sot_token] + self.tokenizer.encode(text) + [self.tokenizer.eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), self.context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > self.context_length:
                tokens = tokens[:self.context_length]  # Truncate
                tokens[-1] = eot_token
            result[i, : len(tokens)] = torch.tensor(tokens)
        return result

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts


def build_dataset(data_path, caption_tokenizer, batch_size=64, image_size=224, num_workers=8):
    dataset = CsvDataset(data_path, image_size=image_size, tokenizer=caption_tokenizer)
    num_samples = len(dataset)

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, sampler=None, drop_last=True)
    dataloader.num_samples = num_samples
    dataloader.num_batches = len(dataloader)
    return dataloader


class CLIP(nn.Module):
    def __init__(self, image_model, text_model):
        super().__init__()
        self.image_model, self.text_model = image_model, text_model
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.image_model.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.image_model.set_grad_checkpointing(enable)
        # self.transformer.grad_checkpointing = enable

    def forward(self, image, text):
        image_features = F.normalize(self.image_model(image), dim=-1)
        text_features = F.normalize(self.text_model(text), dim=-1)
        return image_features, text_features, self.logit_scale.exp()


def build_model(image_model="FlexiViTBase", text_model="GPT2_Base", latents_dim=512, image_input_shape=(224, 224, 3), image_pretrained=None):
    if isinstance(image_model, str):
        model_split = image_model.split(".")
        image_model_class = getattr(getattr(kecam, model_split[0]), model_split[1]) if len(model_split) == 2 else getattr(kecam.models, model_split[0])
        kwargs = {} if image_pretrained == "default" else {"pretrained": image_pretrained}
        image_model = image_model_class(input_shape=image_input_shape, num_classes=latents_dim, classifier_activation=None, **kwargs)

    if isinstance(text_model, str):
        model_split = text_model.split(".")
        text_model_class = getattr(getattr(kecam, model_split[0]), model_split[1]) if len(model_split) == 2 else getattr(kecam.models, model_split[0])
        text_model = text_model_class(include_top=False)
        text_inputs = text_model.inputs[0]
        text_outputs = text_model.outputs[0]
        text_outputs = kecam.clip.models.text_model_index_header(text_inputs, text_outputs, latents_dim)
        text_model = kecam.backend.models.Model(text_inputs, text_outputs, name=text_model.name)
    # text_model(torch.ones([1, 77], dtype=torch.long)).shape

    model = CLIP(image_model, text_model)
    # print({ii:jj.shape for ii , jj in model.named_parameters()})
    return model, image_model, text_model


def build_optimizer(model, lr=1e-3, wd=0.2, beta1=0.9, beta2=0.98, eps=1.0e-6):
    named_parameters = list(model.named_parameters())
    exclude = lambda name, param: param.ndim < 2 or any([ii in name for ii in ["bn", "ln", "bias", "logit_scale", "class_tokens"]])
    params = [
        {"params": [param for name, param in named_parameters if exclude(name, param) and param.requires_grad], "weight_decay": 0.0},
        {"params": [param for name, param in named_parameters if not exclude(name, param) and param.requires_grad], "weight_decay": wd},
    ]
    optimizer = optim.AdamW(params, lr=lr, betas=(beta1, beta2), eps=eps)
    return optimizer


def cosine_lr(optimizer, total_steps, base_lr=1e-3, warmup_steps=10000):
    def _lr_adjuster(step):
        if step < warmup_steps:
            return base_lr * (step + 1) / warmup_steps
        lr = 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps))) * base_lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    return _lr_adjuster


def clip_loss(image_features, text_features, logit_scale):
    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text = logit_scale * text_features @ image_features.T

    labels = torch.arange(logits_per_image.shape[0], device=image_features.device, dtype=torch.long)
    return (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2


def train_one_epoch(model, optimizer, dataloader, loss, scheduler, cur_epoch, grad_clip_norm=10.0):
    model.train()
    bar_format = "{n_fmt}/{total_fmt} [{bar:30}] - ETA: {elapsed}<{remaining} {rate_fmt}{postfix}{desc}"
    process_bar = tqdm(enumerate(dataloader), total=dataloader.num_batches, bar_format=bar_format, ascii=".>>=")
    for id, batch in process_bar:
        step = dataloader.num_batches * cur_epoch + id
        scheduler(step)

        images, texts = batch
        images = images.to(device=global_device, non_blocking=True)
        texts = texts.to(device=global_device, dtype=torch.long, non_blocking=True)
        optimizer.zero_grad()

        with global_context:
            image_out, text_out, logit_scale = model(images, texts)
            losses = loss(image_out, text_out, logit_scale)
        global_scaler.scale(losses).backward()

        if grad_clip_norm > 0:
            global_scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm, norm_type=2.0)
        global_scaler.step(optimizer)
        global_scaler.update()

        with torch.no_grad():
            model.logit_scale.clamp_(0, math.log(100))  # clamp to 4.6052 = ln(100)
        process_bar.desc = " - loss: {:.4f}".format(losses)
        process_bar.refresh()


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--data_path", type=str, default="datasets/coco_dog_cat/captions.tsv", help="tsv format dataset path")
    parser.add_argument("-i", "--input_shape", type=int, default=224, help="Image model input shape")
    parser.add_argument("-m", "--image_model", type=str, default="FlexiViTBase", help="Model name in format [sub_dir].[model_name] like beit.BeitBasePatch16")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("-e", "--epochs", type=int, default=30, help="Total epochs")
    parser.add_argument("--pretrained", type=str, default=None, help="If build model with pretrained weights. Set 'default' for model preset value")
    parser.add_argument("--text_model", type=str, default="GPT2_Base", help="model from this repo `gpt2.[model_name]` like gpt2.GPT2_Base")
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="GPT2Tokenizer",
        help="One of ['GPT2Tokenizer', 'SimpleTokenizer'], or tiktoken one ['gpt2', 'r50k_base', 'p50k_base', 'cl100k_base']"
    )
    return parser.parse_known_args()[0]


if __name__ == "__main__":
    args = parse_arguments()

    caption_tokenizer = getattr(kecam.clip, args.tokenizer)() if hasattr(kecam.clip, args.tokenizer) else kecam.clip.TikToken(args.tokenizer)
    train_dataloader = build_dataset(args.data_path, caption_tokenizer=caption_tokenizer, image_size=args.input_shape, batch_size=args.batch_size)
    print(">>>> Data:", [ii.shape for ii in next(iter(train_dataloader))])

    image_input_shape = (3, args.input_shape, args.input_shape)
    model, image_model, text_model = build_model(args.image_model, args.text_model, image_input_shape=image_input_shape, image_pretrained=args.pretrained)
    print(">>>> image_model name: {}, input_shape: {}, output_shape: {}".format(image_model.name, image_model.input_shape, image_model.output_shape))
    print(">>>> text_model name: {}, input_shape: {}, output_shape: {}".format(text_model.name, text_model.input_shape, text_model.output_shape))

    model.to(device=global_device)
    if hasattr(torch, "compile") and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 6:
        model = torch.compile(model)
    # Always 0, no matter CUDA_VISIBLE_DEVICES
    optimizer = build_optimizer(model)

    total_steps = train_dataloader.num_batches * args.epochs
    warmup_steps = train_dataloader.num_batches * 3
    scheduler = cosine_lr(optimizer, total_steps=total_steps, warmup_steps=warmup_steps)

    start_epoch = 0
    for epoch in range(start_epoch, args.epochs):
        train_one_epoch(model, optimizer, train_dataloader, clip_loss, scheduler, epoch)
        print()
        model.image_model.save_weights("torch_clip_test_image_model.h5")
        model.text_model.save_weights("torch_clip_test_text_model.h5")
