import os
import math
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader

os.environ["KECAM_BACKEND"] = "torch"
import kecam


def read_from_tsv(data_path):
    import csv

    delimiter = "\t" if data_path.endswith(".tsv") else ","
    train_images, train_captions, test_images, test_captions, base_path, is_train = [], [], [], [], ".", True
    with open(data_path) as ff:
        for ii in csv.reader(ff, delimiter=delimiter):
            if ii[0] == "base_path":  # special keys for info
                base_path = ii[1]
            elif ii[0] == "TEST":  # Use this as indicator for start of test set
                is_train = False
            elif is_train:
                train_images.append(ii[0])
                train_captions.append(ii[1])
            else:
                test_images.append(ii[0])
                test_captions.append(ii[1])
    train_images = [os.path.join(base_path, ii) for ii in train_images]
    test_images = [os.path.join(base_path, ii) for ii in test_images]
    return train_images, train_captions, test_images, test_captions


class CaptionDataset(Dataset):
    def __init__(self, images, captions, tokenizer, is_train=True, image_size=224):
        from torchvision.transforms import Normalize, Compose, RandomResizedCrop, Resize, InterpolationMode, ToTensor

        self.images, self.captions, self.tokenizer = images, captions, tokenizer
        self.context_length = self.tokenizer.context_length

        self.mean, self.std = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
        interpolation = InterpolationMode.BICUBIC
        image_size = image_size if isinstance(image_size, (list, tuple)) else (image_size, image_size)
        self.transforms = Compose(
            [
                RandomResizedCrop(image_size, scale=(0.9, 1.0), interpolation=interpolation) if is_train else Resize(image_size, interpolation=interpolation),
                lambda image: image.convert("RGB"),
                ToTensor(),
                Normalize(mean=self.mean, std=self.std),
            ]
        )

    def tokenize(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        all_tokens = [[self.tokenizer.sot_token] + self.tokenizer.encode(text) + [self.tokenizer.eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), self.context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > self.context_length:
                tokens = tokens[: self.context_length]  # Truncate
                tokens[-1] = eot_token
            result[i, : len(tokens)] = torch.tensor(tokens)
        return result

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts


def collate_wrapper(batch):
    images, texts = list(zip(*batch))
    return (torch.stack(images), torch.stack(texts)), torch.arange(len(batch))


def build_dataset(data_path, caption_tokenizer, batch_size=64, image_size=224, num_workers=8):
    train_images, train_captions, test_images, test_captions = read_from_tsv(data_path)
    train_dataset = CaptionDataset(train_images, train_captions, tokenizer=caption_tokenizer, is_train=True, image_size=image_size)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_wrapper, pin_memory=True, sampler=None, drop_last=True
    )
    train_dataloader.num_samples = len(train_images)
    train_dataloader.num_batches = len(train_dataloader)

    test_dataset = CaptionDataset(test_images, test_captions, tokenizer=caption_tokenizer, is_train=False, image_size=image_size)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_wrapper, pin_memory=True, sampler=None, drop_last=True
    )
    test_dataloader.num_samples = len(test_images)
    test_dataloader.num_batches = len(test_dataloader)

    return train_dataloader, test_dataloader


def build_model(model_name, **model_kwargs):
    model_split = model_name.split(".")
    model_class = getattr(getattr(kecam, model_split[0]), model_split[1]) if len(model_split) == 2 else getattr(kecam.models, model_split[0])
    return model_class(**model_kwargs)


def build_optimizer(model, lr=1e-3, weight_decay=0.2, beta1=0.9, beta2=0.98, eps=1.0e-6):
    named_parameters = list(model.named_parameters())
    exclude = lambda name, param: param.ndim < 2 or any([ii in name for ii in ["bn", "ln", "bias", "logit_scale", "gamma"]])
    params = [
        {"params": [param for name, param in named_parameters if exclude(name, param) and param.requires_grad], "weight_decay": 0.0},
        {"params": [param for name, param in named_parameters if not exclude(name, param) and param.requires_grad], "weight_decay": weight_decay},
    ]
    optimizer = torch.optim.AdamW(params, lr=lr, betas=(beta1, beta2), eps=eps)
    return optimizer


def clip_loss(y_true, y_pred):
    return (torch.nn.functional.cross_entropy(y_pred, y_true) + torch.nn.functional.cross_entropy(y_pred.T, y_true)) / 2


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-d", "--data_path", type=str, default="datasets/coco_dog_cat/captions.tsv", help="tsv format dataset path")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("-e", "--epochs", type=int, default=30, help="Total epochs")
    parser.add_argument("-I", "--initial_epoch", type=int, default=0, help="Initial epoch when restore from previous interrupt")
    parser.add_argument("-s", "--basic_save_name", type=str, default="torch_clip_test", help="Basic save name for model and history")

    model = parser.add_argument_group("Model arguments")
    model.add_argument("-i", "--input_shape", type=int, default=224, help="Image model input shape")
    model.add_argument("-m", "--image_model", type=str, default="FlexiViTBase", help="Model name in format [sub_dir].[model_name] like beit.BeitBasePatch16")
    model.add_argument("--image_model_pretrained", type=str, default=None, help="If build model with pretrained weights. Set 'default' for model preset value")
    model.add_argument("--text_model", type=str, default="GPT2_Base", help="model from this repo `gpt2.[model_name]` like gpt2.GPT2_Base")
    model.add_argument(
        "--text_model_pretrained", type=str, default="default", help="Text model pretrained weight, default 'default' for using model preset value"
    )
    model.add_argument(
        "--tokenizer",
        type=str,
        default="GPT2Tokenizer",
        help="One of ['GPT2Tokenizer', 'SimpleTokenizer'], or tiktoken one ['gpt2', 'r50k_base', 'p50k_base', 'cl100k_base']",
    )
    model.add_argument("--latents_dim", type=int, default=512, help="hidden dimension of `image_latents` and `text_latents` before calculating similarity")

    lr_wd = parser.add_argument_group("Learning rate, weight decay arguments")
    lr_wd.add_argument("--lr", type=float, default=1e-3, help="Learning rate ")
    lr_wd.add_argument("--lr_warmup_steps", type=int, default=3, help="Learning rate warmup epochs")
    lr_wd.add_argument("--weight_decay", type=float, default=0.2, help="Weight decay")
    return parser.parse_known_args()[0]


if __name__ == "__main__":
    args = parse_arguments()

    # Always 0, no matter CUDA_VISIBLE_DEVICES
    device = torch.device("cuda:0") if torch.cuda.is_available() and int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")) > 0 else torch.device("cpu")

    caption_tokenizer = getattr(kecam.clip, args.tokenizer)() if hasattr(kecam.clip, args.tokenizer) else kecam.clip.TikToken(args.tokenizer)
    train_dataloader, test_dataloader = build_dataset(
        args.data_path, caption_tokenizer=caption_tokenizer, image_size=args.input_shape, batch_size=args.batch_size
    )
    (image, text), labels = next(iter(train_dataloader))
    print(">>>> Total train samples: {}, total test samples: {}".format(train_dataloader.num_samples, test_dataloader.num_samples))
    print(">>>> Data: image.shape: {}, text.shape: {}, labels.shape: {}".format(image.shape, text.shape, labels.shape))

    image_model_kwargs = {} if args.image_model_pretrained == "default" else {"pretrained": args.image_model_pretrained}
    image_model_kwargs.update({"input_shape": (3, args.input_shape, args.input_shape), "num_classes": args.latents_dim, "classifier_activation": None})
    print(">>>> image_model_kwargs:", image_model_kwargs)
    image_model = build_model(args.image_model, **image_model_kwargs)
    print(">>>> image_model name: {}, input_shape: {}, output_shape: {}".format(image_model.name, image_model.input_shape, image_model.output_shape))

    text_model_kwargs = {} if args.text_model_pretrained == "default" else {"pretrained": args.text_model_pretrained}
    text_model_kwargs.update({"vocab_size": caption_tokenizer.vocab_size, "include_top": False})
    print(">>>> text_model_kwargs:", text_model_kwargs)
    text_model = build_model(args.text_model, **text_model_kwargs)
    print(">>>> text_model name: {}, input_shape: {}, output_shape: {}".format(text_model.name, text_model.input_shape, text_model.output_shape))

    model, image_model, text_model = kecam.clip.convert_to_clip_model(image_model, text_model)
    model.to(device=device)

    if hasattr(torch, "compile") and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 6:
        print(">>>> Calling torch.compile")
        model = torch.compile(model)
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)

    basic_save_name = "clip_{}_{}".format(image_model.name, text_model.name)
    print(">>>> basic_save_name:", basic_save_name)
    callbacks = [
        kecam.imagenet.callbacks.CosineLrScheduler(args.lr, args.epochs, steps_per_epoch=train_dataloader.num_batches, warmup_steps=args.lr_warmup_steps),
        kecam.imagenet.callbacks.MyCheckpoint(basic_save_name=basic_save_name, save_path="checkpoints"),
        kecam.imagenet.callbacks.MyHistory(initial_file=os.path.join("checkpoints", basic_save_name + "_hist.json")),
    ]
    model.compile(optimizer=optimizer, loss=clip_loss, grad_max_norm=10.0, metrics=["acc"])
    model.fit(train_dataloader, epochs=args.epochs, validation_data=test_dataloader, callbacks=callbacks)
