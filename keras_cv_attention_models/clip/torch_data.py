import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, Resize, InterpolationMode, ToTensor


def read_from_tsv(data_path):
    import csv

    delimiter = "\t" if data_path.endswith(".tsv") else ","
    train_images, train_captions, test_images, test_captions, base_path, is_train = [], [], [], [], "", True
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
    if len(base_path) > 0:
        train_images = [os.path.join(base_path, ii) for ii in train_images]
        test_images = [os.path.join(base_path, ii) for ii in test_images]
    return train_images, train_captions, test_images, test_captions


class CaptionDataset(Dataset):
    def __init__(self, images, captions, tokenizer, is_train=True, image_size=224):
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


def init_dataset(data_path, caption_tokenizer, batch_size=64, image_size=224, num_workers=8):
    train_images, train_captions, test_images, test_captions = read_from_tsv(data_path)
    train_dataset = CaptionDataset(train_images, train_captions, tokenizer=caption_tokenizer, is_train=True, image_size=image_size)
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_wrapper, pin_memory=True, sampler=None, drop_last=True
    )

    test_dataset = CaptionDataset(test_images, test_captions, tokenizer=caption_tokenizer, is_train=False, image_size=image_size)
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, collate_fn=collate_wrapper, pin_memory=True, sampler=None, drop_last=True
    )

    return train_dataloader, test_dataloader
