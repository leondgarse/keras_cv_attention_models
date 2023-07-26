"""data"""
import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Normalize, Compose, RandomResizedCrop, InterpolationMode, ToTensor
from keras_cv_attention_models import clip

device = torch.device("cuda:" + os.environ.get("CUDA_VISIBLE_DEVICES", "0")) if torch.cuda.is_available() else torch.device("cpu")
device_type = device.type


class CsvDataset(Dataset):
    def __init__(self, input_filename, tokenizer, image_size=224, sep="\t"):
        df = pd.read_csv(input_filename, header=None, sep=sep, names=["image", "caption"])
        self.images = df["image"].tolist()
        self.captions = df["caption"].tolist()

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

    def tokenize(self, texts, context_length: int = 77):
        if isinstance(texts, str):
            texts = [texts]
        all_tokens = [[self.tokenizer.sot_token] + self.tokenizer.encode(text) + [self.tokenizer.eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                tokens = tokens[:context_length]  # Truncate
                tokens[-1] = eot_token
            result[i, : len(tokens)] = torch.tensor(tokens)
        return result

    def __len__(self):
        return len(self.captions)

    def __getitem__(self, idx):
        images = self.transforms(Image.open(str(self.images[idx])))
        texts = self.tokenize([str(self.captions[idx])])[0]
        return images, texts


caption_tokenizer = clip.SimpleTokenizer()
dataset = CsvDataset("datasets/coco_dog_cat/captions.tsv", tokenizer=caption_tokenizer)
num_samples = len(dataset)

batch_size, num_workers = 4, 8
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, sampler=None, drop_last=True)
dataloader.num_samples = num_samples
dataloader.num_batches = len(dataloader)
print(">>>> Data:", [ii.shape for ii in next(iter(dataloader))])

"""model"""
import torch
from torch import nn
import torch.nn.functional as F

from keras_cv_attention_models import clip, gpt2, beit, backend


class CLIP(nn.Module):
    def __init__(self, image_model, text_model):
        super().__init__()
        self.image_model, self.text_model = image_model, text_model
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def lock_image_tower(self, unlocked_groups=0, freeze_bn_stats=False):
        # lock image tower as per LiT - https://arxiv.org/abs/2111.07991
        self.visual.lock(unlocked_groups=unlocked_groups, freeze_bn_stats=freeze_bn_stats)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.visual.set_grad_checkpointing(enable)
        self.transformer.grad_checkpointing = enable

    def forward(self, image, text):
        image_features = F.normalize(self.image_model(image), dim=-1)
        text_features = F.normalize(self.text_model(text), dim=-1)
        return image_features, text_features, self.logit_scale.exp()


text_model = gpt2.GPT2_Base(include_top=False)
text_inputs = text_model.inputs[0]
text_outputs = text_model.outputs[0]
text_outputs = clip.models.text_model_index_header(text_inputs, text_outputs, 512)
text_model = backend.models.Model(text_inputs, text_outputs)
# text_model(torch.ones([1, 77], dtype=torch.long)).shape
image_model = beit.ViT(num_classes=512, classifier_activation=None)
model = CLIP(image_model, text_model)
model.to(device=device)
# print({ii:jj.shape for ii , jj in model.named_parameters()})

"""optimizer"""
from torch import optim

lr, wd, beta1, beta2, eps = 1e-3, 0.2, 0.9, 0.98, 1.0e-6
named_parameters = list(model.named_parameters())
exclude = lambda n, p: p.ndim < 2 or "bn" in n or "ln" in n or "bias" in n or "logit_scale" in n or "class_tokens" in n
params = [
    {"params": [p for n, p in named_parameters if exclude(n, p) and p.requires_grad], "weight_decay": 0.0},
    {"params": [p for n, p in named_parameters if not exclude(n, p) and p.requires_grad], "weight_decay": wd},
]
optimizer = optim.AdamW(params, lr=lr, betas=(beta1, beta2), eps=eps)

"""lr"""


def cosine_lr(optimizer, base_lr, warmup_length, steps):
    def _lr_adjuster(step):
        if step < warmup_length:
            lr = base_lr * (step + 1) / warmup_length
        else:
            e = step - warmup_length
            es = steps - warmup_length
            lr = 0.5 * (1 + np.cos(np.pi * e / es)) * base_lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    return _lr_adjuster


wd, warmup, accum_freq, epochs = 0.2, 10000, 1, 30
total_steps = (dataloader.num_batches // accum_freq) * epochs
scheduler = cosine_lr(optimizer, lr, warmup, total_steps)

"""loss"""


def clip_loss(image_features, text_features, logit_scale):
    logits_per_image = logit_scale * image_features @ text_features.T
    logits_per_text = logit_scale * text_features @ image_features.T

    labels = torch.arange(logits_per_image.shape[0], device=image_features.device, dtype=torch.long)
    return (F.cross_entropy(logits_per_image, labels) + F.cross_entropy(logits_per_text, labels)) / 2


"""train"""
import math
from tqdm import tqdm
from contextlib import nullcontext

if device_type == "cpu":
    scaler = torch.cuda.amp.GradScaler(enabled=False)
    global_context = nullcontext()
    input_dtype = torch.float32
else:
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    global_context = torch.amp.autocast(device_type=device_type, dtype=torch.float16)
    input_dtype = torch.float16

grad_clip_norm = 10.0
start_epoch = 0
bar_format = "{n_fmt}/{total_fmt} [{bar:30}] - ETA: {elapsed}<{remaining} {rate_fmt}{postfix}{desc}"
for epoch in range(start_epoch, epochs):
    model.train()
    process_bar = tqdm(enumerate(dataloader), total=dataloader.num_batches, bar_format=bar_format, ascii=".>>=")
    for id, batch in process_bar:
        step = dataloader.num_batches * epoch + id
        scheduler(step)

        images, texts = batch
        images = images.to(device=device, dtype=input_dtype, non_blocking=True)
        texts = texts.to(device=device, dtype=torch.long, non_blocking=True)
        optimizer.zero_grad()

        with global_context:
            image_out, text_out, logit_scale = model(images, texts)
            losses = clip_loss(image_out, text_out, logit_scale)
        scaler.scale(losses).backward()

        if grad_clip_norm > 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm, norm_type=2.0)
        scaler.step(optimizer)
        scaler.update()

        with torch.no_grad():
            model.logit_scale.clamp_(0, math.log(100))  # clamp to 4.6052 = ln(100)
        process_bar.desc = " - loss: {:.4f}".format(losses)
        process_bar.refresh()
    print()
