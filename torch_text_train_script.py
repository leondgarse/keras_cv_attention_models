import os
import math
import time
import torch
import inspect
import requests
import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from contextlib import nullcontext

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

BUILDIN_DATASETS = {
    "tinyshakespeare": {"url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"},
}


class Datasets:
    def __init__(self, data_name, tokenizer, split="train", val_split=0.1, block_size=1024, batch_size=12):
        assert data_name in BUILDIN_DATASETS

        self.split, self.val_split, self.data_name, self.tokenizer = split, val_split, data_name, tokenizer
        self.save_path = os.path.join(os.path.expanduser("~"), ".keras", "datasets", self.data_name)
        data_file = "train.bin" if split == "train" else "val.bin"
        data_path = os.path.join(self.save_path, data_file)
        print(">>>> Load data from {}".format(data_path))
        if not os.path.exists(data_path):
            self.download_and_load(val_split)
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")

        self.block_size, self.batch_size = block_size, batch_size

    def download_and_load(self, val_split=0.1):
        train_bin_file, val_bin_file = os.path.join(self.save_path, "train.bin"), os.path.join(self.save_path, "val.bin")
        url = BUILDIN_DATASETS[self.data_name]["url"]
        target_file_path = os.path.join(self.save_path, os.path.basename(url))

        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path, exist_ok=True)
        if not os.path.exists(target_file_path):
            from keras_cv_attention_models.backend import get_file

            # print(">>>> Downloading to:", target_file_path)
            get_file(origin=url, cache_subdir=os.path.join("datasets", self.data_name))

        with open(target_file_path, "r") as ff:
            data = ff.read()
        total = len(data)
        train_split = 1 - val_split
        train_data = data[: int(total * train_split)]
        val_data = data[int(total * train_split) :]

        train_ids = self.tokenizer.encode(train_data)
        val_ids = self.tokenizer.encode(val_data)
        print(f"train has {len(train_ids):,} tokens")
        print(f"val has {len(val_ids):,} tokens")

        # export to bin files
        train_ids = np.array(train_ids, dtype=np.uint16)
        val_ids = np.array(val_ids, dtype=np.uint16)
        train_ids.tofile(train_bin_file)
        val_ids.tofile(val_bin_file)
        return train_bin_file, val_bin_file

    def get_random_batch(self):
        idx = torch.randint(len(self.data) - self.block_size, (self.batch_size,))
        xx = torch.stack([torch.from_numpy((self.data[i : i + self.block_size]).astype(np.int64)) for i in idx])
        yy = torch.stack([torch.from_numpy((self.data[i + 1 : i + 1 + self.block_size]).astype(np.int64)) for i in idx])
        return xx, yy


def build_optimizer(model, lr=1e-3, weight_decay=0.2, beta1=0.9, beta2=0.98, eps=1.0e-6, device_type="cuda"):
    named_parameters = list(model.named_parameters())
    exclude = lambda name, param: param.ndim < 2 or any([ii in name for ii in ["bn", "ln", "bias", "logit_scale"]])
    params = [
        {"params": [param for name, param in named_parameters if exclude(name, param) and param.requires_grad], "weight_decay": 0.0},
        {"params": [param for name, param in named_parameters if not exclude(name, param) and param.requires_grad], "weight_decay": weight_decay},
    ]

    # device_type = global_device.type
    use_fused = (device_type == "cuda") and ("fused" in inspect.signature(torch.optim.AdamW).parameters)
    print(f"using fused AdamW: {use_fused}")
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(params, lr=lr, betas=(beta1, beta2), eps=eps, **extra_args)

    return optimizer


# helps estimate an arbitrarily accurate loss over either split using many batches
def estimate_loss(model, dataset, eval_iters=200):
    with torch.no_grad():
        model.eval()
        # losses = torch.zeros(eval_iters)
        losses = 0
        for iter in range(eval_iters):
            xx, yy = dataset.get_random_batch()
            with global_context:
                logits, loss = model(xx, yy)
            losses += loss.item()
        model.train()
    return losses / eval_iters


# learning rate decay scheduler (cosine with warmup)
def cosine_lr(optimizer, total_steps, base_lr=1e-3, warmup_steps=10000):
    def _lr_adjuster(step):
        if step < warmup_steps:
            lr = base_lr * (step + 1) / warmup_steps
        else:
            lr = 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (total_steps - warmup_steps))) * base_lr
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr
        return lr

    return _lr_adjuster


def train(
    model,
    optimizer,
    train_data,
    val_data,
    scheduler,
    max_iters=600000,
    eval_interval=2000,
    gradient_accumulation_steps=5,
    log_interval=1,
    out_dir="checkpoints",
):
    iter_num = 0
    best_val_loss = 1e9
    train_x, train_y = train_data.get_random_batch()
    while iter_num < max_iters:
        t0 = time.time()
        scheduler(iter_num)

        # evaluate the loss on train/val sets and write checkpoints
        if iter_num > 0 and iter_num % eval_interval == 0:
            train_loss = estimate_loss(model, train_data)
            val_loss = estimate_loss(model, val_data)
            print(f"step {iter_num}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")
            if val_loss < best_val_loss:
                pre_best_ckpt = os.path.join(out_dir, "ckpt_val_loss_{:.4f}.pt".format(best_val_loss))
                if os.path.exists(pre_best_ckpt):
                    os.remove(pre_best_ckpt)

                best_val_loss = val_loss
                checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict()}
                print(f"saving checkpoint to {out_dir}")
                torch.save(checkpoint, os.path.join(out_dir, "ckpt_val_loss_{:.4f}.pt".format(val_loss)))
            torch.save(checkpoint, os.path.join(out_dir, "ckpt_latest.pt"))

        # forward backward update, with optional gradient accumulation to simulate larger batch size
        # and using the GradScaler if data type is float16
        for _ in range(gradient_accumulation_steps):
            with global_context:
                logits = model(train_x)
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), train_y.view(-1), ignore_index=-1)
            # immediately async prefetch next batch while model is doing the forward pass on the GPU
            train_x, train_y = train_data.get_random_batch()
            # backward pass, with gradient scaling if training in fp16
            global_scaler.scale(loss).backward()
        global_scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # clip the gradient
        global_scaler.step(optimizer)  # step the optimizer and scaler if training in fp16
        global_scaler.update()
        optimizer.zero_grad(set_to_none=True)  # flush the gradients as soon as we can, no need for this memory anymore

        if iter_num % log_interval == 0:
            lossf = loss.item()  # loss as float. note: this is a CPU-GPU sync point
            dt = time.time() - t0
            print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
        iter_num += 1


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model", type=str, default="GPT2_Base", help="model from this repo `[model_classs.model_name]` like llama2.LLaMA2_15M")
    parser.add_argument("-i", "--block_size", type=int, default=1024, help="input block size")
    parser.add_argument("-d", "--data_name", type=str, default="tinyshakespeare", help="dataset name")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("-e", "--max_iters", type=int, default=60000, help="max iters")
    parser.add_argument("-I", "--initial_epoch", type=int, default=0, help="Initial epoch when restore from previous interrupt")
    parser.add_argument("-s", "--basic_save_name", type=str, default="torch_clip_test", help="Basic save name for model and history")
    parser.add_argument(
        "-t",
        "--tokenizer",
        type=str,
        default="GPT2Tokenizer",
        help="One of ['GPT2Tokenizer', 'SimpleTokenizer', 'SentencePieceTokenizer'], or tiktoken one ['gpt2', 'r50k_base', 'p50k_base', 'cl100k_base']",
    )
    parser.add_argument("--pretrained", type=str, default=None, help="If build model with pretrained weights. Set 'default' for model preset value")

    # lr_wd = parser.add_argument_group("Learning rate, weight decay arguments")
    parser.add_argument("--lr", type=float, default=6e-4, help="Learning rate ")
    parser.add_argument("--lr_warmup_steps", type=int, default=2000, help="Learning rate warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    return parser.parse_known_args()[0]


if __name__ == "__main__":
    args = parse_arguments()

    caption_tokenizer = getattr(kecam.clip, args.tokenizer)() if hasattr(kecam.clip, args.tokenizer) else kecam.clip.TikToken(args.tokenizer)
    model_split = args.model.split(".")
    model_class = getattr(getattr(kecam, model_split[0]), model_split[1]) if len(model_split) == 2 else getattr(kecam.models, model_split[0])
    kwargs = {} if args.pretrained == "default" else {"pretrained": args.pretrained}
    model = model_class(max_block_size=args.block_size, vocab_size=caption_tokenizer.vocab_size, **kwargs)
    model.to(global_device)
    if hasattr(torch, "compile") and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 6:
        model = torch.compile(model)
    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay, beta1=0.9, beta2=0.95, device_type=global_device.type)
    scheduler = cosine_lr(optimizer, total_steps=args.max_iters, base_lr=args.lr, warmup_steps=args.lr_warmup_steps)

    # poor man's data loader
    train_data = Datasets(args.data_name, caption_tokenizer, split="train", block_size=args.block_size, batch_size=args.batch_size)
    val_data = Datasets(args.data_name, caption_tokenizer, split="val", block_size=args.block_size, batch_size=args.batch_size)

    train(
        model=model,
        optimizer=optimizer,
        train_data=train_data,
        val_data=val_data,
        scheduler=scheduler,
        max_iters=args.max_iters,
        eval_interval=2000,
        gradient_accumulation_steps=5,
        log_interval=1,
        out_dir="checkpoints",
    )
