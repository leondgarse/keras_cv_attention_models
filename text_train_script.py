import os
import numpy as np
import kecam

if kecam.backend.is_torch_backend:  # os.environ["KECAM_BACKEND"] = "torch"
    import torch
    from collections import namedtuple
    from contextlib import nullcontext

    global_strategy = namedtuple("strategy", ["scope"])(nullcontext)  # Fake
    # Always 0, no matter CUDA_VISIBLE_DEVICES
    global_device = torch.device("cuda:0") if torch.cuda.is_available() and int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")) >= 0 else torch.device("cpu")
else:
    import tensorflow as tf
    from keras_cv_attention_models.imagenet.train_func import init_global_strategy

    global_strategy = init_global_strategy(enable_float16=len(tf.config.experimental.get_visible_devices("GPU")) > 0)

BUILDIN_DATASETS = {
    "tinyshakespeare": {"url": "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"},
}


class DatasetGen:
    def __init__(self, data_name, tokenizer, split="train", val_split=0.1, block_size=1024, batch_size=12, steps_per_epoch=2000):
        assert data_name in BUILDIN_DATASETS

        self.split, self.val_split, self.data_name, self.tokenizer = split, val_split, data_name, tokenizer
        self.save_path = os.path.join(os.path.expanduser("~"), ".keras", "datasets", self.data_name)

        self.tokenizer_name = tokenizer.__class__.__name__
        data_file = "train_{}.bin".format(self.tokenizer_name) if split == "train" else "val_{}.bin".format(self.tokenizer_name)
        data_path = os.path.join(self.save_path, data_file)
        print(">>>> Load data from {}".format(data_path))
        if not os.path.exists(data_path):
            self.download_and_load(val_split)
        self.data = np.memmap(data_path, dtype=np.uint16, mode="r")

        self.block_size, self.batch_size, self.steps_per_epoch = block_size, batch_size, steps_per_epoch

    def download_and_load(self, val_split=0.1):
        train_bin_file = os.path.join(self.save_path, "train_{}.bin".format(self.tokenizer_name))
        val_bin_file = os.path.join(self.save_path, "val_{}.bin".format(self.tokenizer_name))
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

    def __iter__(self):
        self.generated = 0
        return self

    def __len__(self):
        return self.steps_per_epoch

    def __next__(self):
        if self.generated < self.steps_per_epoch:
            idx = np.random.uniform(0, len(self.data) - self.block_size, size=(self.batch_size,)).astype(np.int64)
            xx = [self.data[ii : ii + self.block_size] for ii in idx]
            yy = [self.data[ii + 1 : ii + 1 + self.block_size] for ii in idx]
            self.generated += 1
            return np.stack(xx).astype(np.int64), np.stack(yy).astype(np.int64)
        else:
            raise StopIteration

    def __getitem__(self, idx):
        return self.__next__()


def build_torch_dataset(train_dataset_gen, test_dataset_gen):
    from torch.utils.data import DataLoader, IterableDataset

    class DD(IterableDataset):
        def __init__(self, dataset):
            super().__init__()
            self.dataset = dataset

        def __len__(self):
            return len(self.dataset)

        def __iter__(self):
            return iter(self.dataset)

    train_dataset = DataLoader(DD(train_dataset_gen), batch_size=None)
    test_dataset = DataLoader(DD(test_dataset_gen), batch_size=None)
    return train_dataset, test_dataset


def build_tf_dataset(train_dataset_gen, test_dataset_gen):
    block_size, train_steps_per_epoch, test_steps_per_epoch = train_dataset_gen.block_size, train_dataset_gen.steps_per_epoch, test_dataset_gen.steps_per_epoch
    output_signature = (tf.TensorSpec(shape=(None, block_size), dtype=tf.int64), tf.TensorSpec(shape=(None, block_size), dtype=tf.int64))
    options = tf.data.Options()
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA

    train_dataset = tf.data.Dataset.from_generator(lambda: ((xx, yy) for xx, yy in iter(train_dataset_gen)), output_signature=output_signature)
    train_dataset = train_dataset.apply(tf.data.experimental.assert_cardinality(train_steps_per_epoch)).with_options(options)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_generator(lambda: ((xx, yy) for xx, yy in iter(test_dataset_gen)), output_signature=output_signature)
    test_dataset = test_dataset.apply(tf.data.experimental.assert_cardinality(test_steps_per_epoch)).with_options(options)
    return train_dataset, test_dataset


def build_torch_optimizer(model, lr=1e-3, weight_decay=0.2, beta1=0.9, beta2=0.95, eps=1.0e-6):
    import inspect

    named_parameters = list(model.named_parameters())
    exclude = lambda name, param: param.ndim < 2 or any([ii in name for ii in ["gamma", "beta", "bias", "positional_embedding", "no_weight_decay"]])
    params = [
        {"params": [param for name, param in named_parameters if exclude(name, param) and param.requires_grad], "weight_decay": 0.0},
        {"params": [param for name, param in named_parameters if not exclude(name, param) and param.requires_grad], "weight_decay": weight_decay},
    ]

    device_type = named_parameters[0][1].device.type
    use_fused = (device_type == "cuda") and ("fused" in inspect.signature(torch.optim.AdamW).parameters)
    print(">>>> using fused AdamW:", use_fused)
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(params, lr=lr, betas=(beta1, beta2), eps=eps, **extra_args)
    return optimizer


def build_tf_optimizer(lr=1e-3, weight_decay=0.2, beta1=0.9, beta2=0.95, eps=1.0e-6):
    no_weight_decay = ["/gamma", "/beta", "/bias", "/positional_embedding", "/no_weight_decay"]

    optimizer = tf.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay, beta_1=beta1, beta_2=beta2, epsilon=eps)
    optimizer.exclude_from_weight_decay(var_names=no_weight_decay)
    return optimizer


@kecam.backend.register_keras_serializable(package="kecamLoss")
def ravel_loss(y_true, y_pred):
    y_true = kecam.backend.functional.reshape(y_true, [-1])
    y_pred = kecam.backend.functional.reshape(y_pred, [-1, y_pred.shape[-1]])
    return kecam.backend.losses.sparse_categorical_crossentropy(y_true, y_pred, from_logits=True)


def parse_arguments():
    import argparse

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-m", "--model", type=str, default="GPT2_Base", help="model from this repo `[model_classs.model_name]` like llama2.LLaMA2_15M")
    parser.add_argument("-i", "--block_size", type=int, default=1024, help="input block size")
    parser.add_argument("-d", "--data_name", type=str, default="tinyshakespeare", help="dataset name")
    parser.add_argument("-b", "--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("-e", "--epochs", type=int, default=30, help="training epochs, total max iterations=epochs * steps_per_epoch")
    parser.add_argument("-S", "--steps_per_epoch", type=int, default=2000, help="training steps per epoch")
    parser.add_argument("-I", "--initial_epoch", type=int, default=0, help="Initial epoch when restore from previous interrupt")
    parser.add_argument("-s", "--basic_save_name", type=str, default=None, help="Basic save name for model and history")
    parser.add_argument("-r", "--restore_path", type=str, default=None, help="Restore model from saved h5 or pt file. Higher priority than model")
    parser.add_argument(
        "-t",
        "--tokenizer",
        type=str,
        default="GPT2Tokenizer",
        help="One of ['GPT2Tokenizer', 'SimpleTokenizer', 'SentencePieceTokenizer'], or tiktoken one ['gpt2', 'r50k_base', 'p50k_base', 'cl100k_base']",
    )
    parser.add_argument("--pretrained", type=str, default=None, help="If build model with pretrained weights. Set 'default' for model preset value")

    # lr_wd = parser.add_argument_group("Learning rate, weight decay arguments")
    parser.add_argument("--lr_base_512", type=float, default=1e-4, help="Learning rate for batch_size=512, lr = lr_base_512 * 512 / batch_size")
    parser.add_argument("--lr_warmup_steps", type=int, default=3, help="Learning rate warmup steps")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay")
    return parser.parse_known_args()[0]


if __name__ == "__main__":
    args = parse_arguments()

    caption_tokenizer = getattr(kecam.clip, args.tokenizer)() if hasattr(kecam.clip, args.tokenizer) else kecam.clip.TikToken(args.tokenizer)
    train_dataset_gen = DatasetGen(
        args.data_name, caption_tokenizer, split="train", block_size=args.block_size, batch_size=args.batch_size, steps_per_epoch=args.steps_per_epoch
    )
    test_dataset_gen = DatasetGen(
        args.data_name, caption_tokenizer, split="val", block_size=args.block_size, batch_size=args.batch_size, steps_per_epoch=args.steps_per_epoch // 10
    )
    input_data, output_data = next(iter(train_dataset_gen))
    print(">>>> Total train batches: {}, total test batches: {}".format(len(train_dataset_gen), len(test_dataset_gen)))
    print(">>>> Data: input_data.shape: {}, output_data.shape: {}".format(input_data.shape, output_data.shape))
    if kecam.backend.is_torch_backend:
        train_dataset, test_dataset = build_torch_dataset(train_dataset_gen, test_dataset_gen)
    else:
        train_dataset, test_dataset = build_tf_dataset(train_dataset_gen, test_dataset_gen)

    lr = args.lr_base_512 * args.batch_size / 512
    print(">>>> lr:", lr)

    with global_strategy.scope():
        if args.restore_path is None or kecam.backend.is_torch_backend:
            model_split = args.model.split(".")
            model_class = getattr(getattr(kecam, model_split[0]), model_split[1]) if len(model_split) == 2 else getattr(kecam.models, model_split[0])
            kwargs = {} if args.pretrained == "default" else {"pretrained": args.pretrained}
            kwargs.update({"max_block_size": args.block_size, "vocab_size": caption_tokenizer.vocab_size})
            print(">>>> model_kwargs:", kwargs)
            model = model_class(**kwargs)
            print(">>>> model name: {}, input_shape: {}, output_shape: {}".format(model.name, model.input_shape, model.output_shape))
        else:
            print(">>>> Reload from:", args.restore_path)
            model = kecam.backend.models.load_model(args.restore_path)

        if kecam.backend.is_torch_backend:
            model.to(device=global_device)
            if hasattr(torch, "compile") and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 6:
                print(">>>> Calling torch.compile")
                model = torch.compile(model)
            optimizer = build_torch_optimizer(model, lr=args.lr, weight_decay=args.weight_decay, beta1=0.9, beta2=0.95)
            model.compile(optimizer=optimizer, loss=ravel_loss, grad_accumulate=5)

            if args.restore_path is not None:
                print(">>>> Reload weights from:", args.restore_path)
                model.load(args.restore_path)  # Reload wights after compile
        elif model.optimizer is None:
            optimizer = build_tf_optimizer(lr=lr, weight_decay=args.weight_decay, beta1=0.9, beta2=0.95)
            model.compile(optimizer=optimizer, loss=ravel_loss)

        basic_save_name = args.basic_save_name or "text_{}_{}".format(model.name, kecam.backend.backend())
        print(">>>> basic_save_name:", basic_save_name)

        epochs = (args.epochs // args.steps_per_epoch) if args.epochs > args.steps_per_epoch else args.epochs
        warmup_steps = (args.lr_warmup_steps // args.steps_per_epoch) if args.lr_warmup_steps > args.steps_per_epoch else args.lr_warmup_steps
        lr_scheduler = kecam.imagenet.callbacks.CosineLrScheduler(lr, epochs, steps_per_epoch=args.steps_per_epoch, warmup_steps=warmup_steps)
        other_kwargs = {}
        latest_save, hist = kecam.imagenet.train_func.train(
            compiled_model=model,
            epochs=epochs,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            initial_epoch=args.initial_epoch,
            lr_scheduler=lr_scheduler,
            basic_save_name=basic_save_name,
            init_callbacks=[],
            logs=None,
            **other_kwargs,
        )
