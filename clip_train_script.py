import os
import kecam  # os.environ["KECAM_BACKEND"] = "torch"


if kecam.backend.is_torch_backend:
    import torch
    from collections import namedtuple
    from contextlib import nullcontext
    from keras_cv_attention_models.clip import torch_data as data

    global_strategy = namedtuple("strategy", ["scope"])(nullcontext)  # Fake
else:
    import tensorflow as tf

    from keras_cv_attention_models.clip import tf_data as data
    from keras_cv_attention_models.imagenet.train_func import init_global_strategy

    global_strategy = init_global_strategy()


def build_model(model_name, **model_kwargs):
    model_split = model_name.split(".")
    model_class = getattr(getattr(kecam, model_split[0]), model_split[1]) if len(model_split) == 2 else getattr(kecam.models, model_split[0])
    return model_class(**model_kwargs)


def build_torch_optimizer(model, lr=1e-3, weight_decay=0.2, beta1=0.9, beta2=0.98, eps=1.0e-6):
    named_parameters = list(model.named_parameters())
    exclude = lambda name, param: param.ndim < 2 or any([ii in name for ii in ["bn", "ln", "bias", "logit_scale", "gamma"]])
    params = [
        {"params": [param for name, param in named_parameters if exclude(name, param) and param.requires_grad], "weight_decay": 0.0},
        {"params": [param for name, param in named_parameters if not exclude(name, param) and param.requires_grad], "weight_decay": weight_decay},
    ]
    optimizer = torch.optim.AdamW(params, lr=lr, betas=(beta1, beta2), eps=eps)
    return optimizer


def build_tf_optimizer(lr=1e-3, weight_decay=0.2, beta1=0.9, beta2=0.98, eps=1.0e-6):
    no_weight_decay = ["/gamma", "/beta", "/bias", "/positional_embedding", "/no_weight_decay"]

    optimizer = tf.optimizers.AdamW(learning_rate=lr, weight_decay=weight_decay, beta_1=0.9, beta_2=0.98, epsilon=1e-6)
    optimizer.exclude_from_weight_decay(var_names=no_weight_decay)
    return optimizer


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

    caption_tokenizer = getattr(kecam.clip, args.tokenizer)() if hasattr(kecam.clip, args.tokenizer) else kecam.clip.TikToken(args.tokenizer)
    train_dataset, test_dataset = data.init_dataset(
        args.data_path, caption_tokenizer=caption_tokenizer, image_size=args.input_shape, batch_size=args.batch_size
    )
    (image, text), labels = next(iter(train_dataset))
    print(">>>> Total train batches: {}, total test batches: {}".format(len(train_dataset), len(test_dataset)))
    print(">>>> Data: image.shape: {}, text.shape: {}, labels.shape: {}".format(image.shape, text.shape, labels.shape))

    with global_strategy.scope():
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

        if kecam.backend.is_torch_backend:
            # Always 0, no matter CUDA_VISIBLE_DEVICES
            device = torch.device("cuda:0") if torch.cuda.is_available() and int(os.environ.get("CUDA_VISIBLE_DEVICES", "0")) > 0 else torch.device("cpu")
            model.to(device=device)
            if hasattr(torch, "compile") and torch.cuda.is_available() and torch.cuda.get_device_capability()[0] > 6:
                print(">>>> Calling torch.compile")
                model = torch.compile(model)
            optimizer = build_torch_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
        else:
            optimizer = build_tf_optimizer(lr=args.lr, weight_decay=args.weight_decay)

        # callbacks = [
        #     kecam.imagenet.callbacks.CosineLrScheduler(args.lr, args.epochs, steps_per_epoch=len(train_dataset), warmup_steps=args.lr_warmup_steps),
        #     kecam.imagenet.callbacks.MyCheckpoint(basic_save_name=basic_save_name, save_path="checkpoints"),
        #     kecam.imagenet.callbacks.MyHistory(initial_file=os.path.join("checkpoints", basic_save_name + "_hist.json")),
        # ]
        model.compile(optimizer=optimizer, loss=kecam.clip.losses.clip_loss, metrics=["acc"])
        # model.fit(train_dataset, epochs=args.epochs, validation_data=test_dataset, callbacks=callbacks)

        basic_save_name = "clip_{}_{}".format(image_model.name, text_model.name)
        print(">>>> basic_save_name:", basic_save_name)
        lr_scheduler = kecam.imagenet.callbacks.CosineLrScheduler(args.lr, args.epochs, steps_per_epoch=len(train_dataset), warmup_steps=args.lr_warmup_steps)
        other_kwargs = {}
        latest_save, hist = kecam.imagenet.train_func.train(
            compiled_model=model,
            epochs=args.epochs,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            initial_epoch=0,
            lr_scheduler=lr_scheduler,
            basic_save_name=basic_save_name,
            init_callbacks=[],
            logs=None,
            **other_kwargs,
        )
