import os
import keras_cv_attention_models
from tensorflow import keras
from keras_cv_attention_models.imagenet import callbacks, losses


def init_lr_scheduler(lr_base, lr_decay_steps, lr_min=1e-5, lr_decay_on_batch=False, lr_warmup=1e-4, lr_warmup_steps=0, lr_cooldown_steps=0):
    if isinstance(lr_decay_steps, list):
        constant_lr_sch = lambda epoch: callbacks.constant_scheduler(epoch, lr_base=lr_base, lr_decay_steps=lr_decay_steps, warmup_steps=lr_warmup_steps)
        lr_scheduler = keras.callbacks.LearningRateScheduler(constant_lr_sch)
        lr_total_epochs = lr_decay_steps[-1] + lr_decay_steps[0]  # 120 for lr_decay_steps=[30, 60, 90], lr_warmup_steps=4
    elif lr_decay_on_batch:
        lr_scheduler = callbacks.CosineLrScheduler(
            lr_base, lr_decay_steps, m_mul=0.5, t_mul=2.0, lr_min=lr_min, lr_warmup=lr_warmup, warmup_steps=lr_warmup_steps, cooldown_steps=lr_cooldown_steps
        )
        lr_total_epochs = lr_decay_steps + lr_cooldown_steps  # 105 for lr_decay_steps=100, lr_warmup_steps=4, lr_cooldown_steps=5
    else:
        lr_scheduler = callbacks.CosineLrSchedulerEpoch(
            lr_base, lr_decay_steps, m_mul=0.5, t_mul=2.0, lr_min=lr_min, lr_warmup=lr_warmup, warmup_steps=lr_warmup_steps, cooldown_steps=lr_cooldown_steps
        )
        lr_total_epochs = lr_decay_steps + lr_cooldown_steps  # 105 for lr_decay_steps=100, lr_warmup_steps=4, lr_cooldown_steps=5
    return lr_scheduler, lr_total_epochs


def init_optimizer(optimizer, lr_base, weight_decay):
    import tensorflow_addons as tfa

    optimizer = optimizer.lower()
    if optimizer == "sgd":
        optimizer = keras.optimizers.SGD(learning_rate=lr_base, momentum=0.9)
    elif optimizer == "rmsprop":
        optimizer = keras.optimizers.RMSprop(learning_rate=lr_base, momentum=0.9)
    elif optimizer == "lamb":
        bn_weights = ["bn/gamma", "bn/beta"]  # ["bn/moving_mean", "bn/moving_variance"] not in weights
        optimizer = tfa.optimizers.LAMB(learning_rate=lr_base, weight_decay_rate=weight_decay, exclude_from_weight_decay=bn_weights, global_clipnorm=1.0)
    elif optimizer == "adamw":
        optimizer = tfa.optimizers.AdamW(learning_rate=lr_base, weight_decay=lr_base * weight_decay)
    elif optimizer == "sgdw":
        optimizer = tfa.optimizers.SGDW(learning_rate=lr_base, momentum=0.9, weight_decay=lr_base * weight_decay)
    else:
        optimizer = getattr(keras.optimizers, optimizer.capitalize())(learning_rate=lr_base)
    return optimizer


def is_decoupled_weight_decay(optimizer):
    import tensorflow_addons as tfa

    optimizer = optimizer.inner_optimizer if isinstance(optimizer, keras.mixed_precision.LossScaleOptimizer) else optimizer
    return isinstance(optimizer, tfa.optimizers.weight_decay_optimizers.DecoupledWeightDecayExtension)


def init_loss(bce_threshold=1.0, label_smoothing=0):
    if bce_threshold >= 0 and bce_threshold < 1:
        loss = losses.BinaryCrossEntropyTimm(target_threshold=bce_threshold, label_smoothing=label_smoothing)
    else:
        loss = keras.losses.CategoricalCrossentropy(label_smoothing=label_smoothing)
    return loss


def init_model(model, input_shape=(224, 224, 3), num_classes=1000, pretrained=None, restore_path=None, **kwargs):
    print(">>> init_model kwargs:", kwargs)
    model = model.strip().split(".")
    if restore_path:
        import tensorflow_addons as tfa

        print(">>>> Restore model from:", restore_path)
        model = keras.models.load_model(restore_path)
    else:
        # model = cmt.CMTTiny(input_shape=input_shape, num_classes=num_classes, drop_connect_rate=0.2, drop_rate=0.2)
        # model = keras.applications.ResNet50(weights=None, input_shape=input_shape)
        # model = aotnet.AotNet50(num_classes=num_classes, input_shape=input_shape)
        if len(model) == 1:
            model = getattr(keras.applications, model[0])(classes=num_classes, weights=pretrained, input_shape=input_shape, **kwargs)
        else:
            model_class = getattr(getattr(keras_cv_attention_models, model[0]), model[1])
            model = model_class(num_classes=num_classes, input_shape=input_shape, pretrained=pretrained, **kwargs)
        print(">>>> Built model name:", model.name)
    return model


def compile_model(model, optimizer, lr_base, weight_decay, bce_threshold, label_smoothing):
    optimizer = optimizer.lower()
    if optimizer == "sgd" and weight_decay > 0:
        # Add L2 regularizer
        from keras_cv_attention_models import model_surgery

        model = model_surgery.add_l2_regularizer_2_model(model, weight_decay=weight_decay, apply_to_batch_normal=False)
    optimizer = init_optimizer(optimizer, lr_base, weight_decay)
    loss = init_loss(bce_threshold, label_smoothing)
    print(">>>> Loss: {}, Optimizer: {}".format(loss.__class__.__name__, optimizer.__class__.__name__))
    model.compile(optimizer=optimizer, loss=loss, metrics=["acc"])
    return model


# @tf.function(jit_compile=True)
def train(compiled_model, epochs, train_dataset, test_dataset=None, initial_epoch=0, lr_scheduler=None, basic_save_name=None):
    if compiled_model.compiled_loss is None:
        print(">>>> Error: Model NOT compiled.")
        return None

    steps_per_epoch = len(train_dataset)
    if hasattr(lr_scheduler, "steps_per_epoch") and lr_scheduler.steps_per_epoch == -1:
        lr_scheduler.build(steps_per_epoch)
    is_lr_on_batch = True if hasattr(lr_scheduler, "steps_per_epoch") and lr_scheduler.steps_per_epoch > 0 else False

    if basic_save_name is None:
        basic_save_name = "{}".format(compiled_model.name)
    # ckpt_path = os.path.join("checkpoints", basic_save_name + "epoch_{epoch:03d}_val_acc_{val_acc:.4f}.h5")
    # cur_callbacks = [keras.callbacks.ModelCheckpoint(ckpt_path, monitor="val_loss", save_best_only=True)]
    # cur_callbacks = [keras.callbacks.ModelCheckpoint(os.path.join("checkpoints", basic_save_name + ".h5"))]
    cur_callbacks = [callbacks.MyCheckpoint(basic_save_name, monitor="val_acc")]
    hist_file = os.path.join("checkpoints", basic_save_name + "_hist.json")
    if initial_epoch == 0 and os.path.exists(hist_file):
        # os.remove(hist_file)
        os.rename(hist_file, hist_file + ".bak")
    cur_callbacks.append(callbacks.MyHistory(initial_file=hist_file))
    cur_callbacks.append(keras.callbacks.TerminateOnNaN())
    if lr_scheduler is not None:
        cur_callbacks.append(lr_scheduler)

    if lr_scheduler is not None and is_decoupled_weight_decay(compiled_model.optimizer):
        print(">>>> Append weight decay callback...")
        lr_base, wd_base = compiled_model.optimizer.lr.numpy(), compiled_model.optimizer.weight_decay.numpy()
        wd_callback = callbacks.OptimizerWeightDecay(lr_base, wd_base, is_lr_on_batch=is_lr_on_batch)
        cur_callbacks.append(wd_callback)  # should be after lr_scheduler

    compiled_model.fit(
        train_dataset,
        epochs=epochs,
        verbose=1,
        callbacks=cur_callbacks,
        initial_epoch=initial_epoch,
        steps_per_epoch=steps_per_epoch,
        validation_data=test_dataset,
        use_multiprocessing=True,
        workers=8,
    )
