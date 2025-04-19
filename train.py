# Standard library
import argparse
import logging
import os
import random

# Third-party libraries
import torch
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.v2.functional import normalize, to_dtype
from torchvision.utils import save_image

# Local modules
from backbones import get_model
from data import ArcIrisDataLoader, ArcIrisDataset
from data.augmentations import blur, random_noise
from eval import validation, verification
from utils import (
    CombinedMarginLoss,
    PolynomialLRWarmup,
    PartialFC_V2_DP,
    CallBackLogging,
    get_config,
    AverageMeter,
    init_logging,
)


def main(args):
    cfg = get_config(args.config)

    assert torch.cuda.is_available()
    torch.cuda.set_device(0)

    os.makedirs(cfg.output, exist_ok=True)
    init_logging(0, cfg.output)

    summary_writer = SummaryWriter(log_dir=os.path.join(cfg.output, "tensorboard"))

    train_ds = ArcIrisDataset(root_dir=cfg.train_dir)
    train_dl = ArcIrisDataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, pin_memory=True)

    backbone = get_model(cfg.network, dropout=0.0, fp16=cfg.fp16, num_features=cfg.embedding_size).cuda()
    backbone = torch.nn.DataParallel(backbone)
    backbone.train()

    margin_loss = CombinedMarginLoss(
        64,
        cfg.margin_list[0],
        cfg.margin_list[1],
        cfg.margin_list[2],
        cfg.interclass_filtering_threshold,
    )

    module_partial_fc = PartialFC_V2_DP(margin_loss, cfg.embedding_size, cfg.num_classes, cfg.sample_rate, fp16=False)
    module_partial_fc.train().cuda()

    if cfg.optimizer == "sgd":
        # TODO the params of partial fc must be last in the params list
        opt = torch.optim.SGD(params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}], lr=cfg.lr, momentum=0.9, weight_decay=cfg.weight_decay)
    elif cfg.optimizer == "adamw":
        opt = torch.optim.AdamW(params=[{"params": backbone.parameters()}, {"params": module_partial_fc.parameters()}], lr=cfg.lr, weight_decay=cfg.weight_decay)
    else:
        raise

    cfg.total_batch_size = cfg.batch_size
    cfg.warmup_step = cfg.num_image // cfg.total_batch_size * cfg.warmup_epoch
    cfg.total_step = cfg.num_image // cfg.total_batch_size * cfg.num_epoch

    lr_scheduler = PolynomialLRWarmup(
        optimizer=opt,
        warmup_iters=cfg.warmup_step,
        total_iters=cfg.total_step,
    )

    start_epoch = 0
    global_step = 0

    # TODO Resume does not seem to work right.
    if cfg.resume:
        dict_checkpoint = torch.load(os.path.join(cfg.output, f"checkpoint.pt"))
        start_epoch = dict_checkpoint["epoch"]
        global_step = dict_checkpoint["global_step"]
        backbone.module.load_state_dict(dict_checkpoint["state_dict_backbone"])
        module_partial_fc.load_state_dict(dict_checkpoint["state_dict_softmax_fc"])
        opt.load_state_dict(dict_checkpoint["state_optimizer"])
        lr_scheduler.load_state_dict(dict_checkpoint["state_lr_scheduler"])

    for key, value in cfg.items():
        num_space = 25 - len(key)
        logging.info(": " + key + " " * num_space + str(value))

    if cfg.val_target:
        path = os.path.join(cfg.val_target)
        if os.path.exists(path):
            val_bin = verification.load_bin(path)

    best_val = 0
    patience = 10
    no_improve_counter = 0

    callback_logging = CallBackLogging(
        frequent=cfg.frequent,
        total_step=cfg.total_step,
        batch_size=cfg.batch_size,
        start_step = global_step,
        writer=summary_writer,
    )

    loss_am = AverageMeter()
    amp = torch.amp.GradScaler('cuda', growth_interval=100)

    for epoch in range(start_epoch, cfg.num_epoch):
        for batch, local_labels in train_dl:
            global_step += 1

            batch = batch.cuda(non_blocking=True)
            local_labels = local_labels.cuda(non_blocking=True)

            # Convert uint8 grayscale 0-255 values to 0-1 float values
            batch = to_dtype(batch, dtype=torch.float32, scale=True)
            images = batch[0]
            masks = batch[1]

            # Save image preview before augmentation
            if cfg.debug:
                image_tensor = images[0]
                print(f"Before augmentation:")
                print(f"images.device: {images.device}")
                print(f"images.shape: {images.shape}")
                print(f"image_tensor.device: {image_tensor.device}")
                print(f"image_tensor.shape: {image_tensor.shape}")
                print(f"image_tensor.shape: {image_tensor.shape}")
                print(f"image_tensor.min(): {image_tensor.min()}")
                print(f"image_tensor.max(): {image_tensor.max()}")
                save_image(image_tensor, 'debug_before_augmentation.png')

            # Apply augmentation based on strategy from configuration.
            if cfg.strategy == "noise":
                images = random_noise(images, masks)
            elif cfg.strategy == "blur":
                images = blur(images, masks)
            elif cfg.strategy == "randomly_noise_or_blur":
                if random.choice([True, False]):
                    images = random_noise(images, masks)
                else:
                    images = blur(images, masks)
            else:
                cfg.strategy = ""

            # Save image preview after augmentation
            if cfg.debug:
                image_tensor = images[0]
                print(f"After augmentation:")
                print(f"images.device: {images.device}")
                print(f"images.shape: {images.shape}")
                print(f"image_tensor.device: {image_tensor.device}")
                print(f"image_tensor.shape: {image_tensor.shape}")
                print(f"image_tensor.shape: {image_tensor.shape}")
                print(f"image_tensor.min(): {image_tensor.min()}")
                print(f"image_tensor.max(): {image_tensor.max()}")
                save_image(image_tensor, 'debug_after_augmentation.png')

            # Normalizes the grayscale image and expands it from 1 channel to 3 identical grayscale channels
            images = normalize(images, mean=[0.5], std=[0.5])
            target_shape = list(images.shape)
            target_shape[-3] = 3
            images = images.expand(*target_shape)

            if cfg.debug:
                cfg.debug = False
                image_tensor = images[0]
                print(f"After normalization:")
                print(f"images.device: {images.device}")
                print(f"images.shape: {images.shape}")
                print(f"image_tensor.device: {image_tensor.device}")
                print(f"image_tensor.shape: {image_tensor.shape}")
                print(f"image_tensor.shape: {image_tensor.shape}")
                print(f"image_tensor.min(): {image_tensor.min()}")
                print(f"image_tensor.max(): {image_tensor.max()}")

            local_embeddings = backbone(images)
            loss: torch.Tensor = module_partial_fc(local_embeddings, local_labels)

            if cfg.fp16:
                amp.scale(loss).backward()
                if global_step % cfg.gradient_acc == 0:
                    amp.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    amp.step(opt)
                    amp.update()
                    opt.zero_grad()
                    lr_scheduler.step()
            else:
                loss.backward()
                if global_step % cfg.gradient_acc == 0:
                    torch.nn.utils.clip_grad_norm_(backbone.parameters(), 5)
                    opt.step()
                    opt.zero_grad()
                    lr_scheduler.step()

            with torch.no_grad():
                loss_am.update(loss.item(), 1)
                callback_logging(global_step, loss_am, epoch, cfg.fp16, lr_scheduler.get_last_lr()[0], amp)

                # Validation
                if global_step % cfg.verbose == 0 and global_step > 0 and cfg.val_target:
                    backbone.eval()
                    d_prime = validation.ver_test(backbone, global_step, val_bin, cfg.embedding_size)

                    current_val = d_prime
                    if current_val <= best_val:
                        no_improve_counter += 1
                    else:
                        best_val = current_val
                        no_improve_counter = 0
                        path_module = os.path.join(cfg.output, f"model{f'_{cfg.strategy}' if cfg.strategy else ''}_{global_step}.pt")
                        torch.save(backbone.module.state_dict(), path_module)
                        print(f"Model saved with d prime score: {best_val}")

                    if no_improve_counter >= patience:
                        print("Early stopping triggered. Stopping training to prevent overfitting.")
                        break

                    backbone.train()

        if no_improve_counter >= patience:
            break

        if cfg.save_all_states:
            checkpoint = {
                "epoch": epoch + 1,
                "global_step": global_step,
                "state_dict_backbone": backbone.module.state_dict(),
                "state_dict_softmax_fc": module_partial_fc.state_dict(),
                "state_optimizer": opt.state_dict(),
                "state_lr_scheduler": lr_scheduler.state_dict()
            }
            torch.save(checkpoint, os.path.join(cfg.output, f"checkpoint.pt"))

        path_module = os.path.join(cfg.output, "model.pt")
        torch.save(backbone.module.state_dict(), path_module)

        if cfg.dali:
            train_dl.reset()

    if no_improve_counter < patience:
        path_module = os.path.join(cfg.output, "model_final.pt")
        torch.save(backbone.module.state_dict(), path_module)


if __name__ == "__main__":
    torch.backends.cudnn.benchmark = True
    parser = argparse.ArgumentParser(description="ArcIris")
    parser.add_argument("config", type=str, help="py config file")
    main(parser.parse_args())
