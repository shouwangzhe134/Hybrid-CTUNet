# Copyright 2020 - 2021 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import shutil
import time

import numpy as np
import torch
import torch.nn.parallel
import torch.utils.data.distributed
from tensorboardX import SummaryWriter
from torch.cuda.amp import GradScaler, autocast
from utils.utils import distributed_all_gather

from monai.data import decollate_batch
import scipy.ndimage as ndimage
from monai import transforms

# from monai.inferers import sliding_window_inference
from trainer_CUNet import sliding_window_inference

def resample_3d(img, target_size):
    imx, imy, imz = img.shape
    tx, ty, tz = target_size
    zoom_ratio = (float(tx) / float(imx), float(ty) / float(imy), float(tz) / float(imz))
    img_resampled = ndimage.zoom(img, zoom_ratio, order=0, prefilter=False)
    return img_resampled

def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def train_epoch(model, loader, optimizer, scaler, epoch, loss_func, args):
    model.train()
    start_time = time.time()
    run_loss = AverageMeter()
    run_loss1 = AverageMeter() 
    run_loss2 = AverageMeter()
    for idx, batch_data in enumerate(loader):
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["image"], batch_data["label"]
        data, target = data.cuda(args.rank), target.cuda(args.rank)
        for param in model.parameters():
            param.grad = None
        with autocast(enabled=args.amp):
            logits = model(data)
            loss1 = loss_func(logits[0], target)
            loss2 = loss_func(logits[1], target)
            loss = loss1 + loss2
        if args.amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        if args.distributed:
            loss_list = distributed_all_gather([loss], out_numpy=True, is_valid=idx < loader.sampler.valid_length)
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0), n=args.batch_size * args.world_size
            )
        else:
            run_loss.update(loss.item(), n=args.batch_size)
            run_loss1.update(loss1.item(), n=args.batch_size) 
            run_loss2.update(loss2.item(), n=args.batch_size)
        if args.rank == 0:
            print(
                "Epoch {}/{} {}/{}".format(epoch, args.max_epochs, idx, len(loader)),
                "loss: {:.4f}".format(run_loss.avg),
                "loss1: {:.4f}".format(run_loss1.avg),
                "loss2: {:.4f}".format(run_loss2.avg),
                "time {:.2f}s".format(time.time() - start_time),
            )
        start_time = time.time()
    for param in model.parameters():
        param.grad = None
    return run_loss.avg

def val_epoch(model, val_loader, invert_transform, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    post_transform = transforms.Compose(
        [
            # transforms.Activationsd(keys="pred", softmax=True),
            transforms.Invertd(
                keys="pred",  # invert the `pred` data field, also support multiple fields
                transform=invert_transform,
                orig_keys="image",  # get the previously applied pre_transforms information on the `img` data field,
                # then invert `pred` based on this information. we can use same info
                # for multiple fields, also support different orig_keys for different fields
                nearest_interp=False,  # don't change the interpolation mode to "nearest" when inverting transforms
                # to ensure a smooth output, then execute `AsDiscreted` transform
                to_tensor=True,  # convert to PyTorch Tensor after inverting
            ),
            # transforms.AsDiscreted(keys="pred", logit_thresh=0.5),
            transforms.Activationsd(keys="pred", softmax=True),
            transforms.AsDiscreted(keys="pred", argmax=True, to_onehot=True, num_classes=14),
            # transforms.SaveImaged(keys="pred", output_dir="./out", output_postfix="seg", resample=False),
        ]
    )

    with torch.no_grad():
        dice_list_case = []
        for i, batch in enumerate(val_loader):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())

            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            print("Inference on case {}".format(img_name))
            val_outputs = sliding_window_inference(
                val_inputs, (args.roi_x, args.roi_y, args.roi_z), 4, model, overlap=args.infer_overlap, mode="gaussian"
            )
            batch["pred"] = val_outputs 
            batch = [post_transform(i) for i in decollate_batch(batch)]
            print("Done") 

            val_labels = val_labels.cpu().numpy()[0, 0, :, :, :]
            val_infers = batch[0]["pred"].cpu().numpy()
            dice_list_sub = []
            for i in range(1, 14):
                organ_Dice = dice(val_infers[i, :, :, :], val_labels == i)
                dice_list_sub.append(organ_Dice)

            dice_list_case.append(dice_list_sub)
        mean_dice = np.mean(dice_list_case, 0)
        print("Overall Mean Organ Dice: {}".format(np.round(mean_dice, 4)))
        print("Overall Mean Dice: {}".format(np.mean(mean_dice)))

    return np.mean(mean_dice)

def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.logdir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)


def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    args,
    scheduler=None,
    start_epoch=0,
    transform=None,
):
    writer = None
    if args.logdir is not None and args.rank == 0:
        writer = SummaryWriter(log_dir=args.logdir)
        if args.rank == 0:
            print("Writing Tensorboard logs to ", args.logdir)
    scaler = None
    if args.amp:
        scaler = GradScaler()
    val_acc_max = 0.0

    if args.model_name == "cunet":
        filename = "model_res.pt"
    else:
        filename = "model_vit.pt"

    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        print(args.rank, time.ctime(), "Epoch:", epoch)
        epoch_time = time.time()
        train_loss = train_epoch(
            model, train_loader, optimizer, scaler=scaler, epoch=epoch, loss_func=loss_func, args=args
        )
        if args.rank == 0:
            print(
                "Final training  {}/{}".format(epoch, args.max_epochs - 1),
                "loss: {:.4f}".format(train_loss),
                "time {:.2f}s".format(time.time() - epoch_time),
            )
        if args.rank == 0 and writer is not None:
            writer.add_scalar("train_loss", train_loss, epoch)
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            # val_avg_acc = val_epoch(model, val_loader, args)
            val_avg_acc = val_epoch(model, val_loader, transform, args)
            if args.rank == 0:
                print(
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    "acc",
                    val_avg_acc,
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                if writer is not None:
                    writer.add_scalar("val_acc", val_avg_acc, epoch)
                if val_avg_acc > val_acc_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_max, val_avg_acc))
                    val_acc_max = val_avg_acc
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, filename=filename, best_acc=val_acc_max, optimizer=optimizer, scheduler=scheduler
                        )

        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best Accuracy: ", val_acc_max)

    return val_acc_max
