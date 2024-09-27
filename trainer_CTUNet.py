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

# from monai.inferers import sliding_window_inference

from typing import Any, Callable, List, Sequence, Tuple, Union

import torch
import torch.nn.functional as F

from monai.data.utils import compute_importance_map, dense_patch_slices, get_valid_patch_size
from monai.utils import BlendMode, PytorchPadMode, fall_back_tuple, look_up_option

from monai.engines import EnsembleEvaluator, SupervisedEvaluator, SupervisedTrainer

from monai import transforms, data

__all__ = ["sliding_window_inference"]

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
            loss1_1 = loss_func(logits[0][0], target)
            target1 = torch.from_numpy(ndimage.zoom(target.cpu().numpy(), (1,1,0.5,0.5,1), order=0, prefilter=False)).cuda()
            target2 = torch.from_numpy(ndimage.zoom(target.cpu().numpy(), (1,1,0.25,0.25,0.5), order=0, prefilter=False)).cuda()
            # print(target.shape, target1.shape, target2.shape)
            loss1_2 = loss_func(logits[0][1], target1)
            loss1_3 = loss_func(logits[0][2], target2)
            loss1 = loss1_1 + 0.5*(loss1_2 + 0.5*loss1_3)

            loss2_1 = loss_func(logits[1][0], target)
            loss2_2 = loss_func(logits[1][1], target)
            loss2 = (loss2_1 + loss2_2)
            loss = loss1 + 0.5*loss2
            # loss = 0.5 * loss1 + loss2 
            # loss = loss1 + loss2
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

def val_epoch_single(model, val_loader, invert_transform, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    # Invert transforms for original label size.
    post_transform1 = transforms.Compose(
        [
            # transforms.Activationsd(keys="pred", softmax=True),
            transforms.Invertd(
                keys="pred1",  # invert the `pred` data field, also support multiple fields
                transform=invert_transform,
                orig_keys="image",  # get the previously applied pre_transforms information on the `img` data field,
                # then invert `pred` based on this information. we can use same info
                # for multiple fields, also support different orig_keys for different fields
                nearest_interp=False,  # don't change the interpolation mode to "nearest" when inverting transforms
                # to ensure a smooth output, then execute `AsDiscreted` transform
                to_tensor=True,  # convert to PyTorch Tensor after inverting
            ),
            # transforms.AsDiscreted(keys="pred", logit_thresh=0.5),
            # transforms.Activationsd(keys="pred", softmax=True),
            # transforms.AsDiscreted(keys="pred", argmax=True, to_onehot=True, num_classes=14),
            # transforms.SaveImaged(keys="pred", output_dir="./out", output_postfix="seg", resample=False),
        ]
    )
    post_transform2 = transforms.Compose(
        [
            # transforms.Activationsd(keys="pred", softmax=True),
            transforms.Invertd(
                keys="pred2",  # invert the `pred` data field, also support multiple fields
                transform=invert_transform,
                orig_keys="image",  # get the previously applied pre_transforms information on the `img` data field,
                # then invert `pred` based on this information. we can use same info
                # for multiple fields, also support different orig_keys for different fields
                nearest_interp=False,  # don't change the interpolation mode to "nearest" when inverting transforms
                # to ensure a smooth output, then execute `AsDiscreted` transform
                to_tensor=True,  # convert to PyTorch Tensor after inverting
            ),
            # transforms.AsDiscreted(keys="pred", logit_thresh=0.5),
            # transforms.Activationsd(keys="pred", softmax=True),
            # transforms.AsDiscreted(keys="pred", argmax=True, to_onehot=True, num_classes=14),
            # transforms.SaveImaged(keys="pred", output_dir="./out", output_postfix="seg", resample=False),
        ]
    )

    with torch.no_grad():
        dice_list_case1 = []
        dice_list_case2 = []
        for i, batch in enumerate(val_loader):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())

            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            print("Inference on case {}".format(img_name))
            val_outputs = sliding_window_inference(
                val_inputs, (args.roi_x, args.roi_y, args.roi_z), 4, model, overlap=args.infer_overlap, mode="gaussian"
            )
            batch["pred1"] = val_outputs[0]
            batch["pred2"] = val_outputs[1]
            batch = [post_transform1(i) for i in data.decollate_batch(batch)] 
            batch = [post_transform2(i) for i in batch]
            print("Done")

            infers1 = torch.softmax(batch[0]["pred1"], 0) 
            infers2 = torch.softmax(batch[0]["pred2"], 0)
            infers1 = torch.argmax(infers1, axis=0).cpu().numpy()
            infers2 = torch.argmax(infers2, axis=0).cpu().numpy()

            labels = val_labels.cpu().numpy()[0, 0, :, :, :]

            dice_list_sub1 = []
            dice_list_sub2 = []
            for i in range(1, 14):
                organ_Dice1 = dice(infers1==i, labels == i)
                dice_list_sub1.append(organ_Dice1)
                organ_Dice2 = dice(infers2==i, labels == i)
                dice_list_sub2.append(organ_Dice2)

            dice_list_case1.append(dice_list_sub1)
            dice_list_case2.append(dice_list_sub2)

        mean_dice1 = np.mean(dice_list_case1, 0)
        mean_dice2 = np.mean(dice_list_case2, 0)

        print("Overall Mean Organ Dice: {}".format(np.round(mean_dice1, 4)))
        print("Overall Mean Dice: {}".format(np.mean(mean_dice1)))

        print("Overall Mean Organ Dice: {}".format(np.round(mean_dice2, 4)))
        print("Overall Mean Dice: {}".format(np.mean(mean_dice2)))

    return np.mean(mean_dice1), np.mean(mean_dice2)

def val_epoch_hybrid(model, val_loader, invert_transform, args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    model.to(device)

    # Invert transforms for original label size.
    post_transform1 = transforms.Compose(
        [
            # transforms.Activationsd(keys="pred", softmax=True),
            transforms.Invertd(
                keys="pred1",  # invert the `pred` data field, also support multiple fields
                transform=invert_transform,
                orig_keys="image",  # get the previously applied pre_transforms information on the `img` data field,
                # then invert `pred` based on this information. we can use same info
                # for multiple fields, also support different orig_keys for different fields
                nearest_interp=False,  # don't change the interpolation mode to "nearest" when inverting transforms
                # to ensure a smooth output, then execute `AsDiscreted` transform
                to_tensor=True,  # convert to PyTorch Tensor after inverting
            ),
            # transforms.AsDiscreted(keys="pred", logit_thresh=0.5),
            # transforms.Activationsd(keys="pred", softmax=True),
            # transforms.AsDiscreted(keys="pred", argmax=True, to_onehot=True, num_classes=14),
            # transforms.SaveImaged(keys="pred", output_dir="./out", output_postfix="seg", resample=False),
        ]
    )
    post_transform2 = transforms.Compose(
        [
            # transforms.Activationsd(keys="pred", softmax=True),
            transforms.Invertd(
                keys="pred2",  # invert the `pred` data field, also support multiple fields
                transform=invert_transform,
                orig_keys="image",  # get the previously applied pre_transforms information on the `img` data field,
                # then invert `pred` based on this information. we can use same info
                # for multiple fields, also support different orig_keys for different fields
                nearest_interp=False,  # don't change the interpolation mode to "nearest" when inverting transforms
                # to ensure a smooth output, then execute `AsDiscreted` transform
                to_tensor=True,  # convert to PyTorch Tensor after inverting
            ),
            # transforms.AsDiscreted(keys="pred", logit_thresh=0.5),
            # transforms.Activationsd(keys="pred", softmax=True),
            # transforms.AsDiscreted(keys="pred", argmax=True, to_onehot=True, num_classes=14),
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
            batch["pred1"] = val_outputs[0]
            batch["pred2"] = val_outputs[1]
            batch = [post_transform1(i) for i in data.decollate_batch(batch)] 
            batch = [post_transform2(i) for i in batch]
            print("Done")
            
            infers1 = torch.softmax(batch[0]["pred1"], 0) 
            infers2 = torch.softmax(batch[0]["pred2"], 0)
            infers = (infers1 + infers2) / 2.0
            infers1 = torch.argmax(infers1, axis=0).cpu().numpy()
            infers2 = torch.argmax(infers2, axis=0).cpu().numpy()
            infers = torch.argmax(infers, axis=0).cpu().numpy()

            labels = val_labels.cpu().numpy()[0, 0, :, :, :]

            dice_list_sub = []
            for i in range(1, 14):
                organ_Dice = dice(infers==i, labels == i)
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
    val_acc_hybrid_max = 0.0
    val_acc_res_max = 0.0
    val_acc_vit_max = 0.0

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
        b_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            epoch_time = time.time()
            val_avg_acc_hybrid = val_epoch_hybrid(model, val_loader, transform, args) 
            val_avg_acc_res, val_avg_acc_vit = val_epoch_single(model, val_loader, transform, args) 
            if args.rank == 0:
                print(
                    "Final validation  {}/{}".format(epoch, args.max_epochs - 1),
                    "acc_hybrid",
                    val_avg_acc_hybrid,
                    "acc_res",
                    val_avg_acc_res,
                    "acc_vit", 
                    val_avg_acc_vit,
                    "time {:.2f}s".format(time.time() - epoch_time),
                )
                if writer is not None:
                    writer.add_scalar("val_acc_hybrid", val_avg_acc_hybrid, epoch)
                    writer.add_scalar("val_acc_res", val_avg_acc_res, epoch) 
                    writer.add_scalar("val_acc_vit", val_avg_acc_vit, epoch)
                if val_avg_acc_hybrid > val_acc_hybrid_max:
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_hybrid_max, val_avg_acc_hybrid))
                    val_acc_hybrid_max = val_avg_acc_hybrid
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint:
                        save_checkpoint(
                            model, epoch, args, filename="model_hybrid.pt",  best_acc=val_acc_hybrid_max, optimizer=optimizer, scheduler=scheduler
                        )
                if val_avg_acc_res > val_acc_res_max: 
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_res_max, val_avg_acc_res))
                    val_acc_res_max = val_avg_acc_res
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint: 
                        save_checkpoint(
                            model, epoch, args, filename="model_res.pt",  best_acc=val_acc_res_max, optimizer=optimizer, scheduler=scheduler
                        )
                if val_avg_acc_vit > val_acc_vit_max: 
                    print("new best ({:.6f} --> {:.6f}). ".format(val_acc_vit_max, val_avg_acc_vit))
                    val_acc_vit_max = val_avg_acc_vit
                    b_new_best = True
                    if args.rank == 0 and args.logdir is not None and args.save_checkpoint: 
                        save_checkpoint(
                            model, epoch, args, filename="model_vit.pt",  best_acc=val_acc_vit_max, optimizer=optimizer, scheduler=scheduler
                        )

        if scheduler is not None:
            scheduler.step()

    print("Training Finished !, Best ResNet Accuracy: ", val_acc_res_max)
    print("Training Finished !, Best ViT Accuracy: ", val_acc_vit_max)
    print("Training Finished !, Best Accuracy: ", val_acc_hybrid_max)

    return val_acc_hybrid_max


def sliding_window_inference(
    inputs: torch.Tensor,
    roi_size: Union[Sequence[int], int],
    sw_batch_size: int,
    predictor: Callable[..., torch.Tensor],
    overlap: float = 0.25,
    mode: Union[BlendMode, str] = BlendMode.CONSTANT,
    sigma_scale: Union[Sequence[float], float] = 0.125,
    padding_mode: Union[PytorchPadMode, str] = PytorchPadMode.CONSTANT,
    cval: float = 0.0,
    sw_device: Union[torch.device, str, None] = None,
    device: Union[torch.device, str, None] = None,
    *args: Any,
    **kwargs: Any,
) -> torch.Tensor:
    """
    Sliding window inference on `inputs` with `predictor`.

    When roi_size is larger than the inputs' spatial size, the input image are padded during inference.
    To maintain the same spatial sizes, the output image will be cropped to the original input size.

    Args:
        inputs: input image to be processed (assuming NCHW[D])
        roi_size: the spatial window size for inferences.
            When its components have None or non-positives, the corresponding inputs dimension will be used.
            if the components of the `roi_size` are non-positive values, the transform will use the
            corresponding components of img size. For example, `roi_size=(32, -1)` will be adapted
            to `(32, 64)` if the second spatial dimension size of img is `64`.
        sw_batch_size: the batch size to run window slices.
        predictor: given input tensor `patch_data` in shape NCHW[D], `predictor(patch_data)`
            should return a prediction with the same spatial shape and batch_size, i.e. NMHW[D];
            where HW[D] represents the patch spatial size, M is the number of output channels, N is `sw_batch_size`.
        overlap: Amount of overlap between scans.
        mode: {``"constant"``, ``"gaussian"``}
            How to blend output of overlapping windows. Defaults to ``"constant"``.

            - ``"constant``": gives equal weight to all predictions.
            - ``"gaussian``": gives less weight to predictions on edges of windows.

        sigma_scale: the standard deviation coefficient of the Gaussian window when `mode` is ``"gaussian"``.
            Default: 0.125. Actual window sigma is ``sigma_scale`` * ``dim_size``.
            When sigma_scale is a sequence of floats, the values denote sigma_scale at the corresponding
            spatial dimensions.
        padding_mode: {``"constant"``, ``"reflect"``, ``"replicate"``, ``"circular"``}
            Padding mode for ``inputs``, when ``roi_size`` is larger than inputs. Defaults to ``"constant"``
            See also: https://pytorch.org/docs/stable/nn.functional.html#pad
        cval: fill value for 'constant' padding mode. Default: 0
        sw_device: device for the window data.
            By default the device (and accordingly the memory) of the `inputs` is used.
            Normally `sw_device` should be consistent with the device where `predictor` is defined.
        device: device for the stitched output prediction.
            By default the device (and accordingly the memory) of the `inputs` is used. If for example
            set to device=torch.device('cpu') the gpu memory consumption is less and independent of the
            `inputs` and `roi_size`. Output is on the `device`.
        args: optional args to be passed to ``predictor``.
        kwargs: optional keyword args to be passed to ``predictor``.

    Note:
        - input must be channel-first and have a batch dim, supports N-D sliding window.

    """
    num_spatial_dims = len(inputs.shape) - 2
    if overlap < 0 or overlap >= 1:
        raise AssertionError("overlap must be >= 0 and < 1.")

    # determine image spatial size and batch size
    # Note: all input images must have the same image size and batch size
    image_size_ = list(inputs.shape[2:])
    batch_size = inputs.shape[0]

    if device is None:
        device = inputs.device
    if sw_device is None:
        sw_device = inputs.device

    roi_size = fall_back_tuple(roi_size, image_size_)
    # in case that image size is smaller than roi size
    image_size = tuple(max(image_size_[i], roi_size[i]) for i in range(num_spatial_dims))
    pad_size = []
    for k in range(len(inputs.shape) - 1, 1, -1):
        diff = max(roi_size[k - 2] - inputs.shape[k], 0)
        half = diff // 2
        pad_size.extend([half, diff - half])
    inputs = F.pad(inputs, pad=pad_size, mode=look_up_option(padding_mode, PytorchPadMode).value, value=cval)

    scan_interval = _get_scan_interval(image_size, roi_size, num_spatial_dims, overlap)

    # Store all slices in list
    slices = dense_patch_slices(image_size, roi_size, scan_interval)
    num_win = len(slices)  # number of windows per image
    total_slices = num_win * batch_size  # total number of windows

    # Create window-level importance map
    importance_map = compute_importance_map(
        get_valid_patch_size(image_size, roi_size), mode=mode, sigma_scale=sigma_scale, device=device
    )

    # Perform predictions
    output_image1, count_map1 = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
    output_image2, count_map2 = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
    _initialized = False
    for slice_g in range(0, total_slices, sw_batch_size):
        slice_range = range(slice_g, min(slice_g + sw_batch_size, total_slices))
        unravel_slice = [
            [slice(int(idx / num_win), int(idx / num_win) + 1), slice(None)] + list(slices[idx % num_win])
            for idx in slice_range
        ]
        window_data = torch.cat([inputs[win_slice] for win_slice in unravel_slice]).to(sw_device)
        #seg_prob = predictor(window_data, *args, **kwargs).to(device)  # batched patch segmentation
        seg_prob = predictor(window_data, *args, **kwargs)  # batched patch segmentation
        seg_prob1 = seg_prob[0][0].to(device) 
        seg_prob2 = seg_prob[1][0].to(device)

        if not _initialized:  # init. buffer at the first iteration
            output_classes = seg_prob1.shape[1]
            output_shape = [batch_size, output_classes] + list(image_size)
            # allocate memory to store the full output and the count for overlapping parts
            output_image1 = torch.zeros(output_shape, dtype=torch.float32, device=device)
            count_map1 = torch.zeros(output_shape, dtype=torch.float32, device=device)
            output_image2 = torch.zeros(output_shape, dtype=torch.float32, device=device)
            count_map2 = torch.zeros(output_shape, dtype=torch.float32, device=device)
            _initialized = True

        # store the result in the proper location of the full output. Apply weights from importance map.
        for idx, original_idx in zip(slice_range, unravel_slice):
            output_image1[original_idx] += importance_map * seg_prob1[idx - slice_g]
            count_map1[original_idx] += importance_map
            output_image2[original_idx] += importance_map * seg_prob2[idx - slice_g]
            count_map2[original_idx] += importance_map

    # account for any overlapping sections
    output_image1 = output_image1 / count_map1
    output_image2 = output_image2 / count_map2

    final_slicing: List[slice] = []
    for sp in range(num_spatial_dims):
        slice_dim = slice(pad_size[sp * 2], image_size_[num_spatial_dims - sp - 1] + pad_size[sp * 2])
        final_slicing.insert(0, slice_dim)
    while len(final_slicing) < len(output_image1.shape):
        final_slicing.insert(0, slice(None))
    return (output_image1[final_slicing], output_image2[final_slicing])


def _get_scan_interval(
    image_size: Sequence[int], roi_size: Sequence[int], num_spatial_dims: int, overlap: float
) -> Tuple[int, ...]:
    """
    Compute scan interval according to the image size, roi size and overlap.
    Scan interval will be `int((1 - overlap) * roi_size)`, if interval is 0,
    use 1 instead to make sure sliding window works.

    """
    if len(image_size) != num_spatial_dims:
        raise ValueError("image coord different from spatial dims.")
    if len(roi_size) != num_spatial_dims:
        raise ValueError("roi coord different from spatial dims.")

    scan_interval = []
    for i in range(num_spatial_dims):
        if roi_size[i] == image_size[i]:
            scan_interval.append(int(roi_size[i]))
        else:
            interval = int(roi_size[i] * (1 - overlap))
            scan_interval.append(interval if interval > 0 else 1)
    return tuple(scan_interval)
