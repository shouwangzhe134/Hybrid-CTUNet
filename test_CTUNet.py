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

import argparse
import os

import nibabel as nib
import numpy as np
import torch
from networks.hybrid_CTUNet import CTUNet
from trainer_CUNet import dice, resample_3d
from utils.data_utils import get_loader

from trainer_CTUNet import sliding_window_inference as sliding_window_inference_multi

from monai import data, transforms

parser = argparse.ArgumentParser(description="UNETR segmentation pipeline")
parser.add_argument(
    "--pretrained_dir", default="./pretrained_models/", type=str, help="pretrained checkpoint directory"
)
parser.add_argument("--data_dir", default="./dataset/dataset0/", type=str, help="dataset directory")
parser.add_argument("--exp_name", default="test1", type=str, help="experiment name")
parser.add_argument("--json_list", default="dataset_0.json", type=str, help="dataset json file")
parser.add_argument(
    "--pretrained_model_name", default="UNETR_model_best_acc.pth", type=str, help="pretrained model name"
)
parser.add_argument(
    "--saved_checkpoint", default="ckpt", type=str, help="Supports torchscript or ckpt pretrained checkpoint type"
)
parser.add_argument("--mlp_dim", default=3072, type=int, help="mlp dimention in ViT encoder")
parser.add_argument("--hidden_size", default=768, type=int, help="hidden size dimention in ViT encoder")
parser.add_argument("--feature_size", default=64, type=int, help="feature size dimention")
parser.add_argument("--in_channels", default=1, type=int, help="number of input channels")
parser.add_argument("--out_channels", default=14, type=int, help="number of output channels")
parser.add_argument("--num_heads", default=12, type=int, help="number of attention heads in ViT encoder")
parser.add_argument("--res_block", action="store_true", help="use residual blocks")
parser.add_argument("--bottleneck_block", action="store_true", help="use bottleneck blocks")
parser.add_argument("--conv_block", action="store_true", help="use conv blocks")
parser.add_argument("--a_min", default=-175.0, type=float, help="a_min in ScaleIntensityRanged")
parser.add_argument("--a_max", default=250.0, type=float, help="a_max in ScaleIntensityRanged")
parser.add_argument("--b_min", default=0.0, type=float, help="b_min in ScaleIntensityRanged")
parser.add_argument("--b_max", default=1.0, type=float, help="b_max in ScaleIntensityRanged")
parser.add_argument("--space_x", default=1.5, type=float, help="spacing in x direction")
parser.add_argument("--space_y", default=1.5, type=float, help="spacing in y direction")
parser.add_argument("--space_z", default=2.0, type=float, help="spacing in z direction")
parser.add_argument("--roi_x", default=96, type=int, help="roi size in x direction")
parser.add_argument("--roi_y", default=96, type=int, help="roi size in y direction")
parser.add_argument("--roi_z", default=96, type=int, help="roi size in z direction")
parser.add_argument("--distributed", action="store_true", help="start distributed training")
parser.add_argument("--workers", default=8, type=int, help="number of workers")
parser.add_argument("--RandFlipd_prob", default=0.2, type=float, help="RandFlipd aug probability")
parser.add_argument("--RandRotate90d_prob", default=0.2, type=float, help="RandRotate90d aug probability")
parser.add_argument("--RandScaleIntensityd_prob", default=0.1, type=float, help="RandScaleIntensityd aug probability")
parser.add_argument("--RandShiftIntensityd_prob", default=0.1, type=float, help="RandShiftIntensityd aug probability")
parser.add_argument("--pos_embed", default="perceptron", type=str, help="type of position embedding")
parser.add_argument("--norm_name", default="instance", type=str, help="normalization layer type in decoder")
parser.add_argument("--num_depths", default=12, type=int, help="number of depths in ViT")
# to ajust parameters 
parser.add_argument("--infer_overlap", default=0.5, type=float, help="sliding window inference overlap")
parser.add_argument("--dropout_rate", default=0.0, type=float, help="dropout rate")
parser.add_argument("--model_name", default="ctunet", type=str, help="model name")
parser.add_argument("--model_depths", default=50, type=int, help="resnet model depth")
parser.add_argument("--patch_frame", default=16, type=int, help="patch frame")

def process_label(label): 
    spleen = label == 1
    right_kidney = label == 2
    left_kidney = label == 3
    gallbladder = label == 4
    esophagus = label == 5
    liver = label == 6
    stomach = label == 7
    aorta = label == 8
    inferior_veana_cava = label == 9 
    portal_vein_splenic_vein = label == 10
    pancreas = label == 11
    right_adrenal_gland = label == 12
    left_adrenal_gland = label == 13
    return (spleen, right_kidney, left_kidney, gallbladder, esophagus, liver, stomach, aorta, inferior_veana_cava, portal_vein_splenic_vein, pancreas, right_adrenal_gland, left_adrenal_gland)

def main():
    args = parser.parse_args()
    args.test_mode = True
    args.test_challenge = False
    output_directory = "./output/" + args.exp_name
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    invert_transform, val_loader = get_loader(args)
    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pretrained_pth = os.path.join(pretrained_dir, model_name)
    if args.saved_checkpoint == "torchscript":
        model = torch.jit.load(pretrained_pth)
    elif args.saved_checkpoint == "ckpt":
        if args.model_name == "ctunet":
            model1 = CTUNet(
            in_channels=args.in_channels,
            dim_conv_stem=args.feature_size, 
            out_channels=args.out_channels, 
            model_depth=args.model_depths, 
            img_size=(args.roi_x, args.roi_y), 
            frames = args.roi_z, 
            patch_frame = args.patch_frame,
            hidden_size=args.hidden_size, 
            num_depths = args.num_depths, 
            mlp_dim=args.mlp_dim, 
            num_heads=args.num_heads, 
            norm_name=args.norm_name, 
            dropout_rate=args.dropout_rate, 
        )
            model2 = CTUNet(
            in_channels=args.in_channels,
            dim_conv_stem=args.feature_size, 
            out_channels=args.out_channels, 
            model_depth=args.model_depths, 
            img_size=(args.roi_x, args.roi_y), 
            frames = args.roi_z, 
            patch_frame = args.patch_frame,
            hidden_size=args.hidden_size, 
            num_depths = args.num_depths, 
            mlp_dim=args.mlp_dim, 
            num_heads=args.num_heads, 
            norm_name=args.norm_name, 
            dropout_rate=args.dropout_rate, 
        )
            model3 = CTUNet(
            in_channels=args.in_channels,
            dim_conv_stem=args.feature_size, 
            out_channels=args.out_channels, 
            model_depth=args.model_depths, 
            img_size=(args.roi_x, args.roi_y), 
            frames = args.roi_z, 
            patch_frame = args.patch_frame,
            hidden_size=args.hidden_size, 
            num_depths = args.num_depths, 
            mlp_dim=args.mlp_dim, 
            num_heads=args.num_heads, 
            norm_name=args.norm_name, 
            dropout_rate=args.dropout_rate, 
        )
        model1.load_state_dict(torch.load(os.path.join(pretrained_dir, "model_res.pt"))["state_dict"])
        model2.load_state_dict(torch.load(os.path.join(pretrained_dir, "model_vit.pt"))["state_dict"])
        model3.load_state_dict(torch.load(os.path.join(pretrained_dir, "model_hybrid.pt"))["state_dict"])
    model1.eval()
    model1.to(device)
    model2.eval()
    model2.to(device)
    model3.eval()
    model3.to(device)

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
        dice_list_case_res = []
        dice_list_case_vit = []
        dice_list_case = []

        Dice_spleen = []
        Dice_right_kidney = [] 
        Dice_left_kidney = [] 
        Dice_gallbladder = [] 
        Dice_esophagus = [] 
        Dice_liver = [] 
        Dice_stomach = [] 
        Dice_aorta = [] 
        Dice_inferior_veana_cava = [] 
        Dice_portal_vein_splenic_vein = [] 
        Dice_pancreas = [] 
        Dice_right_adrenal_gland = [] 
        Dice_left_adrenal_gland = [] 
        fw = open(output_directory + '/dice.txt', 'a') 

        for i, batch in enumerate(val_loader):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            original_affine = batch["label_meta_dict"]["affine"][0].numpy()

            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            print("Inference on case {}".format(img_name))
            # val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model, overlap=args.infer_overlap, mode="constant")
            val_outputs1 = sliding_window_inference_multi(val_inputs, (args.roi_x, args.roi_y, args.roi_z), 4, model1, overlap=args.infer_overlap, mode="gaussian")[0]
            val_outputs2 = sliding_window_inference_multi(val_inputs, (args.roi_x, args.roi_y, args.roi_z), 4, model2, overlap=args.infer_overlap, mode="gaussian")[1]
            batch["pred1"] = val_outputs1 
            batch["pred2"] = val_outputs2 
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

            dice_list_sub_res = []
            dice_list_sub_vit = []
            dice_list_sub = []
            for i in range(1, 14):
                dice_list_sub_res.append(dice(infers1 == i, labels == i))
                dice_list_sub_vit.append(dice(infers2 == i, labels == i))
                dice_list_sub.append(dice(infers == i, labels == i))

            dice_list_case_res.append(dice_list_sub_res)
            dice_list_case_vit.append(dice_list_sub_vit)
            dice_list_case.append(dice_list_sub)

            nib.save(
                nib.Nifti1Image(infers.astype(np.uint8), original_affine), os.path.join(output_directory, img_name)
            )

            label = process_label(labels)
            infer = process_label(infers)

            Dice_spleen.append(dice(label[0], infer[0]))
            Dice_right_kidney.append(dice(label[1], infer[1]))
            Dice_left_kidney.append(dice(label[2], infer[2]))
            Dice_gallbladder.append(dice(label[3], infer[3]))
            Dice_esophagus.append(dice(label[4], infer[4]))
            Dice_liver.append(dice(label[5], infer[5]))
            Dice_stomach.append(dice(label[6], infer[6]))
            Dice_aorta.append(dice(label[7], infer[7]))
            Dice_inferior_veana_cava.append(dice(label[8], infer[8]))
            Dice_portal_vein_splenic_vein.append(dice(label[9], infer[9]))
            Dice_pancreas.append(dice(label[10], infer[10]))
            Dice_right_adrenal_gland.append(dice(label[11], infer[11]))
            Dice_left_adrenal_gland.append(dice(label[12], infer[12]))

            fw.write('*' * 20 + '\n', )
            fw.write('case: ' + img_name + '\n')
            fw.write('Dice_spleen: {:.4f}\n'.format(Dice_spleen[-1]))
            fw.write('Dice_right_kidney: {:.4f}\n'.format(Dice_right_kidney[-1]))
            fw.write('Dice_left_kidney: {:.4f}\n'.format(Dice_left_kidney[-1]))
            fw.write('Dice_gallbladder: {:.4f}\n'.format(Dice_gallbladder[-1]))
            fw.write('Dice_esophagus: {:.4f}\n'.format(Dice_esophagus[-1]))
            fw.write('Dice_liver: {:.4f}\n'.format(Dice_liver[-1]))
            fw.write('Dice_stomach: {:.4f}\n'.format(Dice_stomach[-1]))
            fw.write('Dice_aorta: {:.4f}\n'.format(Dice_aorta[-1]))
            fw.write('Dice_inferior_veana_cava: {:.4f}\n'.format(Dice_inferior_veana_cava[-1]))
            fw.write('Dice_portal_vein_splenic_vein: {:.4f}\n'.format(Dice_portal_vein_splenic_vein[-1]))
            fw.write('Dice_pancreas: {:.4f}\n'.format(Dice_pancreas[-1]))
            fw.write('Dice_right_adrenal_gland: {:.4f}\n'.format(Dice_right_adrenal_gland[-1]))
            fw.write('Dice_left_adrenal_gland: {:.4f}\n'.format(Dice_left_adrenal_gland[-1]))
        
        fw.write('*' * 20 + '\n', ) 
        fw.write('Mean_Dice\n')
        fw.write('Dice_spleen' + str(np.mean(Dice_spleen)) + '\n') 
        fw.write('Dice_right_kidney' + str(np.mean(Dice_right_kidney)) + '\n')
        fw.write('Dice_left_kidney' + str(np.mean(Dice_left_kidney)) + '\n')
        fw.write('Dice_gallbladder' + str(np.mean(Dice_gallbladder)) + '\n')
        fw.write('Dice_esophagus' + str(np.mean(Dice_esophagus)) + '\n')
        fw.write('Dice_liver' + str(np.mean(Dice_liver)) + '\n')
        fw.write('Dice_stomach' + str(np.mean(Dice_stomach)) + '\n')
        fw.write('Dice_aorta' + str(np.mean(Dice_aorta)) + '\n')
        fw.write('Dice_inferior_veana_cava' + str(np.mean(Dice_inferior_veana_cava)) + '\n')
        fw.write('Dice_portal_vein_splenic_vein' + str(np.mean(Dice_portal_vein_splenic_vein)) + '\n')
        fw.write('Dice_pancreas' + str(np.mean(Dice_pancreas)) + '\n')
        fw.write('Dice_right_adrenal_gland' + str(np.mean(Dice_right_adrenal_gland)) + '\n')
        fw.write('Dice_left_adrenal_gland' + str(np.mean(Dice_left_adrenal_gland)) + '\n')

        fw.write('*'*20+'\n')

        dsc = []
        dsc.append(np.mean(Dice_spleen)) 
        dsc.append(np.mean(Dice_right_kidney))
        dsc.append(np.mean(Dice_left_kidney))
        dsc.append(np.mean(Dice_gallbladder))
        dsc.append(np.mean(Dice_esophagus))
        dsc.append(np.mean(Dice_liver))
        dsc.append(np.mean(Dice_stomach))
        dsc.append(np.mean(Dice_aorta))
        dsc.append(np.mean(Dice_inferior_veana_cava))
        dsc.append(np.mean(Dice_portal_vein_splenic_vein))
        dsc.append(np.mean(Dice_pancreas))
        dsc.append(np.mean(Dice_right_adrenal_gland))
        dsc.append(np.mean(Dice_left_adrenal_gland))
        fw.write('dsc:' + str(np.mean(dsc)) + '\n')

        fw.close()

        mean_dice_res = np.mean(dice_list_case_res, 0) 
        mean_dice_vit = np.mean(dice_list_case_vit, 0) 
        mean_dice = np.mean(dice_list_case, 0)
        print("Overall Mean Organ Dice Based on ResNet: {}".format(np.round(mean_dice_res, 4)))
        print("Overall Mean Dice Based on ResNet: {}".format(np.mean(mean_dice_res)))
        print("Overall Mean Organ Dice Based on ViT: {}".format(np.round(mean_dice_vit, 4)))
        print("Overall Mean Dice Based on ViT: {}".format(np.mean(mean_dice_vit)))
        print("Overall Mean Organ Dice: {}".format(np.round(mean_dice, 4)))
        print("Overall Mean Dice: {}".format(np.mean(mean_dice)))

    with torch.no_grad():
        dice_list_case_res = []
        dice_list_case_vit = []
        dice_list_case = []
        for i, batch in enumerate(val_loader):
            val_inputs, val_labels = (batch["image"].cuda(), batch["label"].cuda())
            original_affine = batch["label_meta_dict"]["affine"][0].numpy()

            img_name = batch["image_meta_dict"]["filename_or_obj"][0].split("/")[-1]
            print("Inference on case {}".format(img_name))
            # val_outputs = sliding_window_inference(val_inputs, (96, 96, 96), 4, model, overlap=args.infer_overlap, mode="constant")
            val_outputs = sliding_window_inference_multi(val_inputs, (args.roi_x, args.roi_y, args.roi_z), 4, model3, overlap=args.infer_overlap, mode="gaussian")
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

            dice_list_sub_res = []
            dice_list_sub_vit = []
            dice_list_sub = []
            for i in range(1, 14):
                dice_list_sub_res.append(dice(infers1==i, labels == i))
                dice_list_sub_vit.append(dice(infers2==i, labels == i))
                dice_list_sub.append(dice(infers==i, labels == i))

            dice_list_case_res.append(dice_list_sub_res)
            dice_list_case_vit.append(dice_list_sub_vit)
            dice_list_case.append(dice_list_sub)

            #nib.save(
            #    nib.Nifti1Image(infers.astype(np.uint8), original_affine), os.path.join(output_directory, img_name)
            #)

        mean_dice_res = np.mean(dice_list_case_res, 0) 
        mean_dice_vit = np.mean(dice_list_case_vit, 0) 
        mean_dice = np.mean(dice_list_case, 0)
        print("Overall Mean Organ Dice Based on ResNet: {}".format(np.round(mean_dice_res, 4)))
        print("Overall Mean Dice Based on ResNet: {}".format(np.mean(mean_dice_res)))
        print("Overall Mean Organ Dice Based on ViT: {}".format(np.round(mean_dice_vit, 4)))
        print("Overall Mean Dice Based on ViT: {}".format(np.mean(mean_dice_vit)))
        print("Overall Mean Organ Dice: {}".format(np.round(mean_dice, 4)))
        print("Overall Mean Dice: {}".format(np.mean(mean_dice)))

if __name__ == "__main__":
    main()
