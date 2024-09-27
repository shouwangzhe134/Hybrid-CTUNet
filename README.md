# Hybrid-CTUNet

- This code is an official implementation of "Hybrid-CTUNet: A Double Complementation Approach for 3D Medical Image Segmentation" based on the open source medical segmentation toolbox [MONAI](https://github.com/Project-MONAI/research-contributions/tree/main).
- 
## Introduction
Medical segmentation is a fundamental problem in medical image computing, and it finds wide application in clinical domains such as medical diagnosis and robotic surgery. In this work, we investigate the distinct spatial characteristics of CNNs and Transformers in their representations of local and global features, while also analyzing the differences in preservation of spatial position within their network structures. They provide a comprehensive explanation of the comple-mentarity between CNN and Transformer. To promote effective complementarity, we propose two novel architectures, namely CUNet and TUNet, which individu-ally preserve the spatial characteristics throughout the overall U-Net process of the encoder and decoder. For feature complementation, we incorporate CUNet and TUNet as parallel branches, named CTUNet, which enhances the long-range dependencies of global information in both the deep and shallow locality. More-over, we design the binary cross-weights for element-wise addition to achieve a more prominent fusion of features with diverse spatial characteristics. For further mask complementation, we construct a Hybrid-CTUNet by integrating the jointly training CTUNet and the independently training TUNet. Extensive empirical analysis conducted on medical datasets confirms the superiority of our proposed method compared to state-of-the-art models.

## Installation 

### Requirements 

- Linux (Windows is not officially supported)
- python 3.9+
- Pytorch 1.12.1
- CUDA 11.3

### Installation Procedure

a. Create a conda virtual environment and activate it.
```shell
conda create -n hybrid-CTUNet python=3.9.18 -y
conda activate hybrid-CTUNet 
```

b. To avoid the problem that could not load library libcudnn_cnn_infer.so.8. Error, install the corresponding cuda version in the conda environment. (optional)
```shell
conda install nvidia/label/cuda-11.3.0::cuda
```

c. Install torch.
```shell
pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
```
or
```shell
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch
```

d. Install MONAI and the recommended dependencies.
```shell
pip install -r requirements.txt
pip install 'monai[all]'
```

e. Reconfirm some python package versions. 
```shell
pip uninstall protobuf
pip install protobuf==3.20.0

pip install scipy==1.11.1

pip install numpy==1.22.4
pip install medpy==0.4.0 
```

## Train and Inference 

#### Train with a single GPU

```shell
CUDA_VISIBLE_DEVICES=3 python main_C_TUNet.py --logdir=CUNet_ds8 --model_name="cunet" --save_checkpoint --model_depths=101
CUDA_VISIBLE_DEVICES=3 python main_C_TUNet.py --logdir=TUNet_pf8 --model_name="tunet" --save_checkpoint --dropout_rate=0.0 --patch_frame=8
CUDA_VISIBLE_DEVICES=3 python main_CTUNet.py --logdir=CTUNet_ds8 --save_checkpoint --model_name="ctunet" --model_depths=101 --dropout_rate=0.0 --patch_frame=8
```

##### Train with multiple GPUs

```shell
CUDA_VISIBLE_DEVICES=3,4 python main_C_TUNet.py --logdir=CUNet_ds8 --model_name="cunet" --save_checkpoint --model_depths=101 --distributed
CUDA_VISIBLE_DEVICES=3,4 python main_C_TUNet.py --logdir=TUNet_pf8 --model_name="tunet" --save_checkpoint --dropout_rate=0.0 --patch_frame=8 --distributed
CUDA_VISIBLE_DEVICES=3,4 python main_CTUNet.py --logdir=CTUNet_ds8 --save_checkpoint --model_name="ctunet" --model_depths=101 --dropout_rate=0.0 --patch_frame=8 --distributed
```

#### Test

```shell
CUDA_VISIBLE_DEVICES=4 python test_C_TUNet.py --pretrained_dir='./runs/CUNet_ds8/' --pretrained_model_name='model_res.pt' --model_name="cunet" --model_depths=101
CUDA_VISIBLE_DEVICES=4 python test_C_TUNet.py --pretrained_dir='./runs/TUNet_pf8/' --pretrained_model_name='model_vit.pt' --model_name="tunet" --model_depths=101
CUDA_VISIBLE_DEVICES=4 python test_CTUNet.py --pretrained_dir='./runs/CTUNet_ds8/' --model_name="ctunet" --model_depths=101 --patch_frame=8 --exp_name="CTUNet_ds8"
```


#### Tensorboard

```shell
tensorboard --logdir="" --bind_all --port=6007
```

## Acknowledgement
Many thanks to the open source codes, i.e., [MONAI](https://github.com/Project-MONAI/research-contributions)
