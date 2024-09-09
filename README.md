

# Adaptive Fine-Grained Feature Mining and RoI Feature Interaction Network for Small Object Detection in Aerial Images



# This paper is submitted to the ICASSP 2025, and the source code of AFMNet will be released when it is accepted.



# Requirements

```
cuda 12.1

pytorch = 2.3.1
python = 3.9.19
cuda = 12.1
numpy = 1.26.4

mmcv==2.1.0
mmdet==3.3.0

```



# How to use?

1. Download the [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset)
2. Install mmdetection
3. Edit the data path
4. Train and Evaluation

### Core File
```

```

# Dataset

Catalogue structure of the [VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset)：

```
Datasets
└─── images
|	├─── train
|	└─── val
|
└─── labels
	└─── annotations  
	|	├─── train_labels.json
	|	└─── val_labels.json
	└─── org	 # official label
	|	├─── train_annotations  
	|	└─── val_annotations    
	└─── gt_mask_s  
	|	└─── train  
	└─── gt_mask_l  
		└─── train  

```annotation
The official .txt annotation format is: <bbox_left>,<bbox_top>,<bbox_width>,<bbox_height>,<score>,<object_category>,<truncation>,<occlusion>.
We provide the VisDrone Dataset annotations (COCO Format) and our ground truth gaussian mask labelas follows:
[Google Drive](https://drive.google.com/file/d/1HYMeZmjT3-yW7PFpIzJtr84Sc8JqIrLG/view?usp=sharing), [Baidu Drive](https://pan.baidu.com/s/1kAsSlg8QHvD83E-3SMrZZg ) `passwd:ki77`.


# Train

To train AFMNet, run the training script below.

```
python ./train.py \
${CONFIG_FILE} \
[optional arguments]
```

For instance：

```
python ./train.py configs/AFM/AFMNet-tood_r50_k13.py
```

# Evaluation

    toolkit: https://github.com/VisDrone/VisDrone2018-DET-toolkit   https://github.com/cocodataset/cocoapi.git
    
    1.  python Inference.py
    
    2.  get_fuse_ret.py

  

# Reference
[1] Kai Chen, Jiaqi Wang, Jiangmiao Pang, Yuhang Cao, Yu Xiong, Xiaoxiao Li, Shuyang Sun, Wansen Feng, Ziwei Liu, Jiarui Xu, et al., “Mmdetection: Open mmlab detection toolbox and benchmark,” arXiv preprint arXiv:1906.07155, 2019.

[2] Pengfei Zhu, Longyin Wen, Dawei Du, Xiao Bian, Heng Fan, Qinghua Hu, and Haibin Ling, “Detection and tracking meet drones challenge,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 11, pp. 7380–7399, 2021.
