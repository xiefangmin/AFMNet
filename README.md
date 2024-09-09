

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
2. Download our ground truth gaussian mask label ( [Baidu Drive](https://pan.baidu.com/s/1kAsSlg8QHvD83E-3SMrZZg ) `passwd:ki77`)
3. Install mmdetection
4. Edit the data path
5. Train and Evaluation



# Dataset

[VisDrone Dataset](https://github.com/VisDrone/VisDrone-Dataset)

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


```



# Train

To train AFMNet, run the training script below.

```
python ./train.py \
${CONFIG_FILE} \
[optional arguments]
```

For instance,

```
python ./train.py configs/AFM/AFMNet-tood_r50_k13.py
```

# Evaluation

    toolkit: https://github.com/VisDrone/VisDrone2018-DET-toolkit   https://github.com/cocodataset/cocoapi.git
    
    1.  python Inference.py
    
    2.  get_fuse_ret.py

  

