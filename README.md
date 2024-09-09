# AFMNet
## Adaptive Fine-Grained Feature Mining and RoI Feature Interaction Network for Small Object Detection in Aerial Images

This paper is submitted to the ICASSP 2025, and the source code of AFMNet will be released when it is accepted.

# AFMNet
cuda 12.1

conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

conda create --name zmdt python=3.8 -y

pip install -r requirements.txt

pip install -U openmim

mim install "mmengine==0.10.3"

mim install "mmcv==2.1.0"

mim install "mmdet==3.3.0"

https://github.com/xiefangmin/AFMNet

## Dataset

https://github.com/VisDrone/VisDrone-Dataset

project_root/dataset/VisDrone2019
                                 /images
                                        /train
                                        /val
                                 /labels
                                        /annotations  
                                            -train_labels.json
                                            -val_labels.json
                                        /org          # official label
                                            /train_annotations  
                                            /val_annotations  
                                        /gt_mask_s  # for fpd
                                            /train
                                        /gt_mask_l  # for fpd
                                            /train



# Train

python ./train.py configs_AHRS/AFM/AFMNet-tood_r50_k13.py

# Evaluation
    toolkit: https://github.com/VisDrone/VisDrone2018-DET-toolkit   https://github.com/cocodataset/cocoapi.git
    
    1.  python Inference.py
    
    2.  get_fuse_ret.py
      
  
