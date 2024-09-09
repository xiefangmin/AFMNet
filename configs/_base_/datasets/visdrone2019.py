# ========================================================
#   PROJECT_NAME ： leb_mmdet
#   FILE_NAME    ： visdrone2019.py 
#   USER         ： XIE 
#   DATE         ： 2024/3/22 - 9:11 
# ========================================================
import platform
import os

backend_args = None
albu_train_transforms = [
    dict(type='ShiftScaleRotate', shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, interpolation=1, p=0.5),
    dict(type='RandomBrightnessContrast', brightness_limit=[0.1, 0.3], contrast_limit=[0.1, 0.3], p=0.2),
    dict(type='OneOf', transforms=[
        dict(type='RGBShift', r_shift_limit=10, g_shift_limit=10, b_shift_limit=10, p=1.0),
        dict(type='HueSaturationValue', hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0)],
         p=0.1),
    # dict(type='JpegCompression', quality_lower=85, quality_upper=95, p=0.2),
    dict(type='ChannelShuffle', p=0.1),
    dict(type='OneOf',
         transforms=[dict(type='Blur', blur_limit=3, p=1.0), dict(type='MedianBlur', blur_limit=3, p=1.0)],
         p=0.1),
]

img_scale = (640, 640)
train_pipeline = [  # 在 mmengine\dataset\base_dataset.py  data = t(data)
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Albu', transforms=albu_train_transforms,
         bbox_params=dict(
             type='BboxParams',
             format='pascal_voc',
             label_fields=['gt_labels'],
             min_visibility=0.0,
             filter_lost_elements=True),
         keymap={'img': 'image', 'gt_bboxes': 'bboxes'},  # 'gt_masks': 'masks',
         skip_img_without_anno=True),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
         )

]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    # dict(type='PackDetInputs', meta_keys=['img', 'gt_bboxes', 'gt_labels']),
    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor')
         )
]

# dataset settings
dataset_type = 'CocoDataset'
data_root = 'D:\Adataset2\VisDrone2019' if platform.system() == 'Windows' else '/home/swu/zmdt/datasets/VisDrone2019'
classes = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')

# ----------------------------------------------------------------------------------------------------------------------
train_dataloader = dict(  # 对应 2.x 版本中的 data.train
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),  # , seed=655
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    # Batch sampler for grouping images with similar aspect ratio into a same batch. It can reduce GPU memory cost.
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),  # 还可以 'palette' :[(156,45,13),]
        data_root=data_root,
        ann_file=os.path.join(*[data_root, 'labels', 'annotations', 'train_labels.json']),
        data_prefix=dict(img=os.path.join(*[data_root, 'images', 'train'])),
        filter_cfg=dict(filter_empty_gt=True, min_size=32),
        pipeline=train_pipeline,
        backend_args=backend_args
    ),

)
val_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),  # , seed=655
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file=os.path.join(*[data_root, 'labels', 'annotations', 'val_labels.json']),
        data_prefix=dict(img=os.path.join(*[data_root, 'images', 'val'])),
        test_mode=True,
        pipeline=test_pipeline,
        backend_args=backend_args
    )
)
test_dataloader = val_dataloader

# ----------------------------------------------------------------------------------------------------------------------
val_evaluator = dict(
    type='CocoMetric',
    ann_file=os.path.join(*[data_root, 'labels', 'annotations', 'val_labels.json']),
    metric=['bbox', ],
    format_only=False,
    backend_args=backend_args
)
test_evaluator = val_evaluator
