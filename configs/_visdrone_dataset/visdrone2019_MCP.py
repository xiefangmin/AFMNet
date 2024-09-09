# ========================================================
#   PROJECT_NAME ： AFM
#   FILE_NAME    ： visdrone2019.py 
#   USER         ： XIE 
#   DATE         ： 2024/3/22 - 9:11 
# ========================================================
import platform
import os




custom_imports = dict(imports=['AFM.transforms.Load_IR',
                               'AFM.transforms.RandomFlip_mask',], allow_failed_imports=False)

backend_args = None
img_scale = (1333, 800)


train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='CSD_IMG', enable=True, csd_hw=(800 // 8, 1344 // 8)),
    dict(type='Pad', size=img_scale),
    dict(
        type="Load_mask", dir_name='gt_mask_s/train',
        model="oval_mask",
        fg_bg_pixel=dict(fg=255, bg=0),
        binary_val=dict(fg=1., bg=0.),
        using_gaussianBlur=False,
        train_hw=(800, 1344),
        stride=(8,)
    ),
    dict(
        type="Load_mask", dir_name='gt_mask_l/train',
        model="oval_mask",
        fg_bg_pixel=dict(fg=255, bg=0),
        binary_val=dict(fg=1., bg=0.),
        using_gaussianBlur=False,
        train_hw=(800, 1344),
        stride=(16,)
    ),
    dict(type='RandomFlip_mask', direction="horizontal", prob=0.5),

    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
                    "gt_obj_mask", "gaussian_mask", 'csd_data',
                    )
         )
]

val_pipeline = [
    dict(type='LoadImageFromFile', backend_args=backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=img_scale, keep_ratio=True),
    dict(type='CSD_IMG', enable=True, csd_hw=(800 // 8, 1344 // 8)),
    dict(type='Load_IR',
         org_lab_root= fr'/home/swu/zmdt/datasets/VisDrone2019/labels/org/',
         suffix=".txt"),
    dict(type='Pad', size=img_scale),


    dict(type='PackDetInputs',
         meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape', 'scale_factor',
                    "gt_obj_mask", "gaussian_mask", 'csd_data', 'ignored_regions',
                    )
         )
]

# dataset settings
dataset_type = 'CocoDataset'
data_root = '/home/swu/zmdt/datasets/VisDrone2019_mcp'
classes = ('pedestrian', 'people', 'bicycle', 'car', 'van', 'truck', 'tricycle', 'awning-tricycle', 'bus', 'motor')

# ----------------------------------------------------------------------------------------------------------------------
train_dataloader = dict(
    batch_size=8,
    num_workers=4,
    persistent_workers=True,
    sampler=dict(type='DefaultSampler', shuffle=True),  # ,
    batch_sampler=dict(type='AspectRatioBatchSampler'),
    # Batch sampler for grouping images with similar aspect ratio into a same batch. It can reduce GPU memory cost.
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
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
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type=dataset_type,
        metainfo=dict(classes=classes),
        data_root=data_root,
        ann_file=os.path.join(*[data_root, 'labels', 'annotations', 'val_labels.json']),
        data_prefix=dict(img=os.path.join(*[data_root, 'images', 'val'])),
        test_mode=True,
        pipeline=val_pipeline,
        backend_args=backend_args
    )
)
test_dataloader = val_dataloader

# ----------------------------------------------------------------------------------------------------------------------
val_evaluator = dict(
    type='CocoMetric',
    ann_file=os.path.join(*[data_root, 'labels', 'annotations', 'val_labels.json']),
    metric=['bbox', ],
    proposal_nums=(10, 100, 500),  # 默认  (100, 300, 1000)
    format_only=False,
    backend_args=backend_args
)
