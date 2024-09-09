_base_ = [
    '../tood/tood_r50_fpn_1x_coco_carafe.py',
    '../_visdrone_dataset/visdrone2019_MCP.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]
import os

os.environ['cur_proj_name'] = "AFMNet-tood_r50_k13"

custom_imports = dict(imports=['AFM.AFM_DET.AFM_OS_HD_F',
                               'AFM.AFM_MODEL.MASK_EST.MASK_EST'],
                      allow_failed_imports=False)

# model settings
model = dict(
    type='AFM_OS_HD_F',  # TOOD
    bbox_head=dict(
        num_classes=10,
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128]
        ),
    ),


    # fpd
    FPD=dict(
        with_fpd=True,
        type='MASK_EST',
        in_channels=256,
        out_channels=1,
        model_num=2,
    ),

    HD_head=dict(
        type='TOODHead',
        with_affm=True,
        with_irfi=True,
        affm_topk=13,
        num_classes=10,
        in_channels=256,
        stacked_convs=6,
        feat_channels=256,
        anchor_type='anchor_free',
        anchor_generator=dict(
            type='AnchorGenerator',
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[4, ]
        ),

        bbox_coder=dict(
            type='DeltaXYWHBBoxCoder',
            target_means=[.0, .0, .0, .0],
            target_stds=[0.1, 0.1, 0.2, 0.2]),
        initial_loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            activated=True,  # use probability instead of logit as input
            gamma=2.0,
            alpha=0.25,
            loss_weight=1.0),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=True,
            activated=True,  # use probability instead of logit as input
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)),


    train_cfg=dict(
        initial_epoch=4,
        initial_assigner=dict(type='ATSSAssigner', topk=4),
        assigner=dict(type='TaskAlignedAssigner', topk=3),
        alpha=1,
        beta=7,
        allowed_border=-1,
        pos_weight=-1,
        debug=False
    ),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.01,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=500),

    test_cfg_HD=dict(
        nms_pre=1000,
        min_bbox_size=3,  # (w > cfg.min_bbox_size) & (h > cfg.min_bbox_size)
        score_thr=0.01,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=500)
)

default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='coco/bbox_mAP_50'),  # , max_keep_ckpts=1
    logger=dict(type='LoggerHook', interval=100),
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=36, val_interval=1)
train_dataloader = dict(batch_size=4, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=1)
test_dataloader = dict(batch_size=1, num_workers=1)

load_from = r"configs/tood/tood_r50_fpn_mstrain_2x_coco.pth"
resume_from = None
