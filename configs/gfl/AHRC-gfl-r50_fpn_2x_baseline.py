# ========================================================
#   PROJECT_NAME ： ahrc_mmdet
#   FILE_NAME    ： AHRC-gfl-r50_fpn_2x_baseline.py
#   USER         ： XIE
#   DATE         ： 2024/3/22 - 13:43
# ========================================================
_base_ = [
    './gfl_r50_fpn_1x_coco.py',
    '../_visdrone_dataset/visdrone2019.py',  # visdrone2019.py
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]


model = dict(
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(num_classes=10),
    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.01,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=500)
)


default_hooks = dict(  # 设置保存间隔 保存最佳模型只保留最新的几个模型
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='auto') # , max_keep_ckpts=10
)

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=2)
train_dataloader = dict(batch_size=8, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=1)
test_dataloader = dict(batch_size=1, num_workers=1)

# work_dir = 'work_dir'
# load_from = r"H:\code3\ahrc-mmdet\configs_AHRC\gfl\gfl_r50_fpn_mstrain_2x_coco_20200629_213802-37bb1edc.pth"  # 从给定路径将模型加载为预训练模型,
load_from=None
resume_from = None  # 从给定路径恢复检查点，训练将从保存检查点的纪元开始恢复
