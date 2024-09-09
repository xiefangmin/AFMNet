# ========================================================
#   PROJECT_NAME ： ahrc-mmdet
#   FILE_NAME    ： AHRC-atss_r50_fpn_2x_baseline.py 
#   USER         ： XIE 
#   DATE         ： 2024/4/18 - 9:14 
# ========================================================
_base_ = [
    './atss_r50_fpn_1x_coco.py',
    '../_visdrone_dataset/visdrone2019.py',
    '../_base_/schedules/schedule_2x.py',
    '../_base_/default_runtime.py'
]

model = dict(
    bbox_head=dict(
        num_classes=10,
    ),

    # training and testing settings
    train_cfg=dict(
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    test_cfg=dict(
        nms_pre=1000,
        min_bbox_size=0,
        score_thr=0.25,
        nms=dict(type='nms', iou_threshold=0.6),
        max_per_img=500)
)

# 设置保存间隔 保存最佳模型只保留最新的几个模型
# , max_keep_ckpts=10   coco/bbox_mAP   coco/bbox_mAP_50
default_hooks = dict(
    checkpoint=dict(type='CheckpointHook', interval=1, save_best='coco/bbox_mAP_50', max_keep_ckpts=1),
)
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=24, val_interval=1)
train_dataloader = dict(batch_size=8, num_workers=4)
val_dataloader = dict(batch_size=1, num_workers=1)
test_dataloader = dict(batch_size=1, num_workers=1)

load_from = r"configs_AHRC/atss/atss_r50_fpn_1x_coco_20200209-985f7bd0.pth"  # 从给定路径将模型加载为预训练模型,
# load_from=None
resume_from = None
# optim_wrapper = dict(  # 优化器封装的配置
#     clip_grad=dict(max_norm=1, norm_type=2),  # 梯度裁剪的配置，设置为 None 关闭梯度裁剪。使用方法请见 https://mmengine.readthedocs.io/en/latest/tutorials/optimizer.html
#     )