# training schedule for 1x
train_cfg = dict(
    type='EpochBasedTrainLoop',  # 训练循环的类型，请参考 https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py
    max_epochs=12,  # 最大训练轮次
    val_interval=1)  # 验证间隔。每 2 个 epoch 验证一次
val_cfg = dict(type='ValLoop')  # 验证循环的类型
test_cfg = dict(type='TestLoop')  # 测试循环的类型

# learning rate
param_scheduler = [ # Linear learning rate warm-up scheduler
    dict(
        type='LinearLR',  # 使用线性学习率预热
        start_factor=0.001, # 学习率预热的系数
        by_epoch=False,  # 按 iteration 更新预热学习率
        begin=0,  # 从第一个 iteration 开始
        end=500),  # 到第 500 个 iteration 结束
    dict(
        type='MultiStepLR',  # 在训练过程中使用 multi step 学习率策略
        by_epoch=True,  # 按 epoch 更新学习率
        begin=0,   # 从第一个 epoch 开始
        end=12,  # 到第 12 个 epoch 结束
        milestones=[8, 11],  # 在哪几个 epoch 进行学习率衰减
        gamma=0.1)  # 学习率衰减系数
]

# optimizer
optim_wrapper = dict(  # 优化器封装的配置
    type='OptimWrapper',  # 优化器封装的类型。可以切换至 AmpOptimWrapper 来启用混合精度训练
    optimizer=dict(  # 优化器配置。支持 PyTorch 的各种优化器。请参考 https://pytorch.org/docs/stable/optim.html#algorithms
        type='SGD',  # 随机梯度下降优化器
        lr=0.02,  # 基础学习率
        momentum=0.9,  # 带动量的随机梯度下降
        weight_decay=0.0001),  # 权重衰减
    clip_grad=None,  # 梯度裁剪的配置，设置为 None 关闭梯度裁剪。使用方法请见 https://mmengine.readthedocs.io/en/latest/tutorials/optimizer.html
    )

# Default setting for scaling LR automatically
#   - `enable` means enable scaling LR automatically
#       or not by default.
#   - `base_batch_size` = (8 GPUs) x (2 samples per GPU).
auto_scale_lr = dict(enable=False, base_batch_size=16)
