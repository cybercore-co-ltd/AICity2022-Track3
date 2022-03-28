_base_ = ['../../../mmaction/_base_/default_runtime.py',
          '../../../mmaction/_base_/schedules/sgd_50e.py',
          '../../../_base_/datasets/aicity_A1_9rgb_224_video.py']
custom_imports = dict(imports=['ccaction'], allow_failed_imports=False)
model = dict(
    type='VidConvRecognizer',
    backbone=dict(
        type='ConvNextVidBaseTem',
        arch='tiny',
        drop_path_rate=0.4,
        init_cfg=dict(type='Pretrained', checkpoint="tiny_1k")
    ),
    cls_head=dict(
        type='MultiviewVidConvHead',
        in_channels=768,
        num_classes=18,
        spatial_type='avg',
        expand_ratio=0.25,
        kernel_size=3,
        dilation=7,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        dropout_ratio=0.5),
    test_cfg=dict(average_clips='prob'),
    train_cfg=None,
)

#----------- AdamW
optimizer = dict(_delete_=True,
                 type='AdamW',
                 lr=0.0008,
                 betas=(0.9, 0.999),
                 weight_decay=0.0005,
                 paramwise_cfg=dict(
                     custom_keys=dict(
                         norm=dict(decay_mult=0.0),
                         backbone=dict(lr_mult=0.25))))

optimizer_config = dict(_delete_=True, grad_clip=None)
lr_config = dict(_delete_=True,
                 policy='CosineAnnealing',
                 min_lr=5e-6,
                 by_epoch=True,
                 warmup='linear',
                 warmup_by_epoch=True,
                 warmup_iters=1)

total_epochs = 25
# find_unused_parameters = True
fp16 = dict(loss_scale=512.0)
# runtime settings
checkpoint_config = dict(interval=5)
log_config = dict(interval=10, hooks=[dict(type='TextLoggerHook')])
evaluation = dict(interval=5, metrics='top_k_accuracy')
load_from = '/home/cybercore/vinh_overal/AICity2022-Track3/best_e10.pth'
resume_from = None
