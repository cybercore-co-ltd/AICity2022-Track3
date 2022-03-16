_base_ = ['../../../mmaction/_base_/default_runtime.py',
          '../../../mmaction/_base_/schedules/sgd_50e.py',
          '../../../_base_/datasets/kinetics400_9rgb_224.py']
custom_imports = dict(imports=['ccaction'], allow_failed_imports=False)
model = dict(
    type='VidConvRecognizer',
    backbone=dict(
        type='ConvNextVidBaseTem',
        arch='small',
        drop_path_rate=0.4,
        init_cfg=dict(type='Pretrained', checkpoint="small_1k")
    ),
    cls_head=dict(
        type='VidConvHead',
        in_channels=768,
        num_classes=400,
        spatial_type='avg',
        expand_ratio=3,
        kernel_size=3,
        dilation=7,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        dropout_ratio=0.2),
    test_cfg=dict(average_clips='prob'),
    train_cfg=None,
)

#----------- AdamW
optimizer = dict(_delete_=True,
                 type='AdamW',
                 lr=0.001,
                 betas=(0.9, 0.999),
                 weight_decay=0.0005,
                 paramwise_cfg=dict(
                     custom_keys=dict(
                         norm=dict(decay_mult=0.0),
                         backbone=dict(lr_mult=0.1))))

optimizer_config = dict(_delete_=True, grad_clip=None)
lr_config = dict(_delete_=True,
                 policy='CosineAnnealing',
                 min_lr=5e-6,
                 by_epoch=True,
                 warmup='linear',
                 warmup_by_epoch=True,
                 warmup_iters=1)

total_epochs = 24
# find_unused_parameters = True
fp16 = dict(loss_scale=512.0)
# runtime settings
checkpoint_config = dict(interval=1, max_keep_ckpts=5)
log_config = dict(interval=50, hooks=[dict(type='TextLoggerHook')])
evaluation = dict(interval=1, metrics='top_k_accuracy')
load_from = None
resume_from = None
