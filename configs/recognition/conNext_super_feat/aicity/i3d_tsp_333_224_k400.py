_base_ = ['../../../mmaction/_base_/default_runtime.py',
          '../../../mmaction/_base_/schedules/sgd_50e.py',
          '../../../_base_/datasets/aicity_A1_9rgb_224.py',
          '../../../mmaction/_base_/models/i3d_r50.py']
custom_imports = dict(imports=['ccaction'], allow_failed_imports=False)
model = dict(
    type='TSPRecognizer',
    cls_head=dict(_delete_=True,
        type='TSPHead',
        in_channels=2048,
        kernel_size=3,
        dilation=7,
        expand_ratio=0.1,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        action_label_head = dict(type='TSNHead', 
                        num_classes=17,
                        dropout_ratio=0.5),
        actioness_head = dict(type='TSNHead',
                        num_classes=4,
                        dropout_ratio=0.5),
       ),
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
                         backbone=dict(lr_mult=0.25))))

optimizer_config = dict(_delete_=True, grad_clip=None)
lr_config = dict(_delete_=True,
                 policy='CosineAnnealing',
                 min_lr=5e-6,
                 by_epoch=True,
                 warmup='linear',
                 warmup_by_epoch=True,
                 warmup_iters=1)

total_epochs = 24
find_unused_parameters = True
fp16 = dict(loss_scale=512.0)
# runtime settings
checkpoint_config = dict(interval=1, max_keep_ckpts=5)
log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook')])
evaluation = dict(interval=1, metrics='top_k_accuracy')
resume_from = None
