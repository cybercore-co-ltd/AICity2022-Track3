_base_ = ['../../../mmaction/_base_/default_runtime.py',
          '../../../mmaction/_base_/schedules/sgd_50e.py',
          '../../../_base_/datasets/aicity_A1_9rgb_224.py']
custom_imports = dict(imports=['ccaction'], allow_failed_imports=False)
model = dict(
    type='NoFogettingRecognizer',
    off_model = dict(
        config='/home/ccvn/Workspace/suhuynh/AICity2022-Track3/configs/recognition/conNext_super_feat/Kinetics400/convnext_vidconv_333_224_kinetics400_T1k.py',
        checkpoint='http://118.69.233.170:60001/open/VidConvNext/convnext_vidconv_333_224_kinetics400_T1k/convnext_vidconv_333_224_kinetics400_T1k_epoch_24.pth'
    ),
    backbone=dict(
        type='ConvNextVidBaseTem',
        arch='tiny',
        drop_path_rate=0.25,
        init_cfg=dict(type='Pretrained', checkpoint="tiny_1k")
    ),
    
    kl_head=dict(
        type='VidConvHead',
        in_channels=768,
        num_classes=400,
        spatial_type='avg',
        expand_ratio=3,
        kernel_size=3,
        dilation=7,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        dropout_ratio=0.2),

    cls_head=dict(
        type='TSPHead',
        in_channels=768,
        kernel_size=3,
        dilation=7,
        expand_ratio=0.25,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        action_label_head = dict(type='TSNHead', 
                        num_classes=17,
                        multi_class=True,
                        label_smooth_eps=0.2,
                        dropout_ratio=0.3),
        actioness_head = dict(type='TSNHead',
                        num_classes=4,
                        multi_class=True,
                        label_smooth_eps=0.3,
                        dropout_ratio=0.3),
       ),

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
# fp16 = dict(loss_scale=512.0)
# runtime settings
checkpoint_config = dict(interval=1, max_keep_ckpts=5)
log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook')])
evaluation = dict(interval=1, metrics='top_k_accuracy')
load_from = 'http://118.69.233.170:60001/open/VidConvNext/convnext_vidconv_333_224_kinetics400_T1k/convnext_vidconv_333_224_kinetics400_T1k_epoch_24.pth'
resume_from = None
