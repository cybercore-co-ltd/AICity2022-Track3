_base_ = ['../mmaction/_base_/default_runtime.py',
          '../mmaction/_base_/schedules/sgd_50e.py',
          '../_base_/datasets/aicity_A1_9rgb_224.py']
data_root = 'data/ai-city-2022/Track3/raw_frames/'
custom_imports = dict(imports=['ccaction'], allow_failed_imports=False)
load_from = 'http://118.69.233.170:60001/open/VidConvNext/convnext_vidconv_333_224_kinetics400_T1k/convnext_vidconv_333_224_kinetics400_T1k_epoch_24.pth'
model = dict(
    type='NoFogettingRecognizer',
    unforget_loss_weight = 50,
    off_model = dict(
        config='configs/aicity/vidconv_kinetics400_T1k.py',
        checkpoint=load_from,
    ),
    
    backbone=dict(
        type='ConvNextVidBaseTem',
        arch='tiny',
        drop_path_rate=0.25,
        init_cfg=None,
    ),

    neck=dict(type='Tem_Conv',
        in_channels=768,
        kernel_size=3,
        dilation=7,
        expand_ratio=0.25,
        norm_cfg=dict(type='SyncBN', requires_grad=True),),
    
    cls_head=dict(
        # cls_head is for distillation
        type='VidConvHead',
        in_channels=768,
        num_classes=400,
        spatial_type='avg',
        expand_ratio=3,
        kernel_size=3,
        dilation=7,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        dropout_ratio=0.2),

    tsp_head=dict(
        type='TSPHead',
        in_channels=192,
        action_label_head = dict(type='TSNHead', 
                        num_classes=17,
                        multi_class=True,
                        label_smooth_eps=0.2,
                        dropout_ratio=0.5),
        actioness_head = dict(type='TSNHead',
                        num_classes=2,
                        multi_class=True,
                        loss_cls=dict(type='CrossEntropyLoss',loss_weight=1),
                        label_smooth_eps=0.3,
                        dropout_ratio=0.5),
       ),

    test_cfg=dict(average_clips='prob'),
    # test_cfg=None,
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
                         backbone=dict(lr_mult=0.25),
                         cls_head=dict(lr_mult=0.05))))

optimizer_config = dict(_delete_=True, grad_clip=None)
lr_config = dict(_delete_=True,
                 policy='CosineAnnealing',
                 min_lr=5e-6,
                 by_epoch=True,
                 warmup='linear',
                 warmup_by_epoch=True,
                 warmup_iters=1)
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        times=3,
        dataset=dict(
            ann_file=data_root+'A1.txt',
            data_prefix=data_root+'A1',
        )),
    val=dict(
        ann_file=data_root+'A2.txt',
        data_prefix=data_root+'A2',
    ),
    test=dict(
        ann_file=data_root+'A2.txt',
        data_prefix=data_root+'A2',
    )
)
total_epochs = 8
find_unused_parameters = True
# fp16 = dict(loss_scale=512.0)
# runtime settings
# checkpoint_config = dict(interval=1, max_keep_ckpts=8)
checkpoint_config = dict(interval=1)
log_config = dict(interval=5, hooks=[dict(type='TextLoggerHook')])
evaluation = dict(interval=1, metrics='top_k_accuracy', topk=(1,2))
# load_from = 'http://118.69.233.170:60001/open/VidConvNext/convnext_vidconv_333_224_kinetics400_T1k/convnext_vidconv_333_224_kinetics400_T1k_epoch_24.pth'
resume_from = None
