_base_ = 'convnext_noforgetting_tsp_333_224_A1.py'
data_root = '/ssd3/data/ai-city-2022/Track3/raw_frames/'
custom_imports = dict(imports=['ccaction'], allow_failed_imports=False)


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
            ann_file=data_root+'combine_A1A2Pseudo_vidconv_round2_bg.txt',
            data_prefix=data_root+'combine_A1A2Pseudo_vidconv_round2_bg',
        )),
)
total_epochs = 8
find_unused_parameters = True

