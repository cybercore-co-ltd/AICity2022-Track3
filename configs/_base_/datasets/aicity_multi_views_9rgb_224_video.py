dataset_type = 'VideoDataset'
data_root = '/ssd3/data/ai-city-2022/Track3/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
image_size = 224
train_pipeline = [
    dict(type='CcDecordInit'),
    dict(type='RandSampleFrames', clip_len=9,
         range=(12, 18), num_clips=5),
    dict(type='CcDecordDecode', crop_drive=True),
    dict(type='Resize', scale=(image_size, image_size), keep_ratio=False),
    dict(type='Imgaug', transforms='default'),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='RandomErasing', probability=0.2),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(type='CcDecordInit'),
    dict(
        type='SampleFrames',
        clip_len=9,
        frame_interval=15,
        num_clips=5,
        test_mode=True),
    dict(type='CcDecordDecode', crop_drive=True),
    dict(type='Resize', scale=(image_size, image_size), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

data = dict(
    videos_per_gpu=4,
    workers_per_gpu=2,
    train=dict(
        type='RepeatDataset',
        times=3,  # Repeat times
        dataset=dict(
            type=dataset_type,
            ann_file=data_root+'csv_file/dashboard_train_without_bg.csv',
            data_prefix=data_root+'train_video',
            pipeline=train_pipeline
        )
    ),
    val=dict(
        type=dataset_type,
        ann_file=data_root+'csv_file/dashboard_val_without_bg.csv',
        data_prefix=data_root+'val_video',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root+'csv_file/dashboard_val_without_bg.csv',
        data_prefix=data_root+'val_video',
        pipeline=val_pipeline))
