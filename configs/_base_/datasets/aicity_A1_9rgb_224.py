dataset_type = 'TSP_RawframeDataset'
data_root = 'data/raw_frames/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
image_size = 224
train_pipeline = [
    dict(type='RandSampleFrames', clip_len=9,
         range=(12,18), num_clips=2),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(image_size, image_size), keep_ratio=False),
    dict(type='Imgaug', transforms='default'),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='RandomErasing', probability=0.2),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
] 
val_pipeline = [
    dict(
        type='AdaptSampleFrames',
        clip_len=9,
        frame_interval=15,
        num_clips=4, 
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(image_size, image_size), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

test_pipeline = [
    dict(
        type='AdaptSampleFrames',
        clip_len=9,
        frame_interval=15,
        num_clips=4,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(image_size, image_size), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

data = dict(
    videos_per_gpu=8,
    workers_per_gpu=4,
    train=dict(
        type='RepeatDataset',
        times=2,
        dataset=dict(
            type=dataset_type,
            ann_file=data_root+'A1.txt',
            data_prefix=data_root+'A1',
            pipeline=train_pipeline),
        ),
    val=dict(
        type=dataset_type,
        ann_file=data_root+'A2.txt',
        data_prefix=data_root+'A2',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root+'A2.txt',
        data_prefix=data_root+'A2',
        pipeline=val_pipeline))