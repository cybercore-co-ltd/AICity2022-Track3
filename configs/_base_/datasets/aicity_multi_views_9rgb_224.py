dataset_type = 'TSP_Multiviews_RawframeDataset'
data_root = '/ssd3/data/ai-city-2022/Track3/raw_frames/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
image_size = 224
train_pipeline = [
    dict(type='RandSampleFrames', clip_len=9,
         range=(12,18), num_clips=5),
    dict(type='RawFrameDecode_multiviews'),
    # dict(type='Resize', scale=(-1, 256)),
    # dict(
    #     type='MultiScaleCrop',
    #     input_size=224,
    #     scales=(1, 0.875, 0.75, 0.66),
    #     random_crop=False,
    #     max_wh_scale_gap=1,
    #     num_fixed_crops=13),
    dict(type='Resize', scale=(image_size, image_size), keep_ratio=False),
    # dict(type='Flip', flip_ratio=0.5),
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
        num_clips=5, 
        test_mode=True),
    dict(type='RawFrameDecode_multiviews'),
    # dict(type='Resize', scale=(-1, 256)),
    # dict(type='CenterCrop', crop_size=image_size),
    # dict(type='ThreeCrop', crop_size=224),
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
        num_clips=5,
        test_mode=True),
    dict(type='RawFrameDecode_multiviews'),
    # dict(type='Resize', scale=(-1, 256)),
    # dict(type='CenterCrop', crop_size=image_size),
    dict(type='Resize', scale=(image_size, image_size), keep_ratio=False),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

data = dict(
    videos_per_gpu=3,
    workers_per_gpu=0,
    train=dict(
        type=dataset_type,
        ann_file=data_root+'combine_A1A2.txt',
        data_prefix=data_root+'combine_A1A2',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root+'multiviews_A2.txt',
        data_prefix=data_root+'A2',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root+'multiviews_A2.txt',
        data_prefix=data_root+'A2',
        pipeline=val_pipeline))