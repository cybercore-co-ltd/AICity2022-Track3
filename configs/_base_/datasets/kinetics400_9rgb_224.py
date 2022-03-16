dataset_type = 'RawframeDataset'
data_root = '/ssd3/data/kinetics400/'
# data_root = '/raid/data/kinetics400/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
image_size = 224
train_pipeline = [
    dict(type='RandSampleFrames', clip_len=9,
         range=(6,12), num_clips=1),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(
        type='MultiScaleCrop',
        input_size=224,
        scales=(1, 0.875, 0.75, 0.66),
        random_crop=False,
        max_wh_scale_gap=1,
        num_fixed_crops=13),
    dict(type='Resize', scale=(image_size, image_size), keep_ratio=False),
    dict(type='Flip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
] 
val_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=9,
        frame_interval=8,
        num_clips=1, 
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=image_size),
    # dict(type='ThreeCrop', crop_size=224),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

test_pipeline = [
    dict(
        type='SampleFrames',
        clip_len=9,
        frame_interval=8,
        num_clips=1,
        test_mode=True),
    dict(type='RawFrameDecode'),
    dict(type='Resize', scale=(-1, 256)),
    dict(type='CenterCrop', crop_size=image_size),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs'])
]

data = dict(
    videos_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root+'kinetics400_train_list_rawframes.txt',
        data_prefix=data_root+'rawframes_train',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root+'kinetics400_val_list_rawframes.txt',
        data_prefix=data_root+'rawframes_val',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root+'kinetics400_val_list_rawframes.txt',
        data_prefix=data_root+'rawframes_val',
        pipeline=val_pipeline))