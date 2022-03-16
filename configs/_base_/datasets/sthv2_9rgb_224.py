dataset_type = 'RawframeDataset'
# data_root = '/raid/data/sthv2/'
# data_root = '/ssd3/data/sth2sth/'
data_root = '/data/sthv2/'

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
image_size = 224
sthv2_flip_label_map = {86: 87, 87: 86, 93: 94, 94: 93, 166: 167, 167: 166}
train_pipeline = [
    # dict(type='RandSampleFrames', clip_len=9,
    #      range=(1,3), num_clips=1, dataset_name='sth2sth'),
    dict(type='RandSampleFrames', clip_len=9,
         range=(5,7), num_clips=1, dataset_name='sth2sth'),
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
    dict(type='Flip', flip_ratio=0.5, flip_label_map=sthv2_flip_label_map),
    # dict(type='Imgaug', transforms='default'),
    dict(type='Normalize', **img_norm_cfg),
    # dict(type='RandomErasing', probability=0.1),
    dict(type='FormatShape', input_format='NCHW'),
    dict(type='Collect', keys=['imgs', 'label'], meta_keys=[]),
    dict(type='ToTensor', keys=['imgs', 'label'])
]
val_pipeline = [
    dict(
        type='AdaptSampleFrames',
        clip_len=9,
        frame_interval=6,
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
        type='AdaptSampleFrames',
        clip_len=9,
        frame_interval=6,
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

data = dict(
    videos_per_gpu=16,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        ann_file=data_root+'/annotations/sthv2_train_list_rawframes.txt',
        data_prefix=data_root+'rawframes',
        pipeline=train_pipeline),
    val=dict(
        type=dataset_type,
        ann_file=data_root+'/annotations/sthv2_val_list_rawframes.txt',
        data_prefix=data_root+'rawframes',
        pipeline=val_pipeline),
    test=dict(
        type=dataset_type,
        ann_file=data_root+'/annotations/sthv2_val_list_rawframes.txt',
        data_prefix=data_root+'rawframes',
        pipeline=val_pipeline))
