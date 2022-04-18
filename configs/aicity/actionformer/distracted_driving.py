
custom_imports = dict(imports=['ccaction'], allow_failed_imports=False)
# dataset settings
dataset_type = 'DistractedDriving'
data_root = '/ssd3/data/ai-city-2022/Track3'
ann_file = 'annotations/a1a2_anns.json'
test_pipeline = [
    dict(type='CCLoadLocalizationFeature', raw_feature_ext='.pkl'),
    dict(
        type='Collect',
        keys=['raw_feature'],
        meta_name='video_meta',
        meta_keys=[
            'video_name', 'duration_second', 'duration_frame', 'annotations',
            'feature_frame'
        ]),
    dict(type='ToTensor', keys=['raw_feature']),
]
train_pipeline = [
    dict(type='CCLoadLocalizationFeature', raw_feature_ext='.pkl', with_TFS=True, dropout=0.5),
    dict(type='GenerateLocalizationLabels'),
    dict(
        type='Collect',
        keys=['raw_feature', 'gt_bbox'],
        meta_name='video_meta',
        meta_keys=['video_name']),
    dict(type='ToTensor', keys=['raw_feature', 'gt_bbox']),
    dict(
        type='ToDataContainer',
        fields=[dict(key='gt_bbox', stack=False, cpu_only=True)])
]
val_pipeline = [
    dict(type='CCLoadLocalizationFeature', raw_feature_ext='.pkl'),
    dict(type='GenerateLocalizationLabels'),
    dict(
        type='Collect',
        keys=['raw_feature', 'gt_bbox'],
        meta_name='video_meta',
        meta_keys=[
            'video_name', 'duration_second', 'duration_frame', 'annotations',
            'feature_frame'
        ]),
    dict(type='ToTensor', keys=['raw_feature', 'gt_bbox']),
    dict(
        type='ToDataContainer',
        fields=[dict(key='gt_bbox', stack=False, cpu_only=True)])
]
data = dict(
    videos_per_gpu=1,
    workers_per_gpu=4,
    train_dataloader=dict(drop_last=True),
    val_dataloader=dict(videos_per_gpu=1),
    test_dataloader=dict(videos_per_gpu=1),
    test=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=test_pipeline,
        data_prefix=data_root + '/raw_video/A1/'),
    val=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=val_pipeline,
        data_prefix=data_root + '/raw_video/A2/'),
    train=dict(
        type=dataset_type,
        ann_file=ann_file,
        pipeline=train_pipeline,
        data_prefix=data_root + '/raw_video/A2/'))
evaluation = dict(interval=1, metrics=['AR@AN'])