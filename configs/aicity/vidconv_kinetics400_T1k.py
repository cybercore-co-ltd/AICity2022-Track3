model = dict(
    type='VidConvRecognizer',
    backbone=dict(
        type='ConvNextVidBaseTem',
        arch='tiny',
        drop_path_rate=0.25,
        init_cfg=None,
    ),
    cls_head=dict(
        type='VidConvHead',
        in_channels=768,
        num_classes=400,
        spatial_type='avg',
        expand_ratio=3,
        kernel_size=3,
        dilation=7,
        norm_cfg=dict(type='SyncBN', requires_grad=True),
        dropout_ratio=0.2),
    test_cfg=dict(average_clips='prob'),
    train_cfg=None,
)
