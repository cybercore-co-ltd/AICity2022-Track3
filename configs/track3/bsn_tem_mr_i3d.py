_base_ = [
    '../_base_/datasets/track3_i3d.py',
    '../mmaction/_base_/models/bsn_tem.py', 
    '../mmaction/_base_/default_runtime.py'
]
# model settings
model = dict(type='BSN_MR')
    
custom_imports = dict(imports=['ccaction'], allow_failed_imports=False)
evaluation = dict(interval=1, metrics=['AR@AN'])

# optimizer
optimizer = dict(
    type='Adam', lr=0.001, weight_decay=0.0001)  # this lr is used for 2 gpus
optimizer_config = dict(grad_clip=None)
# learning policy
lr_config = dict(policy='step', step=7)
total_epochs = 12

# runtime settings
log_config = dict(interval=2, hooks=[dict(type='TextLoggerHook')])
work_dir = './work_dirs/bsn_tem_mr_i3d/'
output_config = dict(out=f'{work_dir}/results.json', output_format='json')
load_from = 'https://download.openmmlab.com/mmaction/localization/bsn/bsn_tem_400x100_1x16_20e_mmaction_clip/bsn_tem_400x100_1x16_20e_mmaction_clip_20200809-0a563554.pth'