_base_ = 'bmn_i3d.py'

model = dict(
    type='BMN_MR',
    intra_loss_weight=10,
)


# runtime settings
data = dict(
    videos_per_gpu=8,
    workers_per_gpu=4,
)
work_dir = './work_dirs/bmn_mr_i3d/'
output_config = dict(out=f'{work_dir}/results.json', output_format='json')