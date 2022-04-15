import torch
# from mmcv import load, dump
# load_from ='/home/ccvn/Workspace/vinh/AICity2022-Track3/no_forgeting_retrain_kinetic400_e2.pth'
load_from = '/home/ccvn/Workspace/chuong/AICity2022-Track3/work_dirs/convnext_noforgetting_tsp_5/epoch_4.pth'
# load_from="/home/ccvn/Workspace/vinh/AICity2022-Track3/no_forgeting_retrain_kinetic400_e3.pth"
ckpt= torch.load(load_from)
state_dict=ckpt['state_dict']
new_state_dict = {}
for k, v in state_dict.items():
    if k.split('.')[0] in ['backbone','cls_head']:
        new_state_dict[k]=v

out_file = load_from.replace('.pth','_K400.pth')
torch.save(new_state_dict,out_file)