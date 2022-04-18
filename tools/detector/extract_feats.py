import argparse
import glob
from logging import raiseExceptions
import os
import pandas as pd
import os.path as osp
import warnings
from multiprocessing import Lock, Pool
import mmcv
import numpy as np
from tqdm import tqdm
from mmcv import Config
from mmaction.datasets.pipelines import Compose
import glob
from mmaction.models import build_model
from mmcv.runner import load_checkpoint
import pickle
import torch
def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('--config', type=str, help='config',
                        default='configs/aicity/convnext_noforgetting_tsp_333_224_pseudoA2_v1.py')
    parser.add_argument('--ckpt', type=str, help='ckpt',
                        default='checkpoints/round1_tsp_62.5.pth')
    parser.add_argument('--in_dir', type=str, help='folder data',
                        default='data/raw_frames/A1/')
    parser.add_argument('--out_dir', type=str, help='out pickle file',
                        default='A1')

    parser.add_argument( '--interval', type=int, default=15, help='frame_interval')
    parser.add_argument( '--stride', type=int, default=8, help='stride')
    parser.add_argument( '--clip_len', type=int, default=9, help='clip_len')

    #add by ngoct
    parser.add_argument('--filelist', default='', help='output prefix')
    parser.add_argument( '--bs', type=int, default=128, help='frame_interval')

    parser.add_argument(
        '--part',
        type=int,
        default=0,
        help='which part of dataset to forward(alldata[part::total])')


    args = parser.parse_args()

    return args

def main():
    args = parse_args()

    # preprocessing data pipeline
    img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_bgr=False)
    image_size = 224
    cfg = Config.fromfile(args.config)
    data_pipeline = [
        dict(type='RawFrameDecode'),
        dict(type='Resize', scale=(224, 224), keep_ratio=False),
        # dict(type='Resize', scale=(-1, 256)),
        # dict(type='CenterCrop', crop_size=image_size),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCHW'),
        dict(type='Collect', keys=['imgs'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs'])
    ]
    data_pipeline = Compose(data_pipeline)

    # input data format
    results = dict(frame_dir = '',
                    total_frames = '',
                    filename_tmpl = 'img_{:05}.jpg',
                    modality = 'RGB',
                    frame_inds = '')

    # build the model and load checkpoint
    model = build_model(
        cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))

    load_checkpoint(model, args.ckpt, map_location='cpu')
    model = model.cuda()
    model.eval()

    model.feature_extraction=True# 4sure return BS*C


    # Using readlines()
    file_list = open(args.filelist, 'r')
    if args.part==1:
        file_list = file_list.readlines()[:15]
    else:
        file_list = file_list.readlines()[15:]

    # loop all videos
    for _vid_name in tqdm(file_list):
        video_path = args.in_dir + _vid_name[:-1]
        save_feats = []
        print(f'Process: {video_path}')
        pkl_path = osp.join(args.out_dir, osp.basename(video_path)+'.pkl')
        results['frame_dir'] = video_path
        total_frames = len(glob.glob(video_path+'/*.jpg'))
        results['total_frames'] = total_frames
        batchs = []
        cnt_bath = 0
        num_clip = range(1,total_frames+1, args.stride)
        with torch.no_grad():
            for i in tqdm(num_clip):
                frame_inds = i + np.arange(args.clip_len)*args.interval
                if frame_inds[-1]>total_frames:
                    break
                results['frame_inds'] = frame_inds
                in_data = data_pipeline(results)
                if (len(batchs)%args.bs==0) and len(batchs)>0:
                    pass
                elif i>=num_clip[-1]:
                   pass
                else:
                    batchs.append(torch.tensor(in_data['imgs']))
                    continue
                in_data_model = torch.stack(batchs)
                out_score = model.forward_backbone(in_data_model.cuda()).cpu().detach().numpy()
                batchs = []
                batchs.append(torch.tensor(in_data['imgs']))
                save_feats.append(out_score)
                
        
        save_feats = np.concatenate(save_feats)
        with open(pkl_path, 'wb') as fout:
            pickle.dump(save_feats, fout)


if __name__ == '__main__':
    main()
