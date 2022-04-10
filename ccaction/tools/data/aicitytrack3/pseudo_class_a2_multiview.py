import argparse
import glob
import os
import pandas as pd
import torch
import os.path as osp
from multiprocessing import Lock, Pool
import mmcv
from operator import itemgetter
from mmcv.parallel import collate, scatter
from mmaction.core import OutputHook
import numpy as np
from tqdm import tqdm
from mmcv import Config
from mmaction.datasets.pipelines import Compose
import glob
from mmaction.models import build_model
from mmcv.runner import load_checkpoint
from pickle5 import pickle


def inference_recognizer_multiview(model, video, outputs=None, as_tensor=True, extract_score=False):
    cfg = model.cfg
    device = next(model.parameters()).device
    test_pipeline = cfg.data.test.pipeline
    data = dict(filename=video, label=-1, start_index=0, modality='RGB')

    test_pipeline = Compose(test_pipeline)
    data = test_pipeline(data)
    data = collate([data], samples_per_gpu=1)

    if next(model.parameters()).is_cuda:
        # scatter to specified GPU
        data = scatter(data, [device])[0]

    # forward the model
    with OutputHook(model, outputs=outputs, as_tensor=as_tensor) as h:
        with torch.no_grad():
            scores = model(return_loss=False, **data)[0]
        returned_features = h.layer_outputs if outputs else None

    if extract_score:
        return scores
    else:
        num_classes = scores.shape[-1]
        score_tuples = tuple(zip(range(num_classes), scores))
        score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)

        top5_label = score_sorted[:5]
        if outputs:
            return top5_label, returned_features
        return top5_label


def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('--config', type=str, help='config',
                        default='/home/ccvn/Workspace/vinh/test/AICity2022-Track3/configs/recognition/conNext_super_feat/aicity/convnext_vidconv_333_224_aicityA1_multi_view_T1k.py')
    parser.add_argument('--ckpt', type=str, help='ckpt',
                        default='/home/ccvn/Workspace/vinh/AICity2022-Track3/work_dirs/T1k_multiview_numclip-5/best_e12.pth')
    parser.add_argument('--in_dir', type=str, help='folder data',
                        default='/raid/data/ai-city-2022/Track3/raw_frames/full_video/A2')
    parser.add_argument('--out_dir', type=str, help='out pickle file',
                        default='./A2')

    parser.add_argument('--interval', type=int,
                        default=15, help='frame_interval')
    parser.add_argument('--stride', type=int, default=6, help='stride')
    parser.add_argument('--clip_len', type=int, default=45, help='stride')
    parser.add_argument('--num_clip', type=int, default=5)
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
        dict(type='RawFrameDecode_multiviews', extract_feat=True),
        dict(type='Resize', scale=(image_size, image_size), keep_ratio=False),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='FormatShape', input_format='NCHW'),
        dict(type='Collect', keys=['imgs'], meta_keys=[]),
        dict(type='ToTensor', keys=['imgs'])
    ]
    data_pipeline = Compose(data_pipeline)

    # input data format
    results = dict(frame_dir='',
                   total_frames='',
                   filename_tmpl='img_{:05}.jpg',
                   modality='RGB',
                   num_clips=args.num_clip,
                   clip_len=args.clip_len,
                   frame_inds='')

    # build the model and load checkpoint
    model = build_model(
        cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg'))

    load_checkpoint(model, args.ckpt, map_location='cpu')
    model = model.cuda()
    model.eval()

    os.makedirs(args.out_dir, exist_ok=True)
    # loop all videos
    for vid_name in glob.glob(args.in_dir + '/*'):

        if "Dashboard" not in vid_name:
            continue
        save_feats = []
        print(f'Process: {vid_name}')
        pkl_path = osp.join(args.out_dir, osp.basename(vid_name)+'.pkl')
        results['frame_dir'] = vid_name

        # combine 3view_totalframe
        dashboard_frames = len(glob.glob(vid_name+'/*.jpg'))
        rear_view_frames = len(
            glob.glob(vid_name.replace("Dashboard", "Rear_view")+'/*.jpg'))
        rightside_vid = vid_name.replace("Dashboard", "Right_side_window")
        if not os.path.exists(rightside_vid):
            rightside_vid = rightside_vid.replace("Right_side", "Rightside")
        rightside_frames = len(glob.glob(rightside_vid+'/*.jpg'))
        total_frames = np.min(
            [dashboard_frames, rear_view_frames, rightside_frames])
        results['total_frames'] = total_frames

        for i in tqdm(range(1, total_frames+1, args.stride)):
            frame_inds = i + np.arange(args.clip_len)*args.interval
            if frame_inds[-1] > total_frames:
                break

            # -------------
            results['frame_inds'] = frame_inds
            in_data = data_pipeline(results)
            import ipdb
            ipdb.set_trace()
            out_score = model.forward_dummy(
                in_data['imgs'].unsqueeze(0).cuda()).cpu().detach().numpy()
            
            # Todo


            # save_feats.append(out_score)
        # save_feats = np.concatenate(save_feats)
        # with open(pkl_path, 'wb') as fout:
            # pickle.dump(save_feats, fout)
if __name__ == '__main__':
    main()
