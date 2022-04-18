# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import glob
import json
import os
import os.path as osp
import random
from tqdm import tqdm 
from mmcv.runner import set_random_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Build file list')
    parser.add_argument('--src-dir', type=str, help='source video directory', 
                        default='data/raw_frames/A1')
    parser.add_argument('--out-file', type=str, help='output rawframe directory',
                        default='data/raw_frames/A1.txt')
    parser.add_argument(
        '--shuffle',
        action='store_true',
        default=False,
        help='whether to shuffle the file list')
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # grab all folders in each class
    frame_dirs = glob.glob(osp.join(args.src_dir, '*', '*'))
    frame_dirs = sorted(frame_dirs)
    def locate_directory(x):
            return osp.join(osp.basename(osp.dirname(x)), osp.basename(x))

    def count_files(dir):
        lst = glob.glob(osp.join(dir, '*.jpg'))
        num_files = len(lst)
        return num_files

    file_list =[]
    for frame_dir in tqdm(frame_dirs):
        # if 'Dashboard_user_id_42271_3_127_148' in frame_dir:
        #     import pdb; pdb.set_trace()
        num_files = count_files(frame_dir)
        if num_files > 60:
            name=locate_directory(frame_dir)
            label = name.split('/')[0]
            if label=='start':
                label = 18
            elif label=='end':
                label = 19
            file_list.append(f'{name} {num_files} {label}\n')

    with open(args.out_file, 'w') as f:
        f.writelines(file_list)
    


if __name__ == '__main__':
    main()
