import mmcv
import shutil
import os
from tqdm import tqdm
from mmaction.localization.proposal_utils import soft_nms
import numpy as np
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Build file list')
    parser.add_argument('json_path', type=str, help='source video directory', 
                        default='actionformer_vidconv.json')
    parser.add_argument('in_path', type=str, help='source video directory', 
                        default='actionformer_vidconv.json')
    parser.add_argument('output_path', type=str, help='source video directory', 
                        default='actionformer_vidconv.json')
    
    
    args = parser.parse_args()

    return args

def get_data_pseudo(args):
    input_pseudo_data=args.in_path
    output_pseudo_data=args.output_path
    a2_pseudo = mmcv.load(args.json_path)['results']
    for _k, _v in tqdm(a2_pseudo.items()):    
        #add bacground
        class_added = []
        cls_0_candidates = [_ for _ in _v if _['score']<0.05 ]
        cls_0_candidates = np.array([_['segment']+[1-_['score']] for _ in cls_0_candidates])
        # soft_nms_alpha=0.4, soft_nms_low_threshold=0.5, soft_nms_high_threshold=0.9, topk=30
        cls0 = soft_nms(cls_0_candidates, 0.4, 0.5, 0.9,30)

        for _seg in cls0:
            start =round(_seg[0])
            end = round(_seg[1])
            duration = end-start
            if duration <=8 and duration>=3: #just add valid segment (3s<=segment<=8s)
                label = 0
                for ind, id_img in enumerate(range(start*30, end*30+1)):
                    src = f'{input_pseudo_data}/{_k}/img_{id_img+1:05d}.jpg'
                    dest_folder = output_pseudo_data + f'/{label}/{_k}_{start}_{end}'
                    dest_folder = dest_folder.replace('_NoAudio_', '_')
                    if not os.path.exists(dest_folder):
                        os.makedirs(dest_folder)
                    dest = dest_folder +f'/img_{ind + 1:05d}.jpg'
                    shutil.copyfile(src, dest)
        
        #add foreground
        for pred in _v:
            start, end = round(pred['segment'][0]), round(pred['segment'][1])
            if len(pred['pred'])>0:
                score = pred['pred'][0][1]
                if score>0.3:#only run vidvonv with score > 0.3
                    label = int(pred['pred'][0][0]+1)
                else:
                    continue
                for ind, id_img in enumerate(range(start*30, end*30+1)):
                    src = f'{input_pseudo_data}/{_k}/img_{id_img+1:05d}.jpg'
                    dest_folder = output_pseudo_data + f'/{label}/{_k}_{start}_{end}'
                    dest_folder = dest_folder.replace('_NoAudio_', '_')
                    if not os.path.exists(dest_folder):
                        os.makedirs(dest_folder)
                    dest = dest_folder +f'/img_{ind + 1:05d}.jpg'
                    shutil.copyfile(src, dest)

if __name__ == '__main__':
    args = parse_args()
    get_data_pseudo(args)