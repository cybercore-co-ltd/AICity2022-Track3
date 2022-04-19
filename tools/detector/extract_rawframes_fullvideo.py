import os
import csv
import os.path as osp
import mmcv
import numpy as np
import glob
import tqdm
from logging import raiseExceptions
import warnings
import argparse

CLASS_NAMES = ['Normal_Forward_Driving', 'Drinking', 'Phone_Call_right', 
            'Phone_Call_left', 'Eating', 'Text_Right', 
            'Text_Left', 'Hair_makeup', 'Reaching_behind', 
            'Adjust_control_panel', 'Pick_up_from_floor_Driver', 'Pick_up_from_floor_Passenger', 
            'Talk_to_passenger_at_the_right', 'Talk_to_passenger_at_backseat', 'yawning', 
            'Hand_on_head', 'Singing_with_music', 'shaking_or_dancing_with_music']

def parse_args():
    parser = argparse.ArgumentParser(description='extract frames')
    parser.add_argument('--in-dir', type=str, help='folder coontain videos',
                        default='data/raw_video/A1')
    parser.add_argument('--out-dir', type=str, help='folder save extracted frames',
                        default='data/raw_frames/full_video/A1')
   
    args = parser.parse_args()

    return args

def crop_resize_write_vid(frames, view, out_full_path):
    new_short = 256
    if 'Dashboard' in view:
        frames = [f[:,380:1800,:] for f in frames]
    elif 'Rear' in view:
        frames = [f[:,750:,:] for f in frames]
    elif 'Right' in view:
        frames = [f[:,750:,:] for f in frames]
    else:
        raiseExceptions('view not supported')

    run_success = -1
    # Save the frames
    if not osp.exists(out_full_path):
        os.makedirs(out_full_path)

    for i, vr_frame in enumerate(frames):
        if vr_frame is not None:
            w, h, _ = np.shape(vr_frame)
            
            if min(h, w) == h:
                new_h = new_short
                new_w = int((new_h / h) * w)
            else:
                new_w = new_short
                new_h = int((new_w / w) * h)
            out_img = mmcv.imresize(vr_frame, (new_h, new_w))
            mmcv.imwrite(out_img,
                            f'{out_full_path}/img_{i + 1:05d}.jpg')
        else:
            warnings.warn(
                'Length inconsistent!'
                f'Early stop with {i + 1} out of {len(frames)} frames.'
            )
            break
    run_success = 0

    return run_success
    
               
if __name__ == '__main__':
    args = parse_args()
    if not osp.exists(args.out_dir):
        os.makedirs(args.out_dir)
    for vid_path in glob.glob(osp.join(args.in_dir,'**/*.MP4')):
        
        print(f"Process video: {vid_path}")
        view = osp.basename(vid_path).split('_')[0]
        out_full_path = osp.join(args.out_dir, osp.basename(vid_path)[:-4])

        vid = mmcv.VideoReader(vid_path)

        crop_resize_write_vid(vid, view, out_full_path)