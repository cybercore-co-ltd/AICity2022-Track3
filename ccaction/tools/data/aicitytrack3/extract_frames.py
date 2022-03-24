# Steps:
# 1. Collect the set of csv files
# 2. Create a list of all the video names
# 3. For each video, extract the frames, crop, resize. To save disk usage, we can reduce the fps from 30 to 15.
# 4. check save frame is in the interval of action:
# 5.    if yes, save to folder {action_name}/video_id 
# 6.    if no, save to folder num_classes/video_id
# 7.    if start, save to folder start/video_id
# 8.    if end, save to folder end/video_id 

import argparse
import glob
from logging import raiseExceptions
import os
import pandas as pd
import os.path as osp
import sys
import warnings
from multiprocessing import Lock, Pool
from datetime import datetime
import mmcv
import numpy as np



def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('--src_dir', type=str, help='source video directory', 
                        default='/ssd3/data/ai-city-2022/Track3/raw_video/A1')
    parser.add_argument('--out_dir', type=str, help='output rawframe directory')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=8,
        help='number of workers to build rawframes')
    parser.add_argument(
        '--out-format',
        type=str,
        default='jpg',
        choices=['jpg', 'h5', 'png'],
        help='output format')
    parser.add_argument(
        '--new-width', type=int, default=0, help='resize image width')
    parser.add_argument(
        '--new-height', type=int, default=0, help='resize image height')
    parser.add_argument(
        '--new-short',
        type=int,
        default=256,
        help='resize image short side length keeping ratio')
    parser.add_argument(
        '--report-file',
        type=str,
        default='build_report.txt',
        help='report to record files which have been successfully processed')
    args = parser.parse_args()

    return args

def crop_resize_write_vid(frames, view, out_full_path):
    # Different view has different crop areas
    if view=='Dashboard':
        frames = [f[140:1000,510:1760,:] for f in frames]
    elif view=='Rear':
        frames = [f[140:1000,820:,:] for f in frames]
    elif view=='Rightside':
        frames = [f[80:1000,750:,:] for f in frames]
    else:
        raiseExceptions('view not supported')

    run_success = -1

    for i, vr_frame in enumerate(frames):
        if vr_frame is not None:
            w, h, _ = np.shape(vr_frame)
            if args.new_short == 0:
                if args.new_width == 0 or args.new_height == 0:
                    # Keep original shape
                    out_img = vr_frame
                else:
                    out_img = mmcv.imresize(
                        vr_frame,
                        (args.new_width, args.new_height))
            else:
                if min(h, w) == h:
                    new_h = args.new_short
                    new_w = int((new_h / h) * w)
                else:
                    new_w = args.new_short
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

    # if run_success == 0:
    #     print(f'{task} {vid_id} {vid_path} done')
    #     sys.stdout.flush()

    #     lock.acquire()
    #     with open(report_file, 'a') as f:
    #         line = full_path + '\n'
    #         f.write(line)
    #     lock.release()
    # else:
    #     print(f'{task} {vid_id} {vid_path}  got something wrong')
    #     sys.stdout.flush()

    return run_success


# def init(lock_):
#     global lock
#     lock = lock_

if __name__ == '__main__':
    args = parse_args()
    csv_files = glob.glob(os.path.join(args.src_dir,'annotations', '*.csv'))
    fps=30
    tol=int(1.75*fps)
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        user_id=csv_file.split('/')[-1].split('.')[0]

        # Convert the start and end time to seconds
        df['Start Time'] = pd.to_timedelta(df['Start Time']).dt.total_seconds()
        df['End Time'] = pd.to_timedelta(df['End Time']).dt.total_seconds()

        # Get the video for each view by splitting the rows
        rows_with_filename = df[~df['Filename'].isnull()]
        row_start = list(rows_with_filename.index.values)
        video_names = rows_with_filename['Filename'].values
        row_end = row_start[1:]
        row_end.append(len(df.index))

        # Split the df by video views
        df_list = []
        for rs,re in zip(row_start,row_end):
            df_list.append(df.iloc[rs:re])

        # Extract video frames
        for vid_name,df_vid in zip(video_names,df_list):
            # Open videos
            vid_path = os.path.join(args.src_dir,user_id,
                                    vid_name[:-1] + 'NoAudio_' + vid_name[-1]+'.MP4')
            vid = mmcv.VideoReader(vid_path) 
            
            # Extract frames for each segment: 
            for i,row in df_vid.iterrows():
                s_time,e_time = int(row['Start Time']),int(row['End Time'])
                camera_view, label = row['Camera View'], row['Label/Class ID'] 
                # We clip the first and last 1.75 seconds to avoid the edge effects
                s,e=s_time*fps , e_time*fps
                import pdb; pdb.set_trace()
                vid_fg = crop_resize_write_vid(vid[s+tol:e-tol], view=camera_view,
                                out_full_path=os.path.join(args.out_dir,
                                            label,f'{vid_name}_{s_time}_{e_time}'))
                vid_start = crop_resize_write_vid(vid[s-tol:s+tol], view=camera_view, 
                                out_full_path=os.path.join(args.out_dir,
                                            18,f'{vid_name}_{s_time}'))
                vid_end = crop_resize_write_vid(vid[e-tol:e+tol], view=camera_view,
                                out_full_path=os.path.join(args.out_dir,
                                            19,f'{vid_name}_{e_time}'))
                
                if i < len(df_vid)-1:
                    # We ignore the background frames after the last action
                    next_s_time = int(df_vid.iloc[i+1]['Start Time'])
                    vid_bg = crop_resize_write_vid(vid[e+tol:next_s_time*fps-tol], view=camera_view,
                                out_full_path=os.path.join(args.out_dir,
                                            0,f'{vid_name}_{e_time}_{next_s_time}'))

                # Save the frames
                if not os.path.exists():
                    os.makedirs(os.path.join(args.out_dir,label))
                

                
                



            
