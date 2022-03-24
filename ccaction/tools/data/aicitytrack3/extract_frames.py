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
import os
import pandas as pd
import os.path as osp
import sys
import warnings
from multiprocessing import Lock, Pool
from datetime import datetime
import mmcv
import numpy as np

def crop_resize_vid(frames, output_size=(224,224), H=1080,W=1920, view=False):
    if view=='Dashboard':
        frames = [f[140:1000,510:1764,:] for f in frames]
    elif view=='Rear':
        frames = [f[140:1000,820:1920,:] for f in frames]
    elif view=='Rightsize':
        frames = [f[140:1000,0:1400,:] for f in frames]
    frames = [mmcv.imresize(f, output_size) for f in frames]
    return frames

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
        default=0,
        help='resize image short side length keeping ratio')
    parser.add_argument('--num-gpu', type=int, default=8, help='number of GPU')
    parser.add_argument(
        '--use-opencv',
        action='store_true',
        help='Whether to use opencv to extract rgb frames')
    parser.add_argument(
        '--report-file',
        type=str,
        default='build_report.txt',
        help='report to record files which have been successfully processed')
    args = parser.parse_args()

    return args

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
                s,e = int(row['Start Time'])*fps,int(row['End Time'])*fps
                # We clip the first and last 1.75 seconds to avoid the edge effects
                vid_start = vid[s-tol:s+tol]
                vid_fg = vid[s+tol:e-tol]
                vid_end = vid[e-tol:e+tol]
                import pdb; pdb.set_trace()
                if i < len(df_vid)-1:
                    # We ignore the background frames after the last action
                    next_s = int(df_vid.iloc[i+1]['Start Time']*fps)
                    vid_bg = vid[e+tol:next_s-tol]
                



            
