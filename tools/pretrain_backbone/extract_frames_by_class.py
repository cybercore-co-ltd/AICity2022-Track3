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
import warnings
from multiprocessing import Lock, Pool
import mmcv
import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description='extract optical flows')
    parser.add_argument('--src-dir', type=str, help='source video directory', 
                        default='data/raw_video/A1')
    parser.add_argument('--out-dir', type=str, help='output rawframe directory',
                        default='data/raw_frames/A1')
    parser.add_argument( '--fps', type=int, default=30, help='fps')
    parser.add_argument(
        '--num-worker',
        type=int,
        default=10,
        help='number of workers to build rawframes')
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
    # if 'Dashboard' in view:
    #     frames = [f[140:1000,510:1760,:] for f in frames]
    # elif 'Rear' in view:
    #     frames = [f[140:1000,820:,:] for f in frames]
    # elif 'Right' in view:
    #     frames = [f[80:1000,750:,:] for f in frames]
    # else:
    #     raiseExceptions('view not supported')

    if 'Dashboard' in view:
        frames = [f[:,380:1800,:] for f in frames]
    elif 'Rear' in view:
        frames = [f[:,750:,:] for f in frames]
    elif 'Right' in view:
        frames = [f[:,750:,:] for f in frames]
    else:
        raiseExceptions('view not supported')

    #  results['imgs'] = [tmp[:, 380:width-120] for tmp in results['imgs']]
    #     results_rear['imgs'] = [tmp[:, 750:] for tmp in results_rear['imgs']]
    #     results_right['imgs'] = [tmp[:, 750:] for tmp in results_right['imgs']]

    run_success = -1
    # Save the frames
    if not osp.exists(out_full_path):
        os.makedirs(out_full_path)
        
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

    return run_success

def parsing_csv_files():
    csv_files = [file
                 for path, subdir, files in os.walk(args.src_dir)
                 for file in glob.glob(os.path.join(path, '*.csv'))]
    # csv_files = glob.glob(osp.join(args.src_dir,'annotations', '*.csv'))
    
    user_videos={}
    for csv_file in csv_files:
        df = pd.read_csv(csv_file)
        user_id=csv_file.split('/')[-1].split('.')[0]

        # Convert the start and end time to seconds
        if len(df['Start Time'][0].split(':'))==2:
            # Fix the format on several csv files
            df['Start Time'] = '00:' + df['Start Time'] 
            df['End Time'] = '00:' + df['End Time'] 
        df['Start Time'] = pd.to_timedelta(df['Start Time']).dt.total_seconds()
        df['End Time'] = pd.to_timedelta(df['End Time']).dt.total_seconds()

        # Get the video for each view by splitting the rows
        rows_with_filename = df[(~df['Filename'].isnull()) 
                                & (df['Filename']!=' ')]
        
        video_names = rows_with_filename['Filename'].values
        # Fix inconsistent video names
        id = user_id.split('_')[-1]
        if id in ['49381','35133']:
            video_names=[vid_name.replace('User','user') for vid_name in video_names]
        elif id in ['72519','65818','79336']:
            video_names=[vid_name.replace('user','User') for vid_name in video_names]
        video_names = [vid_name.strip() for vid_name in video_names]

        row_start = list(rows_with_filename.index.values)
        row_end = row_start[1:]
        row_end.append(len(df.index))

        # Split the df by video views
        df_list = []
        for rs,re in zip(row_start,row_end):
            df_vid = df.iloc[rs:re]
            df_vid = df_vid[(df_vid['Label/Class ID']!='N/A')
                            & (df_vid['Label/Class ID']!='NA ')
                            & (df_vid['Label/Class ID']!='nan')
                            & (df_vid['Label/Class ID']!='NaN')
                            & (~df_vid['Label/Class ID'].isnull())]
            df_vid['Label/Class ID']=df_vid['Label/Class ID'].astype(int)
            df_vid = df_vid[df_vid['Label/Class ID']>0]
            df_list.append(df_vid)

        user_videos[user_id.strip()]=(video_names, df_list)
    return user_videos

def extract_frames(vid_items):
    # Open videos
    vid_name,df_vid,user_id,src_dir,out_dir,fps = vid_items
    vid_path = osp.join(src_dir,user_id,
                            vid_name[:-1] + 'NoAudio_' + vid_name[-1]+'.MP4')
    print(f'Extracting frames for video: {vid_path}')
    vid = mmcv.VideoReader(vid_path) 
    # Extract frames for each segment: 
 
    # if '72519' in user_id:
    #     import pdb; pdb.set_trace()
    df_vid = df_vid.reset_index()
    for index, row in tqdm(df_vid.iterrows(), total=df_vid.shape[0]):
        s_time,e_time = int(row['Start Time']),int(row['End Time'])
        camera_view, label = row['Camera View'], row['Label/Class ID']
        # We clip the first and last 1-1.5 seconds to avoid the edge effects
        s,e=s_time*fps , e_time*fps

        tol=int(min(1,0.1*(e_time-s_time))*fps) 
        fg_path=osp.join(out_dir,f'{label}',f'{vid_name}_{s_time}_{e_time}')
        if (not osp.exists(fg_path)) or (len(os.listdir(fg_path))<e-s-2*tol) :
            # if folder does not exist or the folder is empty
            crop_resize_write_vid(vid[s+tol:e-tol], view=camera_view,
                                out_full_path=fg_path)

        tol=int(1.5*fps) 
        start_path=osp.join(out_dir,'start',f'{vid_name}_{s_time}')
        if (not osp.exists(start_path)) or (len(os.listdir(start_path))<2*tol) :
            crop_resize_write_vid(vid[s-tol:s+tol], view=camera_view, 
                                out_full_path=start_path)

        tol=int(1.5*fps) 
        end_path =osp.join(out_dir,'end',f'{vid_name}_{e_time}')
        if (not osp.exists(end_path)) or (len(os.listdir(end_path))<2*tol):
            crop_resize_write_vid(vid[e-tol:e+tol], view=camera_view,
                out_full_path=end_path)
        
        if index < len(df_vid)-1:
            # We ignore the background frames after the last action
            # if vid_name=='Rightside_window_User_id_72519_3' and label==10:
            #     import pdb; pdb.set_trace()

            next_s_time = int(df_vid.iloc[index+1]['Start Time'])
            next_s = next_s_time*fps
            tol=int(min(1,0.1*(next_s_time-e_time))*fps) 
            if next_s-tol > e+tol:
                bg_path=osp.join(args.out_dir,'0',f'{vid_name}_{e_time}_{next_s_time}')
                if (not osp.exists(bg_path)) or (len(os.listdir(bg_path))<next_s-e-2*tol):
                    crop_resize_write_vid(vid[e+tol:next_s-tol], view=camera_view,
                        out_full_path=bg_path)


def init(lock_):
    global lock
    lock = lock_

if __name__ == '__main__':
    args = parse_args()

    if not osp.isdir(args.out_dir):
        print(f'Creating folder: {args.out_dir}')
        os.makedirs(args.out_dir)

    print('Parsing CSV files')
    user_videos=parsing_csv_files()  

    for user_id, (video_names, df_list) in user_videos.items():
        # Extract video frames
        print("Extracting frames for user: ", user_id)
        # for vid_name, df_vid in zip(video_names,df_list):
        #     extract_frames((vid_name,df_vid,user_id,args.src_dir,args.out_dir,args.fps))
        
        n_videos = len(video_names)
        lock = Lock()
        pool = Pool(args.num_worker, initializer=init, initargs=(lock, ))
        pool.map(
            extract_frames,
            zip(video_names,df_list,
                n_videos* [user_id],
                n_videos* [args.src_dir],
                n_videos* [args.out_dir],
                n_videos* [args.fps]))
        pool.close()
        pool.join()

            


                

                
                



            
