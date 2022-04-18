import argparse

import os
import os.path as osp
import mmcv
import numpy as np
from logging import raiseExceptions
import warnings

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--video-dir',
        type=str,
        default="A1",
        help="folder of trimm video"
    )
    parser.add_argument(
        '--label-file',
        type=str,
        default='dashboard_train_without_bg.csv',
        help='contained only dashboard label'
    )
    parser.add_argument(
        '--outdir',
        type=str,
        default="train_rawframe_ssc",
        help="json files for each video will be stored here"
    )
    return parser.parse_args()

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
    os.makedirs(args.outdir, exist_ok=True)
    # processing video
    for vid_path in os.listdir(args.video_dir):

        vid_path = osp.join(args.video_dir, vid_path)
        print(f"Process video: {vid_path}")
        view = osp.basename(vid_path).split('_')[0]
        out_full_path = os.path.join(args.outdir, osp.basename(vid_path)[:-4])

        vid = mmcv.VideoReader(vid_path)
        crop_resize_write_vid(vid, view, out_full_path)

    # processing label
    tmp_file = open(args.label_file,'r+')
    lines = tmp_file.readlines()
    tmp_file.truncate(0)
    tmp_file.close()
    out_file = open(args.label_file,'w')

    for line in lines:
        line = line.replace("\n","")
        video_name, label = line.split(" ")
        video_name = video_name.replace(".mp4","")
        rear_video = video_name.replace("Dashboard", "Rear_view")
        rightside_video = video_name.replace("Dashboard", "Right_side_window")
        if not os.path.exists(os.path.join(args.outdir, rightside_video)):
            rightside_video = rightside_video.replace("Right_side", "Rightside")

        #process total_frame
        total_frame = np.min([len(os.listdir(os.path.join(args.outdir, video_name))),
                             len(os.listdir(os.path.join(args.outdir, rear_video))),
                             len(os.listdir(os.path.join(args.outdir, rightside_video)))])
        line = video_name+ " "+ str(total_frame) +" " + label+" "+rear_video +" "+rightside_video + "\n"
        out_file.write(line)
    out_file.close()
