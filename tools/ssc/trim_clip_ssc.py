import argparse
import os
import pandas as pd
import math

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--video-dir',
        type=str,
        default="A1",
        help="video folder: example A1"
    )
    parser.add_argument(
        '--label-file',
        type=str,
        default='dashboard_train_without_bg.csv',
        help='contained only dashboard label'
    )
    parser.add_argument(
        '--user-id',
        nargs='+',
        type=str,
        help='user-ids which are wanted to train or eval'
    )
    parser.add_argument(
        '--outdir',
        type=str,
        default="train_video_ssc",
        help="json files for each video will be stored here"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    label_file = pd.DataFrame(columns=['label'])
    label_text = []

    for sub_id in tqdm(args.user_id):
        id_folder = "user_id_"+sub_id

        # process csv_file
        csv_file = pd.read_csv(os.path.join(
            args.video_dir, id_folder, id_folder+".csv"))
        csv_file.Filename = csv_file.Filename[csv_file.Filename.str.strip(
        ) != '']
        csv_file.Filename = csv_file.Filename.fillna(method='ffill')

        for idx in range(len(csv_file)):
            df = csv_file.loc[idx]
            if len(df["Start Time"].split(":")) == 3:
                ftr = [3600, 60, 1]
            else:
                ftr = [60, 1]

            if isinstance(df["Label/Class ID"], str):
                if "na" in df["Label/Class ID"].lower() or int(df["Label/Class ID"]) == 0:
                    continue  
            else:
                if math.isnan(df["Label/Class ID"]) or int(df["Label/Class ID"]) == 0:
                    continue

            video_name = df.Filename.replace(" ", "")
            video_name = video_name[:-1] + "NoAudio_" + video_name[-1]
            sub_video_name = video_name + "_" + \
                df["Start Time"].replace(":", "") + ".mp4"
            video_name = video_name + ".MP4"

            start_time = sum(
                [a*b for a, b in zip(ftr, map(int, df["Start Time"].split(':')))])
            end_time = sum(
                [a*b for a, b in zip(ftr, map(int, df["End Time"].split(':')))])

            video_path = os.path.join(args.video_dir, id_folder, video_name)
            if not os.path.exists(video_path):
                if "user" in video_name:
                    video_name = video_name.replace("user", "User")
                else:
                    video_name = video_name.replace("User", "user")
                video_path = os.path.join(
                    args.video_dir, id_folder, video_name)

            ffmpeg_extract_subclip(video_path, start_time, end_time,
                                   targetname=os.path.join(args.outdir, sub_video_name))

            # dashboard only
            if "Dashboard" in sub_video_name:
                label_text.append(sub_video_name+" " +
                                str(int(df["Label/Class ID"])-1))

    label_file['label'] = label_text
    label_file.to_csv(args.label_file, index=False, header=False)
