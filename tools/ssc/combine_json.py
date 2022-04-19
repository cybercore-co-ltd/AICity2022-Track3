import json
import os
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--proposal',
        type=str,
        default='actionformer_mulviews_thebest_0804.json',
        help='output json file from action-former'
    )
    parser.add_argument(
        '--outdir',
        type=str,
        default="ssc_json_folder",
        help="json files for each video will be stored here"
    )
    parser.add_argument(
        '--output',
        type=str,
        default="actionformer_mulview_ssc.json",
        help="json files output"
    )
    return parser.parse_args()


def process_name(dashboard_video):
    if "Dashboard" in dashboard_video:
        return dashboard_video
    elif "Rear_view" in dashboard_video:
        return dashboard_video.replace("Rear_view", "Dashboard")
    elif "Rightside_window" in dashboard_video:
        return dashboard_video.replace("Rightside_window", "Dashboard")
    elif "Right_side_window" in dashboard_video:
        return dashboard_video.replace("Right_side_window", "Dashboard")


if __name__ == "__main__":
    args = parse_args()
    json_file = json.load(open(args.proposal, "r"))
    json_folder = args.outdir

    new_json_file = json_file.copy()
    new_json_file['results'] = {}

    for video_name, _ in json_file['results'].items():
        dashboard_name = video_name
        dashboard_name = process_name(dashboard_name)

        tmp = json.load(
            open(os.path.join(json_folder, dashboard_name + ".json"), "r"))

        new_json_file['results'][video_name] = tmp[dashboard_name+".MP4"]

    with open(args.output, "w") as f:
        json.dump(new_json_file, f)
