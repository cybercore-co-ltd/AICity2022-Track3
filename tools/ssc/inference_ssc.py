import argparse
import os
import json
import torch
import glob
import numpy as np
from mmaction.apis import init_recognizer

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from mmaction.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
from mmaction.core import OutputHook
from operator import itemgetter

from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        type=str,
        default='configs/aicity/convnext_vidconv_333_224_aicityA1_multi_view_T1k.py',
        help='Config file for detection')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='http://118.69.233.170:60001/open/AICity/track3/vidconv_classifier/best_multiview_ckpt_e12.pth',
        help='Checkpoint file for detection')
    parser.add_argument(
        '--proposal-thr',
        type=float,
        default=0.3
    )
    parser.add_argument(
        '--video-dir',
        type=str,
        default='/ssd3/data/ai-city-2022/Track3/raw_video/A2/',
        help='all videos of A2 is stored in one folder'
    )
    parser.add_argument(
        '--proposal',
        type=str,
        default='actionformer_mulviews_thebest_0804.json',
        help='output json file from action-former'
    )
    parser.add_argument(
        '--device',
        type=str,
        default="cuda:0",
        help="Specify cuda device"
    )
    parser.add_argument(
        '--outdir',
        type=str,
        default="ssc_json_folder",
        help="json files for each video will be stored here"
    )
    return parser.parse_args()


def label_name_mapping(label_id):

    mapping = {
        0: "Drinking",
        1: "Phone Call(right)",
        2: "Phone Call(left)",
        3: "Eating",
        4: "Text(Right)",
        5: "Text(Left)",
        6: "Hair / makeup",
        7: "Reaching behind",
        8: "Adjust control panel",
        9: "Pick up from floor(Driver)",
        10: "Pick up from floor(Passenger)",
        11: "Talk to passenger at the right",
        12: "Talk to passenger at backseat",
        13: "yawning",
        14: "Hand on head",
        15: "Singing with music",
        16: "shaking or dancing with music"
    }
    return mapping[label_id]


def inference_recognizer_multiview(model, video, outputs=None, as_tensor=True):
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

    num_classes = scores.shape[-1]
    score_tuples = tuple(zip(range(num_classes), scores))
    score_sorted = sorted(score_tuples, key=itemgetter(1), reverse=True)

    top5_label = score_sorted[:5]
    if outputs:
        return top5_label, returned_features
    return top5_label


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

    # -----loading model
    model = init_recognizer(args.config, args.checkpoint, device=args.device)
    model.eval()
    os.makedirs(args.outdir, exist_ok=True)

    # ------------------------
    for bmn_keys, results in tqdm(json_file['results'].items()):

        if "Dashboard" not in bmn_keys:
            print("Passing here")
            continue
        video_name = bmn_keys
        # getting user_id
        user_id = video_name.split("_")[-3]
        user_id = "user_id_" + user_id

        video_name = video_name + ".MP4"
        new_json_file = {}

        dashboard_video = video_name
        dashboard_video = process_name(dashboard_video)
        print("Video_name: ", video_name)
        new_json_file[video_name] = []

        for _result in results:
            if _result['score'] < args.proposal_thr:
                _result['pred'] = []
                _result['class_name'] = []
                new_json_file[video_name].append(_result)
                continue
            start_time = int(_result["segment"][0])
            end_time = int(_result["segment"][1])

            # cutting video here
            ffmpeg_extract_subclip(os.path.join(args.video_dir, user_id, dashboard_video), start_time, end_time,
                                   targetname=dashboard_video)

            rear_video = dashboard_video.replace("Dashboard", "Rear_view")
            ffmpeg_extract_subclip(os.path.join(args.video_dir, user_id, rear_video),
                                   start_time, end_time,
                                   targetname=rear_video)

            rightside_video = dashboard_video.replace(
                "Dashboard", "Rightside_window")

            if not os.path.exists(os.path.join(args.video_dir, user_id, rightside_video)):
                if "Rightside" in rightside_video:
                    rightside_video = rightside_video.replace(
                        "Rightside", "Right_side")
                elif "Right_side" in rightside_video:
                    rightside_video = rightside_video.replace(
                        "Right_side", "Rightside")

            ffmpeg_extract_subclip(os.path.join(args.video_dir, user_id, rightside_video),
                                   start_time, end_time,
                                   targetname=rightside_video)
            try:
                pred_result = inference_recognizer_multiview(
                    model, dashboard_video)
            except:
                print("--------- Error Here------------")
                continue

            # ----------- for local evaluation
            pred_class_name = [label_name_mapping(tmp[0])
                               for tmp in pred_result]
            pred_result = [list(map(np.float64, tmp))
                           for tmp in pred_result]
            _result['pred'] = pred_result
            _result['class_name'] = pred_class_name
            new_json_file[video_name].append(_result)

        with open(os.path.join(args.outdir, bmn_keys + ".json"), "w") as f:
            json.dump(new_json_file, f)
        # remove old video name
        filelist = glob.glob(os.path.join("./", "*.MP4"))
        for f in filelist:
            os.remove(f)
