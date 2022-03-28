import os 
import json 
import torch
from mmaction.apis import inference_recognizer, init_recognizer

from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from mmaction.datasets.pipelines import Compose
from mmcv.parallel import collate, scatter
from mmaction.core import OutputHook
from operator import itemgetter

def inference_recognizer_multiview(model, video,outputs=None, as_tensor=True):
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

if __name__=="__main__":
    
    score_thr = 0.1
    # mapping_name = {
    #     "shboard" : "Dashboard",
    #     "ar" : "Rear",
    #     "ghtside": "Rightside"
    # }
    video_dir = "/media/data/ai-city-2022/Track3/A2/val_video_2m"
    json_file = json.load(open("bmn_a2.json"))
    
    #-----loading model
    model = init_recognizer("/home/cybercore/vinh_overal/AICity2022-Track3/configs/recognition/conNext_super_feat/aicity/convnext_vidconv_333_224_aicityA1_T1k.py", 
                            "/home/cybercore/vinh_overal/AICity2022-Track3/current_best_e20.pth", device="cuda:0")
    new_json_file = json_file.copy()
    new_json_file['results'] = {}
    
    #------------------------
    for video_name, results in json_file['results'].items():
        
        
        
        video_name = video_name.split("_")
        dashboard_video = video_name.copy()
        dashboard_video[0] = "Dashboard"    
        dashboard_video = '_'.join(dashboard_video)
        video_name = '_'.join(video_name)
        new_json_file['results'][video_name] = []
        
        for _result in results:
            if _result['score'] < score_thr:
                continue
            
            start_time = int(_result["segment"][0])
            end_time = int(_result["segment"][1])
            
            # cutting video here
            ffmpeg_extract_subclip(os.path.join(video_dir, dashboard_video), start_time, end_time, 
                                   targetname=dashboard_video)
            
            rear_video = dashboard_video.replace("Dashboard", "Rear_view")
            ffmpeg_extract_subclip(os.path.join(video_dir, rear_video),
                                   start_time, end_time, 
                                   targetname=rear_video)
            
            rightside_video = dashboard_video.replace("Dashboard", "Rightside_window")
            
            if not os.path.exists(os.path.join(video_dir, rightside_video)):
                rightside_video = rightside_video.replace("Rightside","Right_side")
            ffmpeg_extract_subclip(os.path.join(video_dir, rightside_video), 
                                   start_time, end_time, 
                                   targetname=rightside_video)
            
            pred_result = inference_recognizer_multiview(model, dashboard_video)
            
            _result['pred'] = pred_result
            new_json_file['results'][video_name].append(_result)
        
        import ipdb; ipdb.set_trace()
        