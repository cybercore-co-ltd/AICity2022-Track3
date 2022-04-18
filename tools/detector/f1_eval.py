from requests import delete
from mmaction.datasets import build_dataset
import mmcv
from mmcv import Config
import random
import numpy as np
import math
import argparse

CLASS_MAPPING =[
                    'Drinking', \
                    'Phone_Call_right', \
                    'Phone_Call_left', \
                    'Eating', \
                    'Text_Right', \
                    'Text_Left', \
                    'Hair_makeup', \
                    'Reaching_behind', \
                    'Adjust_control_panel', \
                    'Pick_up_from_floor_Driver', \
                    'Pick_up_from_floor_Passenger', \
                    'Talk_to_passenger_at_the_right', \
                    'Talk_to_passenger_at_backseat', \
                    'yawning', \
                    'Hand_on_head', \
                    'Singing_with_music', \
                    'shaking_or_dancing_with_music']

def parse_args():
    parser = argparse.ArgumentParser(description='Build file list')
    parser.add_argument('json_output_path', type=str, help='source video directory', 
                        default='actionformer_vidconv.json')
    args = parser.parse_args()

    return args

def get_f1_eval(args):
    cfg = Config.fromfile('configs/aicity/actionformer/distracted_driving.py')
    eval_config = {}
    dataset = build_dataset(cfg.data.test, dict(test_mode=True))
    path = args.json_output_path
    print(path)
    outputs = mmcv.load(path)
    eval_ouputs = []
    if 'ssc' in path:
        for video_name, proposal_list in outputs['results'].items():
            convert_proposal_list = []
            for _proposal in proposal_list:
                if float(_proposal['segment'][1]-_proposal['segment'][0])<5: continue
                if float(_proposal['segment'][1]-_proposal['segment'][0])>35: continue
                for _actions in _proposal['pred'][:1]:
                    _add_segment={}
                    _add_segment['label']=_actions[0]
                    _add_segment['score']=_actions[1]
                    if _add_segment['score']<0.4: continue
                    _add_segment['segment'] = [round(_) for _ in _proposal['segment']]
                    convert_proposal_list.append(_add_segment)
            proposal_list = convert_proposal_list
            eval_ouputs.append({'video_name': video_name, 'proposal_list': proposal_list})
            outputs['results'][video_name]=proposal_list
        mmcv.dump(outputs, 'submit.json')
    else:
        for video_name, proposal_list in outputs['results'].items():
            convert_proposal_list = []
            for _proposal in proposal_list:
                if float(_proposal['segment'][1]-_proposal['segment'][0])<5: continue
                if float(_proposal['segment'][1]-_proposal['segment'][0])>35: continue
                _proposal['segment'] = [round(_) for _ in _proposal['segment']]
                if _proposal['score']<0.4: continue
                convert_proposal_list.append(_proposal)
            proposal_list = convert_proposal_list
            eval_ouputs.append({'video_name': video_name, 'proposal_list': proposal_list})
            outputs['results'][video_name]=proposal_list
        mmcv.dump(outputs, 'submit.json')
    try:
        eval_res = dataset.evaluate(eval_ouputs, **eval_config)
        for name, val in eval_res.items():
            print(f'{name}: {val:.04f}')
    except:
        print("Only submission")
if __name__ == '__main__':
    args = parse_args()
    get_f1_eval(args)

