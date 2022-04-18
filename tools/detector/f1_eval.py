from requests import delete
from mmaction.datasets import build_dataset
import mmcv
from mmcv import Config
import random
import numpy as np
import math
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


cfg = Config.fromfile('configs/track3/bmn_i3d_track3.py')
eval_config = {}
dataset = build_dataset(cfg.data.test, dict(test_mode=True))
path = 'result_eval.json'
print(path)
outputs = mmcv.load(path)
eval_ouputs = []
if 'actionformer_base' in path or 'result_eval.json' in path:
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
elif 'result_e4_A2_a_Chuong_0704.json' in path:
    for video_name, proposal_list in outputs['results'].items():
        convert_proposal_list = []
        for _proposal in proposal_list:
            if float(_proposal['segment'][1]-_proposal['segment'][0])<5: continue
            for _actions in _proposal['pred'][:1]:
                _add_segment={}
                _add_segment['label']=_actions[0]
                _add_segment['score']=_actions[1]
                if _add_segment['score']<0.4: continue
                _add_segment['segment'] =np.floor(np.array(_proposal['segment']))
                convert_proposal_list.append(_add_segment)
        proposal_list = convert_proposal_list
        eval_ouputs.append({'video_name': video_name, 'proposal_list': proposal_list})
        outputs['results'][video_name]=proposal_list
    mmcv.dump(outputs, 'submit.json')
elif 'afsd' in path:
    for video_name, proposal_list in outputs['results'].items():
        convert_proposal_list = []
        for _proposal in proposal_list:
            _proposal['label']=int(dataset.CLASS_MAPPING[_proposal['label']])-1
            convert_proposal_list.append(_proposal)
        proposal_list = convert_proposal_list
        outputs['results'][video_name]=proposal_list
        eval_ouputs.append({'video_name': video_name+'.mp4', 'proposal_list': proposal_list})
    mmcv.dump(outputs, 'submit.json')
#from confustion matrix: 5, 11, 12, 16
elif 'round1_pseudo_vidconv_1404.json' in path:
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
                # if (_proposal['segment'][1]-_proposal['segment'][0])>25:
                #     import ipdb;ipdb.set_trace()
                convert_proposal_list.append(_add_segment)
        proposal_list = convert_proposal_list
        eval_ouputs.append({'video_name': video_name, 'proposal_list': proposal_list})
        outputs['results'][video_name]=proposal_list
    mmcv.dump(outputs, 'submit.json')

eval_res = dataset.evaluate(eval_ouputs, **eval_config)
for name, val in eval_res.items():
    print(f'{name}: {val:.04f}')


train_a1 = {}
data_pseudo = mmcv.load('/home/ccvn/Workspace/ngocnt/actionformer_release/a1_a2pseudo.json')['database']
for _k, _v in data_pseudo.items():
    if _v['subset']== 'training':
        train_a1.update({_k:_v})

val_dict = {}
for _id, _data in data_pseudo.items():
    if data_pseudo[_id]['subset']=='validation':
        pred = outputs['results'][_id]
        if len(pred)==0:
            continue
        data_pseudo[_id]['annotations'] = [{'segment': seg['segment'],\
                                            'label_id': int(seg['label']),\
                                            'label':  CLASS_MAPPING[int(seg['label'])]} for seg in  pred]
        data_pseudo[_id]['subset'] = 'training'
        val_dict.update({_id:data_pseudo[_id]})
train_a1.update(val_dict)
mmcv.dump({'database': train_a1}, 'corrected_a1_a2_pseudo.json')
