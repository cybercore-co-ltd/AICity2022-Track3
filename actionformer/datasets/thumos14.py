import os
import json
import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F
import mmcv
from .datasets import register_dataset
from .data_utils import truncate_feats
import random
from einops import rearrange

@register_dataset("thumos")
class THUMOS14Dataset(Dataset):
    def __init__(
        self,
        is_training,     # if in training mode
        split,           # split, a tuple/list allowing concat of subsets
        feat_folder,     # folder for features
        json_file,       # json file for annotations
        feat_stride,     # temporal stride of the feats
        num_frames,      # number of frames for each feat
        default_fps,     # default fps
        downsample_rate, # downsample rate for feats
        max_seq_len,     # maximum sequence length during training
        trunc_thresh,    # threshold for truncate an action segment
        crop_ratio,      # a tuple (e.g., (0.9, 1.0)) for random cropping
        input_dim,       # input feat dim
        num_classes,     # number of action categories
        file_prefix,     # feature file prefix if any
        file_ext,        # feature file extension if any
        force_upsampling # force to upsample to max_seq_len
    ):
        # file path
        assert os.path.exists(feat_folder) and os.path.exists(json_file)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == None or len(crop_ratio) == 2
        self.feat_folder = feat_folder
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.file_ext = file_ext
        self.json_file = json_file

        # split / training mode
        self.split = split
        self.is_training = is_training

        # features meta info
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = None
        self.crop_ratio = crop_ratio

        # load database and select the subset
        dict_db, label_dict = self._load_json_db(self.json_file)
        assert len(label_dict) == num_classes
        self.data_list = dict_db
        self.label_dict = label_dict

        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'thumos-14',
            'tiou_thresholds': np.linspace(0.8, 0.9, 3),
            # we will mask out cliff diving
            'empty_label_ids': [],
        }

    def get_attributes(self):
        return self.db_attributes

    def _load_json_db(self, json_file):
        # load database and select the subset
        with open(json_file, 'r') as fid:
            json_data = json.load(fid)
        json_db = json_data['database']

        # if label_dict is not available
        if self.label_dict is None:
            label_dict = {}
            for key, value in json_db.items():
                for act in value['annotations']:
                    if act['label']!='Normal_Forward_Driving':
                        label_dict[act['label']] = act['label_id']

        # fill in the db (immutable afterwards)
        dict_db = tuple()
        for key, value in json_db.items():
            # skip the video if not in the split
            if value['subset'].lower() not in self.split:
                continue
            # or does not have the feature file
            splits_name = key.split('_')
            if 'Rear' in splits_name or 'Rightside' in splits_name:
                id=splits_name[4]
            elif 'Right' in splits_name:
                id=splits_name[5]
            elif 'Dashboard' in splits_name:
                id=splits_name[3]
            else:
                import ipdb;ipdb.set_trace()
            sub_track = splits_name[-1]
            feat = 'TSP'
            if feat=='TSP':
                #single view_concat
                feat_file = self.feat_folder+f'User_id_{id}_NoAudio_{sub_track}'+self.file_ext
                # feat_file = self.feat_folder+f'Dashboard_User_id_{id}_NoAudio_{sub_track}'+self.file_ext
            else:
                feat_file = self.feat_folder+f'Dashboard_User_id_{id}_NoAudio_{sub_track}'+self.file_ext

            if not os.path.exists(feat_file):
                feat_file = self.feat_folder+f'user_id_{id}_NoAudio_{sub_track}'+self.file_ext
            if not os.path.exists(feat_file):
                import ipdb;ipdb.set_trace()

            # get fps if available
            if self.default_fps is not None:
                fps = self.default_fps
            elif 'fps' in value:
                fps = value['fps']
            else:
                assert False, "Unknown video FPS."

            # get video duration if available
            if 'duration' in value:
                duration = value['duration']
            else:
                duration = 1e8

            # get annotations if available
            if ('annotations' in value) and (len(value['annotations']) > 0):
                # a fun fact of THUMOS: cliffdiving (4) is a subset of diving (7)
                # we remove all cliffdiving from training and output 0 at inferenece
                # as our model can't assign two labels to the same segment
                segments, labels = [], []
                for act in value['annotations']:
                    # if act['label_id'] != 0:
                        #remove cls normal
                    segments.append(act['segment'])
                    labels.append([label_dict[act['label']]])
                try:
                    segments = np.asarray(segments, dtype=np.float32)
                    labels = np.squeeze(np.asarray(labels, dtype=np.int64), axis=1)
                except:
                    segments = None
                    labels = None

            else:
                segments = None
                labels = None
            dict_db += ({'id': key,
                         'fps' : fps,
                         'duration' : duration,
                         'segments' : segments,
                         'labels' : labels
            }, )

        return dict_db, label_dict

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preporcess the data
        video_item = self.data_list[idx]
        # load features
        splits_name = video_item['id'].split('_')
        if 'Rear' in splits_name or 'Rightside' in splits_name:
            id=splits_name[4]
        elif 'Right' in splits_name:
            id=splits_name[5]
        elif 'Dashboard' in splits_name:
            id=splits_name[3]
        else:
            import ipdb;ipdb.set_trace()
        sub_track = splits_name[-1]
        feat = 'TSP'
        if feat=='TSP':
            #single view_concat
            feat_file = self.feat_folder+f'User_id_{id}_NoAudio_{sub_track}'+self.file_ext
            # feat_file = self.feat_folder+f'Dashboard_User_id_{id}_NoAudio_{sub_track}'+self.file_ext

        else:
            feat_file = self.feat_folder+f'Dashboard_User_id_{id}_NoAudio_{sub_track}'+self.file_ext
        # feat_file = os.path.join(file_feat)
        if not os.path.exists(feat_file):
            feat_file = self.feat_folder+f'user_id_{id}_NoAudio_{sub_track}'+self.file_ext
        if not os.path.exists(feat_file):
            import ipdb;ipdb.set_trace()

        feats = mmcv.load(feat_file)
        # vidconv_file = '/home/ccvn/Workspace/ngocnt/AICity2022-Track3/tsp_features/vidconv/'+feat_file.split('/')[-1].replace('user','User')
        # vidconv_features = mmcv.load(vidconv_file)
        # vidconv_features = rearrange(vidconv_features, 't bs c -> t (bs c)')
        # t_min = min(vidconv_features.shape[0], feats.shape[0])
        # feats = vidconv_features
        #use temporal crop
        _video_item = video_item.copy()

        # deal with downsampling (= increased feat stride)
        feats = feats[::self.downsample_rate, :]
        feat_stride = self.feat_stride * self.downsample_rate
        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))

        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if _video_item['segments'] is not None:
            segments = torch.from_numpy(
                (_video_item['segments'] * _video_item['fps'] - 0.5 * self.num_frames) / feat_stride
            )
            labels = torch.from_numpy(_video_item['labels'])
        else:
            segments, labels = None, None

        # return a data dict
        data_dict = {'video_id'        : _video_item['id'],
                     'feats'           : feats,      # C x T
                     'segments'        : segments,   # N x 2
                     'labels'          : labels,     # N
                     'fps'             : _video_item['fps'],
                     'duration'        : _video_item['duration'],
                     'feat_stride'     : feat_stride,
                     'feat_num_frames' : self.num_frames}

        # truncate the features during training
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, self.crop_ratio
            )

        return data_dict


    def _TemporalRandomCrop(self, raw_feature, video_item, stride=0.5,clip_length=4,fps=30):
        """Random Crop the video along the temporal dimension.

        Required keys are:
        "duration_frame", "duration_second", 
        "feature_frame","annotations", 
        """
        import random
        self.crop_length = 256
        video_second = video_item['duration']
        feature_length = raw_feature.shape[0] # Total features extract from full video
        segments = video_item['segments']
        labels = video_item['labels']
        n_segments = len(segments)


        # convert crop_length to second unit
        haft_clip_length = clip_length/2
        patch_length = min(self.crop_length, feature_length)
        patch_length_second = (patch_length)*stride

        # patch_length_second = (patch_length-1)*stride+clip_length
        start_max = video_second-patch_length_second

        while True:
            # Select the start frame randomly
            i = random.randint(0, n_segments - 2)
            start_0 = 0 if i==0 else segments[i-1][1]
            start_1 = min(segments[i][0], start_max)
            start_choice = np.arange(start_0,start_1,stride).tolist()
            if len(start_choice)==0:
                continue
            start = random.choice(start_choice)
            end = start + patch_length_second

            # Crop the feature according to the start and end frame
            start_feature_idx = int((start+1)/stride)
            end_feature_idx = start_feature_idx + patch_length
            raw_feature = raw_feature[start_feature_idx:end_feature_idx, :]
            
            # Modify the labels
            new_segments, new_labels = [], []
            for _seg, _label in zip(segments,labels):
                if _seg[0]>=start and _seg[0]<end:
                    # _start=_seg[0]-(start+stride),
                    # _end= min(_seg[1]-(start+stride),patch_length_second)
                    _start = _seg[0]-start
                    _end = min(_seg[1]-start,patch_length_second)
                    new_segments.append([_start, _end])
                    new_labels.append(_label)
            video_item['segments'] = np.array(new_segments)
            video_item['labels'] = np.array(new_labels)
            video_item['duration'] = patch_length_second
            return raw_feature, video_item


