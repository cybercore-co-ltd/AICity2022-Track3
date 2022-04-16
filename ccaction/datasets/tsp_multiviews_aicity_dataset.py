from mmaction.datasets.builder import DATASETS
from mmaction.datasets.rawframe_dataset import RawframeDataset
import copy
import os.path as osp
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, defaultdict

import mmcv
import numpy as np
import torch
from mmcv.utils import print_log
from torch.utils.data import Dataset

from mmaction.core import (mean_average_precision, mean_class_accuracy,
                    mmit_mean_average_precision, top_k_accuracy)

@DATASETS.register_module()
class TSP_Multiviews_RawframeDataset(RawframeDataset):

    def load_annotations(self):
        """Load annotation file to get video information."""
        if self.ann_file.endswith('.json'):
            return self.load_json_annotations()
        video_infos = []
        with open(self.ann_file, 'r') as fin:
            for line in fin:
                line_split = line.strip().split()
                video_info = {}
                idx = 0
                # idx for frame_dir
                frame_dir = line_split[idx]
                if self.data_prefix is not None:
                    frame_dir = osp.join(self.data_prefix, frame_dir)
                video_info['frame_dir'] = frame_dir
                idx += 1
                if self.with_offset:
                    # idx for offset and total_frames
                    video_info['offset'] = int(line_split[idx])
                    video_info['total_frames'] = int(line_split[idx + 1])
                    idx += 2
                else:
                    # idx for total_frames
                    video_info['total_frames'] = int(line_split[idx])
                    idx += 1
                # idx for label[s]
                label = line_split[idx]
                video_info['label'] = int(label)
                
                idx += 1
                video_info['frame_dir_rear'] = osp.join(self.data_prefix, line_split[idx])

                idx += 1
                video_info['frame_dir_right'] = osp.join(self.data_prefix, line_split[idx])

                video_infos.append(video_info)

        return video_infos
