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
class TSP_RawframeDataset(RawframeDataset):
    def evaluate(self,
                 results,
                 metrics='top_k_accuracy',
                 metric_options=dict(top_k_accuracy=dict(topk=(1, 5))),
                 logger=None,
                 **deprecated_kwargs):
        """Perform evaluation for common datasets.

        Args:
            results (tuple): Output results.
            metrics (str | sequence[str]): Metrics to be performed.
                Defaults: 'top_k_accuracy'.
            metric_options (dict): Dict for metric options. Options are
                ``topk`` for ``top_k_accuracy``.
                Default: ``dict(top_k_accuracy=dict(topk=(1, 5)))``.
            logger (logging.Logger | None): Logger for recording.
                Default: None.
            deprecated_kwargs (dict): Used for containing deprecated arguments.
                See 'https://github.com/open-mmlab/mmaction2/pull/286'.

        Returns:
            dict: Evaluation results dict.
        """
        # Protect ``metric_options`` since it uses mutable value as default
        metric_options = copy.deepcopy(metric_options)
        
        act_score = results[1::2]
        c_score = results[::2]
        
        actioness_score=[]
        for a in act_score:
            actioness_score.extend(a)
        
        cls_score = []
        for c in c_score:
            cls_score.extend(c)

        # import ipdb; ipdb.set_trace()
        if deprecated_kwargs != {}:
            warnings.warn(
                'Option arguments for metrics has been changed to '
                "`metric_options`, See 'https://github.com/open-mmlab/mmaction2/pull/286' "  # noqa: E501
                'for more details')
            metric_options['top_k_accuracy'] = dict(
                metric_options['top_k_accuracy'], **deprecated_kwargs)

        if not isinstance(cls_score, list):
            raise TypeError(f'cls_score must be a list, but got {type(cls_score)}')
        assert len(cls_score) == len(self), (
            f'The length of cls_score is not equal to the dataset len: '
            f'{len(cls_score)} != {len(self)}')

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = [
            'top_k_accuracy'
        ]

        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        eval_results = OrderedDict()
        gt_labels = [ann['label'] for ann in self.video_infos]

        for metric in metrics:
            msg = f'Evaluating {metric} ...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'top_k_accuracy':
                topk = metric_options.setdefault('top_k_accuracy',
                                                 {}).setdefault(
                                                     'topk', (1, 5))
                if not isinstance(topk, (int, tuple)):
                    raise TypeError('topk must be int or tuple of int, '
                                    f'but got {type(topk)}')
                if isinstance(topk, int):
                    topk = (topk, )

                # get foreground
                # import ipdb; ipdb.set_trace()
                arr_labels = np.array(gt_labels)
                foreground_idx = (arr_labels < 18) & (arr_labels >0)

                # actioness label
                actioness_labels = np.zeros_like(arr_labels)
                actioness_labels[foreground_idx] = 1
                actioness_labels[arr_labels==18]= 2
                actioness_labels[arr_labels==19]= 3
                
                # cls label
                # import ipdb; ipdb.set_trace()
                cls_labels = arr_labels[foreground_idx] - 1
                cls_score = np.array(cls_score)[foreground_idx]

                #save score and labels
                # score = [np.argmax(x) for x in cls_score]
                # np.save('score.npy', score)
                # np.save('label.npy', cls_labels)  
                # import ipdb; ipdb.set_trace()
                # actioness
                top_k_acc = top_k_accuracy(actioness_score, actioness_labels, topk)
                log_msg = []
                for k, acc in zip(topk, top_k_acc):
                    eval_results[f'top{k}_acc_actioness'] = acc
                    log_msg.append(f'\ntop{k}_acc_actioness\t{acc:.4f}')
                # log_msg = ''.join(log_msg)
                # print_log(log_msg, logger=logger)

                # cls
                top_k_acc = top_k_accuracy(cls_score.tolist(), cls_labels, topk)
                for k, acc in zip(topk, top_k_acc):
                    eval_results[f'top{k}_acc_cls'] = acc
                    log_msg.append(f'\ntop{k}_acc_cls\t{acc:.4f}')
                
                # log_msg = ''.join(log_msg)
                # print_log(log_msg, logger=logger)

                continue

        return eval_results