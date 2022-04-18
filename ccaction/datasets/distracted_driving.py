from doctest import debug_src
import warnings
import numpy as np
from collections import OrderedDict
import copy

from mmaction.datasets import DATASETS, ActivityNetDataset
from mmaction.core import average_recall_at_avg_proposals
import mmcv
@DATASETS.register_module()
class DistractedDriving(ActivityNetDataset):
    """Track 3 Datasets
    """
    CLASS_MAPPING ={
                    'Drinking':1, \
                    'Phone_Call_right':2, \
                    'Phone_Call_left':3, \
                    'Eating':4, \
                    'Text_Right':5, \
                    'Text_Left':6, \
                    'Hair_makeup':7, \
                    'Reaching_behind':8, \
                    'Adjust_control_panel':9, \
                    'Pick_up_from_floor_Driver':10, \
                    'Pick_up_from_floor_Passenger':11, \
                    'Talk_to_passenger_at_the_right':12, \
                    'Talk_to_passenger_at_backseat':13, \
                    'yawning':14, \
                    'Hand_on_head':15, \
                    'Singing_with_music':16, \
                    'shaking_or_dancing_with_music':17}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _pairwise_temporal_matching(self, candidate_segments, target_segments, thr_proposal = 0.0,  margin_temporal=1, thr_cls = 0.1):
        #candidate_segments: mx4(start, end, proposal_sc, cls)
        #target_segments: nx3(start, end, cls)
        _tp, _fp, _fn = [], [], []
        if candidate_segments.shape[0]==0:
            _fn.extend([_ for _ in target_segments])
            return [], [], _fn, target_segments.shape[0]
        candidate_segments = candidate_segments[candidate_segments[:,2]>=thr_proposal]
        if candidate_segments.shape[0]==0:
            _fn.extend([_ for _ in target_segments])
            return [], [], _fn, target_segments.shape[0]
        n, m = target_segments.shape[0], candidate_segments.shape[0]
        # if n==0:
        #     _fn.extend([_ for _ in candidate_segments])
        #     return [], [], _fn, 0
        # print('Number of prediction in eval', m)
        for i in range(m):
            if candidate_segments.shape[1]==4:
                s_pred,e_pred, _, cls_pred = candidate_segments[i, :]
            else:
                s_pred,e_pred, cls_pred = candidate_segments[i, :]
            for j in range(n):
                s_gt, e_gt, cls_name = target_segments[j, :]
                duration = float(e_gt)-float(s_gt)
                cls_gt = int(self.CLASS_MAPPING[cls_name])
                if int(cls_pred)==cls_gt:
                    if abs(float(s_pred)-float(s_gt))<=margin_temporal \
                        and abs(float(e_pred)-float(e_gt))<=margin_temporal:
                        _tp.append(j)
                    else:
                        _fp.append(j)
        # print(_tp, _fp)
        # print(set(_tp+_fp))
        return _tp, _fp, [1]*(n-len(set(_tp+_fp))), n

    def _get_f1_score(self,ground_truths, proposals):
        #ground_truths     |dict of num_videos: start, end, cls
        #proposals         |list of num_videos: Nx4(start,end,score, cls)
        sum=0
        for k,v in proposals.items():
            sum+=len(v)
        # print('Number of samples', sum)

        total_tp, total_fp, total_fn = [], [], []
        num_gts=0
        for video_id in ground_truths:
            # Get proposals for this video.
            _proposals_video_id = proposals[video_id]
            _ground_truth_video_id = ground_truths[video_id]
            _tp, _fp, _fn, _n_gt = self._pairwise_temporal_matching(_proposals_video_id, _ground_truth_video_id)
            # print(len(_tp), len(_fp), len(_fn), _n_gt )
            total_tp.extend(_tp)
            total_fp.extend(_fp)
            total_fn.extend(_fn)
            num_gts+=_n_gt

        num_tp, num_fp, num_fn = len(total_tp), len(total_fp), len(total_fn)
        # print(num_gts, num_tp, num_fp, num_fn)
        try:
            recall = num_tp/(num_tp+num_fn)
        except:
            recall = 0
        try:
            precision = num_tp/(num_tp+num_fp)
        except:
            precision = 0
        try:
            f1_score = 2*recall*precision/(precision+recall)
        except:
            f1_score = 0
        return recall, precision, f1_score
        
    def evaluate(
            self,
            results,
            metrics=['AR@AN', 'F1'],
            metric_options={
                'AR@AN':
                dict(
                    max_avg_proposals=100,
                    temporal_iou_thresholds=np.linspace(0.5, 0.95, 10))
            },
            logger=None,
            **deprecated_kwargs):
        """
        Return the F1-score:
        +-------------------+-------+-----------------+-----------------+
        | metric\conditions | label |    start_time   |     end_time    |
        +-------------------+-------+-----------------+-----------------+
        |         TP        | True  | abs(pred-gt)<1s | abs(pred-gt)<1s |
        +-------------------+-------+-----------------+-----------------+
        |         FP        | False | abs(pred-gt)<1s | abs(pred-gt)<1s |
        +-------------------+-------+-----------------+-----------------+
        |         FN        |   X   | abs(pred-gt)>1s | abs(pred-gt)>1s |
        +-------------------+-------+-----------------+-----------------+
        """
        # Protect ``metric_options`` since it uses mutable value as default
        metric_options = copy.deepcopy(metric_options)

        if deprecated_kwargs != {}:
            warnings.warn(
                'Option arguments for metrics has been changed to '
                "`metric_options`, See 'https://github.com/open-mmlab/mmaction2/pull/286' "  # noqa: E501
                'for more details')
            metric_options['AR@AN'] = dict(metric_options['AR@AN'],
                                           **deprecated_kwargs)

        if not isinstance(results, list):
            raise TypeError(f'results must be a list, but got {type(results)}')
        assert len(results) == len(self), (
            f'The length of results is not equal to the dataset len: '
            f'{len(results)} != {len(self)}')

        metrics = metrics if isinstance(metrics, (list, tuple)) else [metrics]
        allowed_metrics = ['AR@AN', 'F1']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')

        eval_results = OrderedDict()
        ground_truth = self._import_ground_truth()
        proposal, num_proposals = self._import_proposals(results)

        for metric in metrics:
            if metric=='AR@AN':
                temporal_iou_thresholds = metric_options.setdefault(
                    'AR@AN', {}).setdefault('temporal_iou_thresholds',
                                            np.linspace(0.5, 0.95, 10))
                max_avg_proposals = metric_options.setdefault(
                    'AR@AN', {}).setdefault('max_avg_proposals', 100)
                if isinstance(temporal_iou_thresholds, list):
                    temporal_iou_thresholds = np.array(temporal_iou_thresholds)

                recall, _, _, auc = (
                    average_recall_at_avg_proposals(
                        ground_truth,
                        proposal,
                        num_proposals,
                        max_avg_proposals=max_avg_proposals,
                        temporal_iou_thresholds=temporal_iou_thresholds))
                eval_results['auc'] = auc
                eval_results['AR@1'] = 100*np.mean(recall[:, 0])
                eval_results['AR@5'] = 100*np.mean(recall[:, 4])
                eval_results['AR@10'] = 100*np.mean(recall[:, 9])
                eval_results['AR@100'] = 100*np.mean(recall[:, 99])
            elif metric=='F1':
                recall_track3, precision_track3, f1_score_track3 = self._get_f1_score(ground_truth, proposal)
                eval_results['F1_track3'] = 100*f1_score_track3
                eval_results['recall_track3'] = 100*recall_track3
                eval_results['precision_track3'] = 100*precision_track3
            else:
                eval_results=None

        return eval_results
    
    # def self._import_ground_truth():
    def _import_proposals(self, results):
        """Read predictions from results."""
        proposals = {}
        num_proposals = 0
        for result in results:
            video_id = result['video_name']
            this_video_proposals = []
            for proposal in result['proposal_list']:
                t_start, t_end = proposal['segment']
                score = proposal['score']
                #prediction from 0..16
                if 'label' in proposal:
                    label = proposal['label']+1
                    #1x4(start, end, proposal_sc, cls_idx)
                    this_video_proposals.append([t_start, t_end, score, label])
                else:
                    #1x3(start, end, proposal_sc)
                    this_video_proposals.append([t_start, t_end, score])
                num_proposals += 1
            proposals[video_id] = np.array(this_video_proposals)
        return proposals, num_proposals
    
    def proposals2json(self, results, show_progress=False):
        """Convert all proposals to a final dict(json) format.

        Args:
            results (list[dict]): All proposals.
            show_progress (bool): Whether to show the progress bar.
                Defaults: False.

        Returns:
            dict: The final result dict. E.g.

            .. code-block:: Python

                dict(video-1=[dict(segment=[1.1,2.0]. score=0.9),
                              dict(segment=[50.1, 129.3], score=0.6)])
        """
        result_dict = {}
        print('Convert proposals to json format')
        if show_progress:
            prog_bar = mmcv.ProgressBar(len(results))
        for result in results:
            video_name = result['video_name']
            result_dict[video_name] = result['proposal_list']
            if show_progress:
                prog_bar.update()
        return result_dict
    
    def _import_ground_truth(self):
        """Read ground truth data from video_infos."""
        ground_truth = {}
        for video_info in self.video_infos:
            video_id = video_info['video_name']
            this_video_ground_truths = []
            for ann in video_info['annotations']:
                t_start, t_end = ann['segment']
                label = ann['label']
                this_video_ground_truths.append([t_start, t_end, label])
            ground_truth[video_id] = np.array(this_video_ground_truths)
        return ground_truth

    def load_annotations(self):
        """Load the annotation according to ann_file into video_infos."""
        video_infos = []
        anno_database = mmcv.load(self.ann_file)['database']
        for video_name in anno_database:
            video_info = anno_database[video_name]
            if video_info['subset']!='validation': continue
            #just load validation test
            video_info['video_name'] = video_name
            video_infos.append(video_info)
        return video_infos