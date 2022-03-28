import numpy as np
import random
import math
import os
from mmaction.datasets.builder import PIPELINES
from mmaction.datasets.pipelines import SampleFrames
from mmaction.datasets.pipelines import SampleAVAFrames
from mmaction.datasets.pipelines import DecordInit, DecordDecode

@PIPELINES.register_module()
class InvalidBoxFilter():
    """Filter invalid proposals and ground truths.
    Invalid sample is zero height/width."""

    def __init__(self, min_area=0):
        self.min_area = min_area

    def __call__(self, results):
        # filter invalid proposals
        if 'proposals' in results:
            valid_inds = self.get_valid_inds(results['proposals'])
            if np.any(valid_inds):
                results['proposals'] = results['proposals'][valid_inds]
            else:
                results['proposals'] = np.array(
                    [[0, 0, 1, 1]], dtype=np.float32)

        # filter invalid gt boxes
        if 'gt_bboxes' in results:
            valid_inds = self.get_valid_inds(results['gt_bboxes'])
            if np.any(valid_inds):
                results['gt_bboxes'] = results['gt_bboxes'][valid_inds]
                if 'gt_labels' in results:
                    results['gt_labels'] = results['gt_labels'][valid_inds]
            else:
                results['gt_bboxes'] = np.array(
                    [[0, 0, 1, 1]], dtype=np.float32)
                if 'gt_labels' in results:
                    results['gt_labels'] = np.eye(
                        81)[np.array([0])]
                    results['gt_labels'] = results['gt_labels'].astype(
                        np.float32)

        return results

    def get_valid_inds(self, bboxes):
        valid_inds = []
        for box in bboxes:
            valid_inds.append(self.get_area(box) > self.min_area)
        valid_inds = np.array(valid_inds)
        return valid_inds

    def get_area(self, box):
        box = box.astype(int)
        area = (box[2] - box[0]) * (box[3] - box[1])
        return area


@PIPELINES.register_module()
class IdxSampleAVAFrames(SampleAVAFrames):
    """Adding:
        - center_idx to results dict"""

    def __call__(self, results):
        fps = results['fps']
        timestamp = results['timestamp']
        timestamp_start = results['timestamp_start']
        shot_info = results['shot_info']

        center_index = fps * (timestamp - timestamp_start) + 1

        skip_offsets = np.random.randint(
            -self.frame_interval // 2, (self.frame_interval + 1) // 2,
            size=self.clip_len)
        frame_inds = self._get_clips(center_index, skip_offsets, shot_info)
        results['center_idx'] = center_index
        results['frame_inds'] = np.array(frame_inds, dtype=np.int)
        results['clip_len'] = self.clip_len
        results['frame_interval'] = self.frame_interval
        results['num_clips'] = 1
        results['crop_quadruple'] = np.array([0, 0, 1, 1], dtype=np.float32)
        return results


@PIPELINES.register_module()
class CloneOriginal():
    """Clone original items"""

    def __init__(self, keys=['imgs', 'proposals', 'gt_bboxes']):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            new_key = key + '_org'
            results[new_key] = results[key].copy()
        return results

@PIPELINES.register_module()
class SampleFramesNonOverlap(SampleFrames):
    """Sample frames from the video with non-overlap between clip
    """
    def _get_clips(self, num_frames):
        """Get clip offsets for buffer storing mode
        """
        ori_clip_len = self.clip_len * self.frame_interval
        if self.num_clips > 0:
            clip_offsets = np.arange(self.num_clips) * ori_clip_len
        else:
            clip_offsets = np.zeros((1, ), dtype = np.int)

        return clip_offsets

    def _sample_clips(self, num_frames):
        return self._get_clips(num_frames)


@PIPELINES.register_module()
class RandSampleFrames(SampleFrames):
    def __init__(self, range=(4,6), dataset_name='K400', *args, **kwargs):
        self.range = range
        self.dataset_name = dataset_name
        super(RandSampleFrames, self).__init__(*args, **kwargs)

    def __call__(self, results):
        
        if self.dataset_name=='sth2sth':
            total_frames = results['total_frames']
            real_interval = total_frames/self.clip_len
            max_interval = random.randint(self.range[0],self.range[1])
            if real_interval > max_interval:
                self.frame_interval = max_interval
            else:
                self.frame_interval = math.floor(real_interval) if real_interval> 1 else 1

        else:
            self.frame_interval = random.randint(self.range[0],self.range[1])
        
        return super().__call__(results)

@PIPELINES.register_module()
class AdaptSampleFrames(SampleFrames):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_interval = self.frame_interval

    def __call__(self, results):

        total_frames = results['total_frames']
        real_interval = total_frames//self.clip_len
        self.frame_interval = min(max(1,real_interval),self.max_interval)
       
        return super().__call__(results)
    
    
@PIPELINES.register_module()
class CcDecordInit(DecordInit):
    
    def __init__(self,*args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def __call__(self, results):
                
        results_rearview = results.copy()
        results_rearview['filename'] = results_rearview['filename'].replace("Dashboard","Rear_view")
        
        results_rightside = results.copy()
        results_rightside['filename'] = results_rightside['filename'].replace("Dashboard","Rightside_window")
        if not os.path.exists(results_rightside['filename']):
            results_rightside['filename']=results_rightside['filename'].replace("Rightside","Right_side")
        results = super().__call__(results)
        results_rearview = super().__call__(results_rearview)
        results_rightside = super().__call__(results_rightside)
        results['other_view'] = [results_rearview, results_rightside]
        return results
    
    
@PIPELINES.register_module()
class CcDecordDecode(DecordDecode):
    def __init__(self, crop_drive=False, *args, **kwargs):
        self.crop_drive=crop_drive
        super().__init__(*args, **kwargs)
        
    def __call__(self, results):
        #----- rear view
        results_rearview = results.copy()
        results_rearview['filename'] = results_rearview['other_view'][0]['filename']
        results_rearview['video_reader'] = results_rearview['other_view'][0]['video_reader']
        results_rearview['total_frames'] = results_rearview['other_view'][0]['total_frames']
        frame_ids = results_rearview['frame_inds']
        
        desire_ids = np.where(frame_ids >= results_rearview['total_frames'])[0]
        frame_ids[desire_ids] = results_rearview['total_frames']-1
        results_rearview['frame_inds'] = frame_ids
        results_rearview['other_view']=None
        
        #----- rightside view
        results_rightside = results.copy()
        results_rightside['filename'] = results_rightside['other_view'][1]['filename']
        results_rightside['video_reader'] = results_rightside['other_view'][1]['video_reader']
        frame_ids = results_rightside['frame_inds']
        results_rightside['total_frames'] = results_rightside['other_view'][1]['total_frames']
        
        desire_ids = np.where(frame_ids >= results_rightside['total_frames'])[0]
        frame_ids[desire_ids] = results_rightside['total_frames']-1
        results_rightside['frame_inds'] = frame_ids
        results_rightside['other_view']=None
        
        
        results['other_view']=None
        
        results = super().__call__(results)
        results_rearview = super().__call__(results_rearview)
        results_rightside = super().__call__(results_rightside)
        
        if self.crop_drive:
            height, width = results['img_shape']
            results['imgs'] = [tmp[:, 380:width-120] for tmp in results['imgs']]
            results_rearview['imgs'] = [tmp[:, 750:] for tmp in results_rearview['imgs']]
            results_rightside['imgs'] = [tmp[:, 750:] for tmp in results_rightside['imgs']]
            
        # combine results['imgs']= [Dashboard, rearview, rightside]
        results['imgs'] = results['imgs'] + results_rearview['imgs'] + results_rightside['imgs']
        results["view_name"] = [results['filename'], results_rearview['filename'], results_rightside['filename']]
        del results_rearview, results_rightside
        return results
        