import numpy as np
import random
import math
from mmaction.datasets.builder import PIPELINES
from mmaction.datasets.pipelines import SampleFrames
from mmaction.datasets.pipelines import SampleAVAFrames

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
    def __init__(self, range=(12,18), *args, **kwargs):
        self.range = range
        super(RandSampleFrames, self).__init__(*args, **kwargs)

    def __call__(self, results):
        total_frames = results['total_frames']
        real_interval = max(1,total_frames//self.clip_len)
        rand_interval = random.randint(self.range[0],self.range[1])
        self.frame_interval = min(rand_interval, real_interval)
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