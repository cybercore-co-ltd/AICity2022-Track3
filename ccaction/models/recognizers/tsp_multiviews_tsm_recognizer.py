from mmaction.models.builder import RECOGNIZERS
from .vidconv_recognizer import VidConvRecognizer
from .tsp_recognizer import TSPRecognizer
from .tsp_multiviews_tsm_recognizer import TSP_Multiviews_Recognizer
from mmcv.runner import auto_fp16
import torch.nn.functional as F
import torch
from einops import rearrange
import torch.nn as nn

class TemporalShift(nn.Module):
    """Temporal shift module.
    This module is proposed in
    `TSM: Temporal Shift Module for Efficient Video Understanding
    <https://arxiv.org/abs/1811.08383>`_
    Args:
        net (nn.module): Module to make temporal shift.
        num_segments (int): Number of frame segments. Default: 3.
        shift_div (int): Number of divisions for shift. Default: 8.
    """

    def __init__(self, num_segments=5, shift_div=8):
        super().__init__()
        self.num_segments = num_segments
        self.shift_div = shift_div

    def forward(self, x):
        """Defines the computation performed at every call.
        Args:
            x (torch.Tensor): The input data.
        Returns:
            torch.Tensor: The output of the module.
        """
        x = self.shift(x, self.num_segments, shift_div=self.shift_div)
        return x

    @staticmethod
    def shift(x, num_segments, shift_div=3):
        """Perform temporal shift operation on the feature.
        Args:
            x (torch.Tensor): The input feature to be shifted.
            num_segments (int): Number of frame segments.
            shift_div (int): Number of divisions for shift. Default: 3.
        Returns:
            torch.Tensor: The shifted feature.
        """
        # [N, C, H, W]
        n, c, h, w = x.size()

        # [N // num_segments, num_segments, C, H*W]
        # can't use 5 dimensional array on PPL2D backend for caffe
        x = x.view(-1, num_segments, c, h * w)

        # get shift fold
        fold = c // shift_div

        # split c channel into three parts:
        # left_split, mid_split, right_split
        left_split = x[:, :, :fold, :]
        mid_split = x[:, :, fold:2 * fold, :]
        right_split = x[:, :, 2 * fold:, :]

        # can't use torch.zeros(*A.shape) or torch.zeros_like(A)
        # because array on caffe inference must be got by computing

        # shift left on num_segments channel in `left_split`
        zeros = left_split - left_split
        blank = zeros[:, :1, :, :]
        left_split = left_split[:, 1:, :, :]
        left_split = torch.cat((left_split, blank), 1)

        # shift right on num_segments channel in `mid_split`
        zeros = mid_split - mid_split
        blank = zeros[:, :1, :, :]
        mid_split = mid_split[:, :-1, :, :]
        mid_split = torch.cat((blank, mid_split), 1)

        # right_split: no shift

        # concatenate
        out = torch.cat((left_split, mid_split, right_split), 2)

        # [N, C, H, W]
        # restore the original dimension
        return out.view(n, c, h, w)

@RECOGNIZERS.register_module()
class TSP_Multiviews_TSM_Recognizer(TSP_Multiviews_Recognizer):

    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.tsm = TemporalShift()

    @auto_fp16()
    def forward_train(self, imgs, labels,**kwargs):
        # Input shape [BS, views*Seg*T, 3, H, W]
        batches = imgs.shape[0]
        num_segs = imgs.shape[1]//self.clip_frames
        # imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        frames = imgs.shape[1] // 3
        imgs =imgs.reshape((batches,3, frames) + imgs.shape[2:])
        imgs = torch.transpose(imgs, 0,1)
        x = []
        for idx in range(len(imgs)):
            view = imgs[idx]
            view = view.reshape((-1, ) + view.shape[2:])
            view = self.extract_feat(view)
            view = self.tsm(view)
            x.append(view)
        x = torch.stack(x)
        x = rearrange(x, 'v (b l) c h w -> b (v l) c h w', b=batches)
        _, _, channel, height, width= x.shape
        x = x.reshape(-1, channel, height,width)

        return self.forward_train_(x, labels, num_segs=num_segs, **kwargs)
    