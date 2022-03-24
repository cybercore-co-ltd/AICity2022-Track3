from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks import DropPath, ConvModule
from mmcv.cnn.utils.weight_init import trunc_normal_, trunc_normal_init
from einops import rearrange

from mmaction.models.builder import HEADS 
from mmaction.models.heads.tsn_head import TSNHead

@HEADS.register_module()
class MultiviewVidConvHead(TSNHead):
    def __init__(self,
                    num_classes,
                    in_channels, 
                    loss_cls=dict(type='CrossEntropyLoss'),
                    spatial_type='avg',
                    consensus=dict(type='AvgConsensus', dim=1),
                    dropout_ratio=0.4,
                    init_std=0.01,
                    expand_ratio=3,
                    kernel_size=3, 
                    dilation=7,
                    norm_cfg=None,
                    **kwargs):
        fc_chn = int(in_channels * expand_ratio)
        super().__init__(num_classes,fc_chn,loss_cls=loss_cls,
                        consensus=consensus,spatial_type=spatial_type, 
                        dropout_ratio=dropout_ratio,init_std=init_std,**kwargs)
        self.temp_conv = nn.Sequential(
                        ConvModule(
                        in_channels,
                        fc_chn,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        norm_cfg=norm_cfg,
                        act_cfg=None
                        ),)

        self.second_temp_conv = nn.Sequential(
                        ConvModule(
                        fc_chn*3,
                        fc_chn,
                        kernel_size=1,
                        dilation=1,
                        norm_cfg=norm_cfg,
                        act_cfg=None
                        ),
                        nn.GELU())
        
    def forward(self, x, num_segs):
        x = self.temp_conv(x) #B,C,3H,3W --> B,3C,H,W
        
        # stack 3 view
        num_segs=1
        bs = x.shape[0] // 3
        x = torch.cat([x[:bs], x[bs:2*bs], x[2*bs:]], axis=1)
        x = self.second_temp_conv(x)
        return super().forward(x, num_segs)