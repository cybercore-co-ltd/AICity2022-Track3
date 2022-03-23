from typing import Optional
from numpy import zeros_like
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks import DropPath, ConvModule
from mmcv.cnn.utils.weight_init import trunc_normal_, trunc_normal_init
from einops import rearrange

from mmaction.models.builder import HEADS 
from mmaction.models.heads.tsn_head import TSNHead
from mmaction.models import builder

@HEADS.register_module()
class TSPHead(nn.Module):
    def __init__(self,
                    in_channels, 
                    expand_ratio=3,
                    kernel_size=3, 
                    dilation=7,
                    norm_cfg=None,
                    action_label_head = None,
                    actioness_head = None,
                    **kwargs):
        fc_chn = int(in_channels * expand_ratio)
        super().__init__()
        self.temp_conv = nn.Sequential(
                        ConvModule(
                        in_channels,
                        fc_chn,
                        kernel_size=kernel_size,
                        dilation=dilation,
                        norm_cfg=norm_cfg,
                        act_cfg=None
                        ),
                        nn.GELU())
        if action_label_head is None or actioness_head is None:
            raise ImportError('Please add head config')
        else:
            action_label_head['in_channels'] = fc_chn
            actioness_head['in_channels'] = fc_chn
            self.cls_branch = builder.build_head(action_label_head)
            self.actioness_branch = builder.build_head(actioness_head)
    
    def forward(self, x, num_segs):
        x = self.temp_conv(x) #B,C,3H,3W --> B,3C,H,W
        
        cls_score = self.cls_branch(x, num_segs)
        actioness_score = self.actioness_branch(x, num_segs)

        return (cls_score, actioness_score)

    def init_weights(self):
        """Initiate the parameters from scratch."""
        self.cls_branch.init_weights()
        self.actioness_branch.init_weights()
    
    def loss(self, scores, labels, **kwargs):
        foreground_idx = labels < self.cls_branch.num_classes 
        num_foreground = foreground_idx.sum()
        if num_foreground >0:
            cls_labels = labels[foreground_idx]
            cls_score = scores[0][foreground_idx]
            cls_loss = self.cls_branch.loss(cls_score, cls_labels, **kwargs)
        else:
            cls_loss = {}
        
        
        actioness_labels = torch.zeros_like(labels)
        actioness_labels[foreground_idx] = 1
        actioness_loss = self.actioness_branch.loss(scores[1], actioness_labels, **kwargs)

        cls_loss['loss_actioness']=actioness_loss['loss_cls']
            
        return cls_loss
        