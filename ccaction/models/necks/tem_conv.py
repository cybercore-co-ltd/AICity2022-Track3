from mmaction.models.builder import NECKS
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, xavier_init

@NECKS.register_module()
class Tem_Conv(nn.Module):
    def __init__(self,
                    in_channels, 
                    expand_ratio=3,
                    kernel_size=3, 
                    dilation=7,
                    norm_cfg=None,
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
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                xavier_init(m, distribution='uniform')
            if isinstance(m, nn.BatchNorm3d):
                constant_init(m, 1)

    def forward(self, x):
        x = self.temp_conv(x) #B,C,3H,3W --> B,3C,H,W
        

        return x
