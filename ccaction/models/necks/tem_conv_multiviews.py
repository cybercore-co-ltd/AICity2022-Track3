from mmaction.models.builder import NECKS
import torch.nn as nn
from mmcv.cnn import ConvModule, constant_init, xavier_init
from einops import rearrange

@NECKS.register_module()
class Tem_Conv_multiviews(nn.Module):
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
        self.numclip_conv = ConvModule(
                                fc_chn,
                                fc_chn,
                                kernel_size=(1,3),
                                )
        self.view_conv = ConvModule(
                                fc_chn,
                                fc_chn,
                                kernel_size=3,
                                )
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                xavier_init(m, distribution='uniform')
            if isinstance(m, nn.BatchNorm3d):
                constant_init(m, 1)

    def forward(self, x, num_segs):
        x = self.temp_conv(x) #B,C,3H,3W --> B,3C,H,W
        if isinstance(x, tuple):
                shapes = [y.shape for y in x]
                assert 1 == 0, f'x is tuple {shapes}'
        x = self.avg_pool(x)

        x = x.reshape(x.shape[0], -1) # [bs, in_channels]
        x = x.reshape((-1, num_segs) + x.shape[1:])
        
        x = rearrange(x, 'b (v l) f -> b f v l', v=3) # [batch feature view num_clip]
        
        x = self.numclip_conv(x) # [batch, feature, 3,3]
        
        x = self.view_conv(x) # [N, in_channels, 1, 1]
        
        x = x.view(x.size(0), -1)
        

        return x
