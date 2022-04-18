import torch.nn as nn
from mmcv.cnn.bricks import ConvModule
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
                 num_clip=5,
                 **kwargs):
        fc_chn = int(in_channels * expand_ratio)
        super().__init__(num_classes, fc_chn, loss_cls=loss_cls,
                         consensus=consensus, spatial_type=spatial_type,
                         dropout_ratio=dropout_ratio, init_std=init_std, **kwargs)
        self.temp_conv = nn.Sequential(
            ConvModule(
                in_channels,
                fc_chn,
                kernel_size=kernel_size,
                dilation=dilation,
                norm_cfg=norm_cfg,
                act_cfg=None
            ),)
        self.numclip_conv = ConvModule(
            fc_chn,
            fc_chn,
            kernel_size=(1, num_clip),
            groups=fc_chn,
            act_cfg=None,
        )
        self.view_conv = ConvModule(
            fc_chn,
            fc_chn,
            kernel_size=(3, 1),
            groups=fc_chn,
            norm_cfg=norm_cfg,
            act_cfg=None,
        )

    def forward(self, x, num_segs, batches=1):

        x = self.temp_conv(x)  # B,C,3H,3W --> B,3C,H,W
        if self.avg_pool is not None:
            if isinstance(x, tuple):
                shapes = [y.shape for y in x]
                assert 1 == 0, f'x is tuple {shapes}'
            x = self.avg_pool(x)
        # [N * num_segs, in_channels, 1, 1]

        _, channel, height, width = x.shape
        x = x.reshape(x.shape[0], -1)  # [bs, in_channels]
        x = x.reshape((-1, num_segs) + x.shape[1:])

        # [batch feature view num_clip]
        x = rearrange(x, 'b (v l) f -> b f v l', v=3)

        x = self.numclip_conv(x)  # [batch, feature, 3,1]

        x = self.view_conv(x)  # [N, in_channels, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)

        x = x.view(x.size(0), -1)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score
