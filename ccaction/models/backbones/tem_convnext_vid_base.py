from cv2 import sqrt
import torch.nn as nn
import torch
from mmaction.models import BACKBONES
from einops import rearrange
from ccaction.models.backbones.convnext import ConvNeXt, Block, LayerNorm
from mmcv.cnn.bricks import DropPath
from mmcv.cnn.utils.weight_init import trunc_normal_
import math


class TempBlock(Block):

    def __init__(self, dim, kernel_size, dilation, drop_path=0., layer_scale_init_value=1e-6, temp_layer_scale_init_value=1e-2):
        super().__init__(dim, drop_path, layer_scale_init_value)
        self.kernel_size = kernel_size
        self.dwconv_tem = nn.Conv2d(dim, dim, kernel_size=kernel_size,
                                    groups=dim, dilation=dilation)  # depthwise conv
        self.gamma_tem = nn.Parameter(temp_layer_scale_init_value * torch.ones((dim)),
                                      requires_grad=True) if temp_layer_scale_init_value > 0 else None

    def forward(self, x):
        input = x

        # spatial branch
        sa = self.dwconv(x)
        sa = sa.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)

        # temporal branch
        tem = self.dwconv_tem(x)
        tem = tem.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        if self.gamma_tem is not None:
            tem = self.gamma_tem * tem
        tem = tem.repeat(1, self.kernel_size[0], self.kernel_size[1], 1)

        x = sa + tem
        x = self.norm(x)

        # MLP
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


@BACKBONES.register_module()
class ConvNextVidBaseTem(ConvNeXt):
    def __init__(self,
                 temporal_stack=(3, 3),
                 temporal_dilation=14,
                 add_tem_from_stage = 3,
                 *args, **kwargs):
        self.temporal_stack = temporal_stack
        self.clip_frames = temporal_stack[0]*temporal_stack[1]
        self.temporal_dilation = temporal_dilation
        self.add_tem_from_stage = add_tem_from_stage
        super().__init__(*args, **kwargs)

    def init_network(self):
        depths, dims, layer_scale_init_value = self.arch_settings[self.arch]
        # stem and 3 intermediate downsampling conv layers
        self.downsample_layers = nn.ModuleList()
        stem = nn.Sequential(
            nn.Conv2d(3, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        # 4 feature resolution stages, each consisting of multiple residual blocks
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(
            0, self.drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            if i < self.add_tem_from_stage-1:
                stage = nn.Sequential(
                    *[Block(dim=dims[i], drop_path=dp_rates[cur + j],
                            layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
                )
            else:
                stage = nn.Sequential(
                    *[TempBlock(dim=dims[i], kernel_size = self.temporal_stack, 
                                dilation=self.temporal_dilation // (2**(i - 2)), 
                                drop_path=dp_rates[cur + j],
                                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
                )

            self.stages.append(stage)
            cur += depths[i]



    def forward_early_net(self, x):
        """ EarlyNet constitues from Stem, Stage2, Stage3 and Downsample of stage4"""
        for i in range(2):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.downsample_layers[2](x)
        return x

    def forward_end_net(self, x):
        # shape of x - (BSx9)xCxhxw (for img_size=224, h=w=224/16=14)
        NT, C, H, W = x.shape
        x = rearrange(x, '(b t) c h w -> b c t h w', t=self.clip_frames)
        x = rearrange(x, 'b c (row col) h w -> b c (row h) (col w)', row=self.temporal_stack[0])
        x = self.stages[2](x)
        x = self.downsample_layers[3](x)
        x = self.stages[3](x)
        return x

    def forward(self, x):
        x = self.forward_early_net(x)
        x = self.forward_end_net(x)
        # x = self.final_stage(x)
        return x


if __name__ == '__main__':
    input = torch.randn(32, 3, 224, 224)
    model = ConvNextVidBaseTem(arch='tiny',
                               temporal_stack=(4,4),
                               temporal_dilation=14,
                               init_cfg=dict(type='Pretrained', checkpoint="tiny_1k"),)
    out = model(input)
    print(out.shape)
