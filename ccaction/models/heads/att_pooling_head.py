from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn.bricks import DropPath
from mmcv.cnn.utils.weight_init import trunc_normal_, trunc_normal_init
from einops import rearrange

from mmaction.models.builder import HEADS 
from mmaction.models.heads.base import BaseHead

# Code is adopted from https://github.com/facebookresearch/deit/blob/main/patchconvnet_models.py
class Mlp(nn.Module):
    def __init__(self,
                in_features: int,
                hidden_features: Optional[int] = None,
                out_features: Optional[int] = None,
                act_layer: nn.Module = nn.GELU,
                drop: float = 0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        return x

class Learned_Aggregation_Layer(nn.Module):
    """ Implement Single Head Attention"""
    def __init__(self,
                dim: int,
                qkv_bias: bool = False,
                qk_scale: Optional[float] = None,
                attn_drop: float = 0.0,
                proj_drop: float = 0.0):
        super().__init__()
        self.scale = qk_scale or dim**-0.5
        
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        q = (self.scale*self.q(x[:, 0])).unsqueeze(1)
        k = self.k(x)
        v = self.v(x)

        attn = q @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x_cls = attn @ v
        x_cls = self.proj(x_cls)
        x_cls = self.proj_drop(x_cls)

        return x_cls

class AttnPooling(nn.Module):
    def __init__(
        self,
        dim: int,
        mlp_ratio: float = 4.0,
        qkv_bias: bool = False,
        qk_scale: Optional[float] = None,
        drop: float = 0.0,
        drop_path: float = 0.0,
        attn_drop: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer=nn.LayerNorm,
        init_values: float = 1e-4,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Learned_Aggregation_Layer(
            dim, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.gamma_1 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.gamma_2 = nn.Parameter(init_values * torch.ones((dim)), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor, x_cls: torch.Tensor):
        u = torch.cat((x_cls, x), dim=1)
        x_cls = x_cls + self.gamma_1 * self.attn(self.norm1(u))
        x_cls = x_cls + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x_cls)))
        return x_cls

@HEADS.register_module()
class AttnPoolingHead(BaseHead):
    def __init__(self,
                num_classes,
                in_channels,
                attn_pool_cfg=dict(
                    mlp_ratio=4.0,
                    qkv_bias=False,
                    qk_scale=None,
                    drop=0.0,
                    drop_path=0.0,
                    attn_drop=0.0,
                    init_values= 1e-4,
                ),
                loss_cls=dict(type='CrossEntropyLoss'),
                init_std=0.02,
                **kwargs):
        super().__init__(num_classes, in_channels, loss_cls=loss_cls, **kwargs)
        self.init_std = init_std

        self.cls_token = nn.Parameter(torch.zeros(1, 1, in_channels))
        self.pooling = AttnPooling(dim=in_channels, **attn_pool_cfg)
        self.norm = nn.LayerNorm(in_channels)
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)
        self.init_weights()

    def init_weights(self):
        """Initiate the parameters from scratch."""
        def _init_weights(m):
            if isinstance(m, nn.Linear):
                trunc_normal_(m.weight, std=self.init_std)
                if isinstance(m, nn.Linear) and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.LayerNorm):
                nn.init.constant_(m.bias, 0)
                nn.init.constant_(m.weight, 1.0)
        self.apply(_init_weights)
        trunc_normal_init(self.fc_cls, std=self.init_std)
        trunc_normal_(self.cls_token, std=self.init_std)

    def forward(self, x, num_segs):
        # Flatten the Segments
        x = rearrange(x, '(b t) c h w -> b (t h w) c', t=num_segs)
        B = x.shape[0]
        # Attention Pooling 
        cls_tokens = self.cls_token.expand(B, -1, -1)
        cls_tokens = self.pooling(x, cls_tokens)
        # Normalization
        x = torch.cat((cls_tokens, x), dim=1)
        x = self.norm(x)
        # Classification 
        cls_score = self.fc_cls(x[:,0])
        return cls_score

if __name__ == '__main__':
    input = torch.randn(4, 768, 21, 21)
    model = AttnPoolingHead(
                num_classes=400, 
                in_channels=768,
                attn_pool_cfg=dict(mlp_ratio=2))
        
    out = model(input, num_segs=2)
    print(out.shape)