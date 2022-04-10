from mmaction.models.builder import RECOGNIZERS
from .vidconv_recognizer import VidConvRecognizer
from .tsp_recognizer import TSPRecognizer
from mmcv.runner import auto_fp16
import torch.nn.functional as F
import torch
from einops import rearrange

@RECOGNIZERS.register_module()
class TSP_Multiviews_Recognizer(TSPRecognizer):

    def forward_train_(self, x, labels, num_segs=1,**kwargs):
        if self.with_neck:
            x = self.neck(x, num_segs)

        cls_score = self.cls_head(x, 1)
        gt_labels = labels.squeeze().long()
        loss = self.cls_head.loss(cls_score, gt_labels,  **kwargs)

        return loss

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
            x.append(view)
        x = torch.stack(x)
        x = rearrange(x, 'v (b l) c h w -> b (v l) c h w', b=batches)
        _, _, channel, height, width= x.shape
        x = x.reshape(-1, channel, height,width)

        return self.forward_train_(x, labels, num_segs=num_segs, **kwargs)

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
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
            x.append(view)
        x = torch.stack(x)
        x = rearrange(x, 'v (b l) c h w -> b (v l) c h w', b=batches)
        _, _, channel, height, width= x.shape
        x = x.reshape(-1, channel, height,width)
        
        # x: list of T elements. each element have size NCHW
        if self.with_neck:
            x = self.neck(x, num_segs)

        # When using `TSNHead` or `TPNHead`, shape is [batch_size, num_classes]
        # When using `TSMHead`, shape is [batch_size * num_crops, num_classes]
        # `num_crops` is calculated by:
        #   1) `twice_sample` in `SampleFrames`
        #   2) `num_sample_positions` in `DenseSampleFrames`
        #   3) `ThreeCrop/TenCrop/MultiGroupCrop` in `test_pipeline`
        #   4) `num_clips` in `SampleFrames` or its subclass if `clip_len != 1`
        # import ipdb; ipdb.set_trace()
        # should have cls_head if not extracting features
        score = self.cls_head(x, 1)
        # if isinstance(cls_score, tuple):
        #     cls_score = cls_score[0]
        # assert cls_score.size()[0] % batches == 0
        # calculate num_crops automatically
        cls_score = self.average_clip(score[0],
                                    score[0].size()[0] // batches)
        actioness_score = self.average_clip(score[1],
                                    score[1].size()[0] // batches)

        return (cls_score.cpu().detach().numpy(), actioness_score.cpu().detach().numpy())

    @auto_fp16()
    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        return self._do_test(imgs)
    
    @auto_fp16()
    def forward_backbone(self, imgs):
        
        batches = imgs.shape[0]
        num_segs = imgs.shape[1]//self.clip_frames
       
        frames = imgs.shape[1] // 3
        imgs =imgs.reshape((batches,3, frames) + imgs.shape[2:])
        imgs = torch.transpose(imgs, 0,1)
        x = []
        for idx in range(len(imgs)):
            view = imgs[idx]
            view = view.reshape((-1, ) + view.shape[2:])
            view = self.extract_feat(view)
            x.append(view)
        x = torch.stack(x)
        x = rearrange(x, 'v (b l) c h w -> b (v l) c h w', b=batches)
        _, _, channel, height, width= x.shape
        x = x.reshape(-1, channel, height,width)
        
        # x: list of T elements. each element have size NCHW
        if self.with_neck:
            x = self.neck(x, num_segs)

        # import ipdb; ipdb.set_trace()        
        return x


    