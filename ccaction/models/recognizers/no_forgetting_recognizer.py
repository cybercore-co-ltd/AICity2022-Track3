from mmaction.models.builder import RECOGNIZERS
from mmaction.models import Recognizer2D
from mmcv.runner import auto_fp16
import torch 
import torch.nn as nn
from einops import rearrange
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmaction.models import build_model, build_head
import torch.nn.functional as F

@RECOGNIZERS.register_module()
class NoFogettingRecognizer(Recognizer2D):
    def __init__(self,kl_head=None,off_model=None,*args,**kwargs):
        super().__init__(*args,**kwargs)

        self.kl_head = build_head(kl_head) if kl_head else None

        self.off_cfg = Config.fromfile(off_model['config'])
        self.off_model = build_model(
            self.off_cfg.model,
            train_cfg=self.off_cfg.get('train_cfg'),
            test_cfg=self.off_cfg.get('test_cfg'))
        load_checkpoint(self.off_model, off_model['checkpoint'])
        self.off_model.eval()

        self.clip_frames = self.backbone.clip_frames
        # self.kl_loss = nn.KLDivLoss(log_target=True)
        # self.distill_loss = nn.CrossEntropyLoss()
        
    def forward_train_(self, imgs, labels, num_segs=1,**kwargs):
        x = self.extract_feat(imgs)

        if self.kl_head is not None:
            kl_logit = self.kl_head(x, num_segs)
            kl_logit = F.softmax(kl_logit)

        if self.with_neck:
            x = self.neck(x)

        cls_score = self.cls_head(x, num_segs)
        gt_labels = labels.squeeze().long()
        loss = self.cls_head.loss(cls_score, gt_labels,  **kwargs)

        # off model
        with torch.no_grad():
            batches = imgs.shape[0]//num_segs//self.clip_frames
            off_logit = self.off_model.forward_offline(imgs, num_segs, batches)
            off_logit = F.softmax(off_logit)
        # loss_kl = self.distill_loss(kl_logit, off_logit)
        loss['loss_distill'] = -0.02*torch.sum(off_logit*torch.log(kl_logit), dim=1).mean()
        
        return loss

    @auto_fp16()
    def forward_train(self, imgs, labels,**kwargs):
        # Input shape [BS, Seg*T, 3, H, W]
        num_segs = imgs.shape[1]//self.clip_frames
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        return self.forward_train_(imgs, labels, num_segs=num_segs, **kwargs)
        
    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        batches = imgs.shape[0]
        num_segs = imgs.shape[1]//self.clip_frames
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        x = self.extract_feat(imgs)
        
        # x: list of T elements. each element have size NCHW
        if self.with_neck:
            x = self.neck(x)

        # When using `TSNHead` or `TPNHead`, shape is [batch_size, num_classes]
        # When using `TSMHead`, shape is [batch_size * num_crops, num_classes]
        # `num_crops` is calculated by:
        #   1) `twice_sample` in `SampleFrames`
        #   2) `num_sample_positions` in `DenseSampleFrames`
        #   3) `ThreeCrop/TenCrop/MultiGroupCrop` in `test_pipeline`
        #   4) `num_clips` in `SampleFrames` or its subclass if `clip_len != 1`

        # should have cls_head if not extracting features
        score = self.cls_head(x, num_segs)
        # if isinstance(cls_score, tuple):
        #     cls_score = cls_score[0]
        # assert cls_score.size()[0] % batches == 0
        # calculate num_crops automatically
        cls_score = self.average_clip(score[0],
                                    score[0].size()[0] // batches)
        actioness_score = self.average_clip(score[1],
                                    score[1].size()[0] // batches)

        return (cls_score.cpu().numpy(), actioness_score.cpu().numpy())

    @auto_fp16()
    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        return self._do_test(imgs)

    def _do_fcn_test(self, imgs):
        # [N, num_crops * num_segs, C, H, W] ->
        # [N * num_crops * num_segs, C, H, W]
        batches = imgs.shape[0]
        num_segs = imgs.shape[1]//self.clip_frames
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        if self.test_cfg.get('flip', False):
            imgs = torch.flip(imgs, [-1])
        
        # input imgs: NCTHW 1, 3, 9, 192, 192
        x = []
        for i in range(imgs.shape[2]):
            x.append(self.extract_feat(imgs[:,:,i,:,:]))
        
        # x: list of T elements. each element have size NCHW
        if self.with_neck:
            x = self.neck(x)
        else:
            x = x.reshape((-1, num_segs) +
                          x.shape[1:]).transpose(1, 2).contiguous()

        # When using `TSNHead` or `TPNHead`, shape is [batch_size, num_classes]
        # When using `TSMHead`, shape is [batch_size * num_crops, num_classes]
        # `num_crops` is calculated by:
        #   1) `twice_sample` in `SampleFrames`
        #   2) `num_sample_positions` in `DenseSampleFrames`
        #   3) `ThreeCrop/TenCrop/MultiGroupCrop` in `test_pipeline`
        #   4) `num_clips` in `SampleFrames` or its subclass if `clip_len != 1`
        cls_score = self.cls_head(x, fcn_test=True)
        if isinstance(cls_score, tuple):
            cls_score = cls_score[0]

        assert cls_score.size()[0] % batches == 0
        # calculate num_crops automatically
        cls_score = self.average_clip(cls_score,
                                      cls_score.size()[0] // batches)
        return cls_score

    @auto_fp16()   
    def forward_dummy(self, imgs, softmax=1):
        """Used for computing network FLOPs.

        See ``tools/analysis/get_flops.py``.

        Args:
            imgs (torch.Tensor): Input images.

        Returns:
            Tensor: Class score.
        """
        assert self.with_cls_head
        # batches = imgs.shape[0]
        # imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        # num_segs = imgs.shape[0] // batches
        num_segs=1
        batches=1
       # input imgs: NCTHW 1, 3, 9, 192, 192
        # x = []
        # # for i in range(imgs.shape[2]):
        # x.append(self.extract_feat(imgs))
        
        outs = self.extract_feat(imgs)
        # # x: list of T elements. each element have size NCHW
        # num_segs = 1
        if self.with_neck:
            outs = self.neck(outs)

        outs = self.cls_head(outs, num_segs)
        # if 1:
        #     outs = nn.functional.softmax(outs)
        return outs

    @auto_fp16()
    def forward_backbone(self, imgs):
        batches = imgs.shape[0]
        num_segs = imgs.shape[1]//self.clip_frames
        imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        x = self.extract_feat(imgs)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(-1, x.shape[1])
        return x