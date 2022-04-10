from mmaction.models.builder import RECOGNIZERS
from mmaction.models import Recognizer2D
from mmcv.runner import auto_fp16
import torch
import torch.nn as nn
from einops import rearrange


@RECOGNIZERS.register_module()
class VidConvMultiViewRecognizer(Recognizer2D):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_frames = self.backbone.clip_frames

    def forward_train_(self, x, labels, num_segs=1, **kwargs):
        if self.with_neck:
            x = self.neck(x)

        cls_score = self.cls_head(x, num_segs)
        gt_labels = labels.squeeze().long()
        loss = self.cls_head.loss(cls_score, gt_labels,  **kwargs)

        return loss

    @auto_fp16()
    def forward_train(self, imgs, labels, **kwargs):
        # Input shape [BS, Seg*T, 3, H, W]
        batches = imgs.shape[0]
        num_segs = imgs.shape[1]//self.clip_frames
        # imgs = imgs.reshape((-1, ) + imgs.shape[2:])

        frames = imgs.shape[1] // 3
        imgs = imgs.reshape((batches, 3, frames) + imgs.shape[2:])
        imgs = torch.transpose(imgs, 0, 1)
        x = []
        for idx in range(len(imgs)):
            view = imgs[idx]
            view = view.reshape((-1, ) + view.shape[2:])
            view = self.extract_feat(view)
            x.append(view)
        x = torch.stack(x)
        x = rearrange(x, 'v (b l) c h w -> b (v l) c h w', b=batches)
        _, _, channel, height, width = x.shape
        x = x.reshape(-1, channel, height, width)

        return self.forward_train_(x, labels, num_segs=num_segs, **kwargs)

    def _do_test(self, imgs):
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        batches = imgs.shape[0]

        num_segs = imgs.shape[1]//self.clip_frames
        # imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        # x = self.extract_feat(imgs)
        frames = imgs.shape[1] // 3  # with 3 here is 3 views
        imgs = imgs.reshape((batches, 3, frames) + imgs.shape[2:])
        imgs = torch.transpose(imgs, 0, 1)
        x = []
        for idx in range(len(imgs)):
            view = imgs[idx]
            view = view.reshape((-1, ) + view.shape[2:])
            view = self.extract_feat(view)
            x.append(view)
        x = torch.stack(x)
        x = rearrange(x, 'v (b l) c h w -> b (v l) c h w', b=batches)
        _, _, channel, height, width = x.shape
        x = x.view(-1, channel, height, width)

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
        cls_score = self.cls_head(x, num_segs, batches)

        if isinstance(cls_score, tuple):
            cls_score = cls_score[0]
        assert cls_score.size()[0] % batches == 0
        # calculate num_crops automatically
        cls_score = self.average_clip(cls_score,
                                      cls_score.size()[0] // batches)
        return cls_score

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
            x.append(self.extract_feat(imgs[:, :, i, :, :]))

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
    def forward_test(self, imgs):
        """Defines the computation performed at every call when evaluation and
        testing."""
        if self.test_cfg.get('fcn_test', False):
            # If specified, spatially fully-convolutional testing is performed
            assert not self.feature_extraction
            assert self.with_cls_head
            return self._do_fcn_test(imgs).cpu().numpy()
        return self._do_test(imgs).cpu().numpy()

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
        """Defines the computation performed at every call when evaluation,
        testing and gradcam."""
        batches = imgs.shape[0]

        num_segs = imgs.shape[1]//self.clip_frames
        # imgs = imgs.reshape((-1, ) + imgs.shape[2:])
        # x = self.extract_feat(imgs)
        frames = imgs.shape[1] // 3  # with 3 here is 3 views
        imgs = imgs.reshape((batches, 3, frames) + imgs.shape[2:])
        imgs = torch.transpose(imgs, 0, 1)
        x = []
        for idx in range(len(imgs)):
            view = imgs[idx]
            view = view.reshape((-1, ) + view.shape[2:])
            view = self.extract_feat(view)
            x.append(view)
        x = torch.stack(x)
        x = rearrange(x, 'v (b l) c h w -> b (v l) c h w', b=batches)
        _, _, channel, height, width = x.shape
        x = x.view(-1, channel, height, width)

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
        outs = self.cls_head(x, num_segs, batches)
        if 1:
            outs = nn.functional.softmax(outs)
        return outs
