from mmaction.models.builder import RECOGNIZERS
from .vidconv_recognizer import VidConvRecognizer
from mmcv.runner import auto_fp16

@RECOGNIZERS.register_module()
class TSPRecognizer(VidConvRecognizer):
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