from mmaction.models.localizers import BMN
from mmaction.models.builder import LOCALIZERS
from ccaction.models.losses import InterConsistencyLoss, IntraConsistencyLoss

@LOCALIZERS.register_module()
class BMN_MR(BMN):
    """
    Bottom-Up Temporal Action Localization with Mutual Regularization
    From paper https://arxiv.org/abs/2002.07358
    """
    def __init__(self, intra_loss_weight=1, 
                    *args,**kwargs):
        super().__init__(*args,**kwargs)
        self.intraC_loss = IntraConsistencyLoss(loss_weight=intra_loss_weight)
        # self.interC_loss = InterConsistencyLoss(loss_weight=inter_loss_weight)
    
    def forward_train(self, raw_feature, label_confidence, label_start,
                      label_end):
        """Define the computation performed at every call when training."""
        confidence_map, start, end = self._forward(raw_feature)
        loss_bmn = self.loss_cls(confidence_map, start, end, label_confidence,
                             label_start, label_end,
                             self.bm_mask.to(raw_feature.device))
        loss_intraC = self.intraC_loss((start, end),(label_start, label_end))
        loss_dict = dict(loss_bmn=loss_bmn[0],loss_intraC=loss_intraC)
        return loss_dict