from mmaction.models.localizers import TEM
from mmaction.models.builder import LOCALIZERS
from ccaction.models.losses import InterConsistencyLoss, IntraConsistencyLoss

@LOCALIZERS.register_module()
class BUMR(TEM):
    """
    Bottom-Up Temporal Action Localization with Mutual Regularization
    From paper https://arxiv.org/abs/2002.07358
    """
    def __init__(self,inter_loss_weight=1, intra_loss_weight=1, 
                    *args,**kwargs):
        super().__init__(*args,**kwargs)
        self.interC_loss = InterConsistencyLoss(loss_weight=inter_loss_weight)
        self.intraC_loss = IntraConsistencyLoss(loss_weight=intra_loss_weight)
    
    def forward_train(self, raw_feature, label_action, label_start, label_end):
        """Define the computation performed at every call when training."""
        tem_output = self._forward(raw_feature)
        score_action = tem_output[:, 0, :]
        score_start = tem_output[:, 1, :]
        score_end = tem_output[:, 2, :]

        loss_action = self.loss_cls(score_action, label_action,
                                    self.match_threshold)
        loss_start_small = self.loss_cls(score_start, label_start,
                                         self.match_threshold)
        loss_end_small = self.loss_cls(score_end, label_end,
                                       self.match_threshold)

        loss_interC = self.interC_loss(tem_output, (label_action, label_start, label_end))
        loss_intraC = self.intraC_loss(tem_output)
        loss_dict = {
            'loss_action': loss_action * self.loss_weight,
            'loss_start': loss_start_small,
            'loss_end': loss_end_small,
            'loss_interC': loss_interC,
            'loss_intraC': loss_intraC,
        }

        return loss_dict