import torch
import torch.nn as nn
import torch.nn.functional as F

from mmaction.models.builder import LOSSES  

def intra_consistency_loss(heat, gt):
    """Calculate intra-consistency loss.
    Args:
        heat (torch.Tensor): Predicted heatmap of size [BsxT]
        gt (torch.Tensor): Binary Groundtruth of size [BsxT]
    Returns:
        torch.Tensor: Intra-consistency loss.
    """
    distance  = (heat.unsqueeze(-1)-heat.unsqueeze(1)).abs() # [BS,T,T]
    # gt -> [N,T,1]
    gt_1 = gt.unsqueeze(-1)  # [Bs,T] -> [Bs,T,1]
    gt_2 = gt.unsqueeze(1)   # [Bs,T] -> [Bs,1,T]
    eye_mat = torch.diag_embed(torch.ones_like(gt)) # [Bs,T,T]
    M_gt_1 = F.relu(torch.matmul(gt_1, gt_2) - eye_mat) # [Bs,T,T]
    M_gt_2 = (gt_1 - gt_2) * (gt_1 - gt_2) # [Bs,T,T]
    M_gt_3 = torch.ones_like(M_gt_1) - eye_mat - M_gt_1 - M_gt_2 # [Bs,T,T]
    pairs_1 = M_gt_1.sum(dim=(1,2)) + 1 # [Bs]
    pairs_2 = M_gt_2.sum(dim=(1,2)) + 1 # [Bs]
    pairs_3 = M_gt_3.sum(dim=(1,2)) + 1 # [Bs]

    consistency_1 = (distance * M_gt_1).sum(dim=(1,2)) / pairs_1 # [Bs]
    consistency_2 = - (distance * M_gt_2).sum(dim=(1,2)) / pairs_2 # [Bs]
    consistency_3 = (distance * M_gt_3).sum(dim=(1,2)) / pairs_3 # [Bs]
    consistency_loss = consistency_1 + consistency_2 + consistency_3
    return consistency_loss

def inter_consistency_loss(action_heat, start_heat, end_heat):
    """ Calculate inter-consistency loss.
    Args:
        action_heat (torch.Tensor): action heatmap of size [Bs,T]
        start_heat (torch.Tensor): start heatmap of size [Bs,T]
        end_heat (torch.Tensor): end heatmap of size [Bs,T]
    Returns:
        torch.Tensor: Inter-consistency loss.
    """

    diff = torch.cat([action_heat[:,1:,]-action_heat[:,:-1,], 
                      action_heat[:,-1:,]-action_heat[:,-2:-1,]], 1) # [N,T]
    zeros_tmp = torch.zeros_like(diff)
    diff_1 =   torch.where(diff>=0, diff,zeros_tmp) 
    diff_0 = - torch.where(diff<=0, diff,zeros_tmp) 

    start_diff_consistency = (diff_1 - start_heat).abs()
    end_diff_consistency = (diff_0 - end_heat).abs()       
    consistency_loss = end_diff_consistency + start_diff_consistency
    return consistency_loss

@LOSSES.register_module()
class IntraConsistencyLoss(nn.Module):
    def __init__(self, loss_weight=1):
        super().__init__()
        self.loss_weight = loss_weight

    def forward(self, pred, labels):
        loss = 0
        # the pred and labels can be (action, start, end) or just (start,end) probabilites
        for (heat,gt) in zip(pred, labels):
            loss = loss + intra_consistency_loss(heat, gt)

        loss = self.loss_weight * loss.mean()/len(pred)
        return loss

@LOSSES.register_module()
class InterConsistencyLoss(nn.Module):

    def __init__(self, loss_weight=1):
        super(InterConsistencyLoss, self).__init__()
        self.loss_weight = loss_weight

    def forward(self, y_pred):
        a_heat, s_heat, e_heat = y_pred
        inter_loss = self.loss_weight * inter_consistency_loss(a_heat, s_heat, e_heat)

        return inter_loss



      








