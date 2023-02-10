import math
import warnings

import mmcv
import torch
import torch.nn as nn

from mmdet.core import bbox_overlaps
from ..builder import LOSSES
from .utils import weighted_loss

def fp16_clamp(x, min=None, max=None):
    if not x.is_cuda and x.dtype == torch.float16:
        # clamp for cpu float16, tensor fp16 has no clamp implementation
        return x.float().clamp(min, max).half()

    return x.clamp(min, max)

@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def iou_loss(pred, target, linear=False, mode='log', eps=1e-6):
    """IoU loss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.
    The loss is calculated as negative log of IoU.

    Args:
        pred (torch.Tensor): Predicted bboxes of format (x1, y1, x2, y2),
            shape (n, 4).
        target (torch.Tensor): Corresponding gt bboxes, shape (n, 4).
        linear (bool, optional): If True, use linear scale of loss instead of
            log scale. Default: False.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
        eps (float): Eps to avoid log(0).

    Return:
        torch.Tensor: Loss tensor.
    """
    assert mode in ['linear', 'square', 'log']
    if linear:
        mode = 'linear'
        warnings.warn('DeprecationWarning: Setting "linear=True" in '
                      'iou_loss is deprecated, please use "mode=`linear`" '
                      'instead.')
    ious = bbox_overlaps(pred, target, is_aligned=True).clamp(min=eps)
   
    if mode == 'linear':
        loss = 1 - ious
    elif mode == 'square':
        loss = 1 - ious**2
    elif mode == 'log':
        loss = -ious.log()
    else:
        raise NotImplementedError
    return loss

def comgt(pred, pos_assigned_gt_inds, target):
    comgt = torch.tensor(0,dtype=torch.float,device=pred[0].device)
    comgt_smoothl1 = torch.tensor(0,dtype=torch.float,device=pred[0].device)
    nums_com_gt = 0

    for i in range(0, len(pos_assigned_gt_inds)):
        if pos_assigned_gt_inds[i] is None:
            continue
        dic = {}
        for j in range(0, len(pos_assigned_gt_inds[i])):
            if pos_assigned_gt_inds[i][j].item() in dic:
                dic[pos_assigned_gt_inds[i][j].item()].append(j)
            else:
                dic[pos_assigned_gt_inds[i][j].item()] = [j]
        nums_gt = 0
        for gt_ind in dic:
            if len(dic[gt_ind]) == 1:
                continue
            nums_gt += 1
            temp = []
            for pred_ind in dic[gt_ind]:
                temp.append(pred[i][pred_ind])
            temp_ts = torch.stack(temp, 0)
            mean = temp_ts.mean(0)
            mean = mean.detach()
            sml = nn.SmoothL1Loss(reduction='mean')
            comgt_smoothl1 += sml(target[i][gt_ind], mean)
        nums_com_gt += nums_gt
    if nums_com_gt > 0:
        comgt = 10.0 * comgt_smoothl1 / nums_com_gt
    
    return comgt

# @mmcv.jit(derivate=True, coderize=True)
def repgt(pred, pos_assigned_gt_inds, target):
    nums_repgt = 0
    sigma_repgt = 0.9
    repgt_smoothln = torch.tensor(0,dtype=torch.float,device=pred[0].device)
    repgt = torch.tensor(0,dtype=torch.float,device=pred[0].device)
    for i in range(0, len(pos_assigned_gt_inds)):
        if pos_assigned_gt_inds[i] is None:
            continue
        
        overlaps = []
        argmax = []
        for j in range(0, pred[i].shape[0]):
            mmax = 0
            arg = -1
            for k in range(0, target[i].shape[0]):
                if k == pos_assigned_gt_inds[i][j]:
                    continue
                ovlp = aaoverlap(pred[i][j], target[i][k])
                if ovlp > mmax:
                    mmax = ovlp
                    arg = k
            overlaps.append(mmax)
            argmax.append(arg)

        for k in range(0, len(overlaps)):
            if(overlaps[k] == 0):
                continue
            nums_repgt += 1
            iog = IoG(pred[i][k], target[i][argmax[k]])
            if iog>sigma_repgt:
                repgt_smoothln+=((iog-sigma_repgt)/(1-sigma_repgt)-math.log(1-sigma_repgt))
            elif iog<=sigma_repgt:
                repgt_smoothln+=-math.log(max(1-iog, 1e-6))

    if nums_repgt>0:
        repgt = 10.0 * repgt_smoothln/nums_repgt 

    return repgt 

def aaoverlap(boxa, boxb):
        lt = torch.max(boxa[:2],
                    boxb[:2])
        rb = torch.min(boxa[2:],
                    boxb[2:]) 

        wh = fp16_clamp(rb - lt, min=0)
        overlaps = wh[0] * wh[1]
        return overlaps
    
def IoG(box_a, box_b):                                                                                             
    inter_xmin = torch.max(box_a[0], box_b[0])                                                                     
    inter_ymin = torch.max(box_a[1], box_b[1])                                                                     
    inter_xmax = torch.min(box_a[2], box_b[2])                                                                     
    inter_ymax = torch.min(box_a[3], box_b[3])                                                                     
    Iw = torch.clamp(inter_xmax - inter_xmin, min=0)                                                               
    Ih = torch.clamp(inter_ymax - inter_ymin, min=0)                                                               
    I = Iw * Ih                                                                                                    
    G = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])                                                              
    return I / G

@LOSSES.register_module()
class RepLoss(nn.Module):
    """IoULoss.

    Computing the IoU loss between a set of predicted bboxes and target bboxes.

    Args:
        linear (bool): If True, use linear scale of loss else determined
            by mode. Default: False.
        eps (float): Eps to avoid log(0).
        reduction (str): Options are "none", "mean" and "sum".
        loss_weight (float): Weight of loss.
        mode (str): Loss scaling mode, including "linear", "square", and "log".
            Default: 'log'
    """

    def __init__(self,
                 linear=False,
                 eps=1e-6,
                 reduction='mean',
                 loss_weight=1.0,
                 mode='log'):
        super(RepLoss, self).__init__()
        assert mode in ['linear', 'square', 'log']
        if linear:
            mode = 'linear'
            warnings.warn('DeprecationWarning: Setting "linear=True" in '
                          'IOULoss is deprecated, please use "mode=`linear`" '
                          'instead.')
        self.mode = mode
        self.linear = linear
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                pos_assigned_gt_inds,
                target,
                pred2,
                target2,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function.

        Args:
            pred (torch.Tensor): The prediction.
            target (torch.Tensor): The learning target of the prediction.
            weight (torch.Tensor, optional): The weight of loss for each
                prediction. Defaults to None.
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Defaults to None. Options are "none", "mean" and "sum".
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        if (weight is not None) and (not torch.any(weight > 0)) and (
                reduction != 'none'):
            if pred2.dim() == weight.dim() + 1:
                weight = weight.unsqueeze(1)
            return (pred2 * weight).sum()  # 0
        if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n, 4) to (n,) to match the
            # iou_loss of shape (n,)
            assert weight.shape == pred2.shape
            weight = weight.mean(-1)
        
        loss = self.loss_weight * iou_loss(
            pred2,
            target2,
            weight,
            mode=self.mode,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        
        rt = repgt(pred, pos_assigned_gt_inds, target)
        loss += rt

        ct = comgt(pred, pos_assigned_gt_inds, target)
        loss += ct

        return loss
    

        
