"""
    CompletionFormer
    ======================================================================

    L1 loss implementation
"""


import torch
import torch.nn as nn


class L1Loss(nn.Module):
    def __init__(self, args):
        super(L1Loss, self).__init__()

        self.args = args
        self.t_valid = 0.0001

    def forward(self, pred, gt):
        gt = torch.clamp(gt, min=0, max=self.args.max_depth)
        pred = torch.clamp(pred, min=0, max=self.args.max_depth)

        # print("--> Pred contains NaN ? {}".format(torch.any(torch.isnan(pred))))
        # print("--> Gt contains NaN ? {}".format(torch.any(torch.isnan(gt))))
        
        mask = (gt > self.t_valid).type_as(pred).detach()
        pred = torch.nan_to_num(pred)
        d = torch.abs(pred - gt) * mask

        d = torch.sum(d, dim=[1, 2, 3])
        num_valid = torch.sum(mask, dim=[1, 2, 3])
        # print("--> Has zero valid pixel? {}".format(torch.any(torch.where(num_valid==0, 1, 0).bool())))
        loss = d / (num_valid + 1e-8)

        loss = loss.sum()

        return loss
