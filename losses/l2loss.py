"""
    CompletionFormer
    ======================================================================

    L2 loss implementation
"""


import torch
import torch.nn as nn


class L2Loss(nn.Module):
    def __init__(self, args, mode = 'depth'):
        super(L2Loss, self).__init__()

        self.args = args
        self.t_valid = 0.0001
        self.mode = mode

        print("Loss mode:", mode)

    def forward(self, pred, gt):
            
        if self.mode == 'depth':
            gt = torch.clamp(gt, min=0, max=self.args.max_depth)
            pred = torch.clamp(pred, min=0, max=self.args.max_depth)

            mask = (gt > self.t_valid).type_as(pred).detach()
        else:
            mask = torch.tile(torch.norm(gt, dim=1, keepdim=True), (1,3,1,1)) > 0.3


        d = (pred - gt)**2 * mask

        d = torch.sum(d, dim=[1, 2, 3])
        num_valid = torch.sum(mask, dim=[1, 2, 3])

        loss = d / (num_valid + 1e-8)

        loss = loss.sum()

        return loss
