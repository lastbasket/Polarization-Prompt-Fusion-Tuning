import torch
import torch.nn as nn
import numpy as np
from losses.l1loss import L1Loss
from losses.l2loss import L2Loss


def get_mae(pred_camera_normal, net_gt, net_mask):
 
    h, w = pred_camera_normal.shape[-2:]

    net_gt = net_gt[:, :, :h,:w]
    pred_camera_normal = pred_camera_normal[:, :, :h,:w]

    net_mask = net_mask[:, :, :h,:w]

    valid_mask = net_mask *  torch.where(torch.mean(net_gt, dim=1, keepdim=True) != 0 , torch.ones(torch.mean(net_gt, dim=1, keepdim=True).size()).cuda(), torch.zeros(torch.mean(net_gt, dim=1, keepdim=True).size()).cuda())
    mae_map = torch.sum(net_gt * pred_camera_normal, dim=1, keepdim=True)#.clip(-1,1)
    mae_map = torch.acos(mae_map) * 180. / np.pi

    b = net_gt.size(0)
    maes = []
    for i in range(b):
        mae_map_i = mae_map[i:i+1]
        valid_mask_i = valid_mask[i:i+1]
        mae = torch.mean(mae_map_i[(valid_mask_i * mae_map_i)>0])
        maes.append(mae.item())
    
    return np.mean(maes)


class Loss:
    def __init__(self, args) -> None:
        self.l1loss = L1Loss(args)
        self.l2loss = L2Loss(args)
        self.args = args

    def forward(self, sample, output):
        dep_mask = sample['mask']
        norm_mask = sample['norm_mask']
        pred_dep = output['depth']
        pred_norm = output['norm']

        gt_dep = sample['depth']
        gt_norm = sample['norm']

        depth_loss = self.l1loss(pred_dep, gt_dep)*self.args.l1_weight + self.l2loss(pred_norm, gt_norm)*self.args.l2_weight
        depth_loss = depth_loss * dep_mask

        norm_loss = get_mae(pred_norm, gt_norm, norm_mask)
                    
        return depth_loss, norm_loss
        
