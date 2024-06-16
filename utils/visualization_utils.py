import cv2
import numpy as np
import torch
import os

def depth_to_colormap(depth, max_depth):
    """
    depth: (torch.Tensor) of shape (1, H, W)
    max_depth: the maximum depth used to normalize depth values into 0-255 for visualization
    """
    npy_depth = depth.detach().cpu().numpy()[0]
    vis = ((npy_depth / max_depth) * 255).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_JET)
    return vis

def norm_to_colormap(norm):
    """
    norm: (torch.Tensor) of shape (3, H, W)
    """
    norm = norm.permute(1,2,0).detach().cpu().numpy()
    vis = ((norm+1) * 255/2).astype(np.uint8)
    vis = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)
    return vis

def save_visualization(pred, gt, dep, folder, with_error_map=False):
    out = depth_to_colormap(pred, 2.6)
    gt_ = depth_to_colormap(gt, 2.6)
    sparse = depth_to_colormap(dep, 2.6)

    cv2.imwrite(os.path.join(folder, "out.png"), out)
    cv2.imwrite(os.path.join(folder, "sparse.png"), sparse)
    cv2.imwrite(os.path.join(folder, "gt.png"), gt_)
    
    if with_error_map:
        gt_mask = gt.detach().cpu().numpy()[0].transpose(1,2,0)
        gt_mask[gt_mask <= 0.001] = 0
        err = torch.abs(pred-gt)

        error_map_vis = depth_to_colormap(err, 0.55)
        error_map_vis[np.tile(gt_mask, (1,1,3))==0] = 0
        
        cv2.imwrite(os.path.join(folder, 'err.png'), error_map_vis)
        # -- we implicitly assume when error map is specified, we also wish to store the raw prediction for later analysis --
        cv2.imwrite(os.path.join(folder, 'out_raw.png'), (pred.detach().cpu().numpy()[0]*1000).astype(np.uint16)) # in mm
        
def save_raw_data(pred, gt, dep, folder, pol=None, rgb=None):
    cv2.imwrite(os.path.join(folder, "out_raw.png"), (pred.detach().cpu().numpy()[0]*1000).astype(np.uint16))
    cv2.imwrite(os.path.join(folder, "sparse_raw.png"), (dep.detach().cpu().numpy()[0]*1000).astype(np.uint16))
    cv2.imwrite(os.path.join(folder, "gt_raw.png"), (gt.detach().cpu().numpy()[0]*1000).astype(np.uint16))
    
    if rgb is not None:
        rgb = rgb.detach().cpu().numpy()[0].transpose((1, 2, 0)) # (H, W, 3)
        H, W, _ = rgb.shape
        std = np.tile(np.array([0.229, 0.224, 0.225])[None,None,:], (H, W, 1))
        mu = np.tile(np.array([0.485, 0.456, 0.406])[None,None,:], (H, W, 1))
        rgb = rgb * std
        rgb = rgb + mu
        rgb = (rgb * 255).astype(np.uint8)
        cv2.imwrite(os.path.join(folder, "rgb.png"), rgb)
        
    if pol is not None:
        pol = pol.detach().cpu().numpy()[0].transpose((1, 2, 0))
        np.save(os.path.join(folder, "pol.npy"), pol)