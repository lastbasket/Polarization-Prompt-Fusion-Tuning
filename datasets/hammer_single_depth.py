import os
import warnings

import numpy as np
import cv2
import json
import h5py
from . import BaseDataset

import random

from PIL import Image
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF

warnings.filterwarnings("ignore", category=UserWarning)

class HammerSingleDepthDataset(BaseDataset):
    def __init__(self, args, mode):
        """object initializer

        Args:
            args (obj): execution arguments, the used arguments are:
                        * dir_data: the path to the root of the dataset
                        * data_txt: the text file containing a list of paths to each data, there should be a "DATA_ROOT" placeholder to substitute the data root path into
                        * path_to_vd: the path to the npy file containing the viewing direction data
                        * use_norm: boolean indicating if the normals are used
                        * use_pol: boolean indicating if the polarization data are used


            mode (str): can be either 'train' or 'val' or 'test', which alters the behavior during data loading
        """
        super(HammerSingleDepthDataset, self).__init__(args, mode)

        self.args = args
        self.mode = mode

        # -- the camera intrinsic parameters --
        self.K = torch.Tensor([7.067553100585937500e+02, 7.075133056640625000e+02, 5.456326819328060083e+02, 3.899299663507044897e+02])

        # -- the data files --
        with open(args.data_txt.replace("MODE", mode), "r") as file:
            files_names = file.read().split("\n")[:-1]
        # files_names = files_names[:10]

        
        PERCENTAGE = args.data_percentage
        if PERCENTAGE < (1-1e-6):
            random.shuffle(files_names)
            files_names = files_names[:int(len(files_names) * PERCENTAGE)]

        self.rgb_files = [s.replace("DATA_ROOT", args.dir_data) for s in files_names] # note that the original paths in the path list is pointing to the rgb images
        self.sparse_depth_d435_files = [s.replace("DATA_ROOT", args.dir_data).replace("rgb", "depth_d435") for s in files_names]
        self.sparse_depth_l515_files = [s.replace("DATA_ROOT", args.dir_data).replace("rgb", "depth_l515") for s in files_names]
        self.sparse_depth_itof_files = [s.replace("DATA_ROOT", args.dir_data).replace("rgb", "depth_tof") for s in files_names]
        self.gt_files = [s.replace("DATA_ROOT", args.dir_data).replace("rgb", "_gt") for s in files_names]
        if self.args.use_pol:    
            pol_folder = ''
            if self.args.pol_rep == 'grayscale-4':
                pol_folder = 'pol_grayscale'
            elif self.args.pol_rep == 'rgb-12':
                pass
            elif self.args.pol_rep == 'leichenyang-7':
                # pass
                pol_folder = 'pol_processed'
                self.vd = np.load('./utils/vd.npy')
            self.pol_files = [s.replace("DATA_ROOT", args.dir_data).replace("rgb", pol_folder) for s in files_names] # note that the polarizatins are stored as npy files
        if self.args.use_norm:
            self.norm_files = [s.replace("DATA_ROOT", args.dir_data).replace("rgb", "norm") for s in files_names] # note that the normals are stored as npy files

        if args.use_single:
            self.depth_type = args.depth_type
            print('Use', self.depth_type)

    def __len__(self):
        return len(self.rgb_files)
        # return 120

    def __getitem__(self, idx):
        """return data item

        Args:
            idx (int): the index of the date element within the mini-batch

        Return:
            output (dict): the output dictionary of four elements, which are
                        * 'rgb': the pytorch tensor of the RGB 3-channel image data
                        * 'dep': the pytorch tensor of the 1-channel sparse depth (i.e. to be completed)
                        * 'gt': the pytorch tensor of the 1-channel groundtruth depth
                        * 'K': the pytorch tensor of the camera intrinsic matrix, defined to be [fx, fy, cx, cy]
                        * 'mask': the mask of invalid data in the groundtruth
                        * 'pol': [IF use_pol IS TRUE] the pytorch tensor of the 7-channel polarization representation, otherwise all-zero
                        * 'norm': [IF use_norm IS TRUE] the pytorch tensor of the 3-channel normal map, otherwise all-zero
        """
        # idx = 6000
        orig_idx = idx
        idx = orig_idx

        def np2tensor(sample):
            # HWC -> CHW
            sample_tensor = torch.from_numpy(sample.copy().astype(np.float32)).permute(2,0,1)
            return sample_tensor
        
        # -- prepare sparse depth --
        sparse_depth_file = None

        if self.depth_type == 0:
            sparse_depth_file = self.sparse_depth_d435_files[idx]
        elif self.depth_type == 1:
            sparse_depth_file = self.sparse_depth_l515_files[idx]
        elif self.depth_type == 2:
            sparse_depth_file = self.sparse_depth_itof_files[idx]

        sparse_depth = cv2.imread(sparse_depth_file, -1)[:,:,None] # (H, W, 1)
        sparse_depth = sparse_depth[::4,::4,...] / 1000.0
        sparse_depth = np2tensor(sparse_depth) # (1, H, W)

        # -- prepare gt depth --
        gt = cv2.imread(self.gt_files[idx], -1)[:,:,None] # (H, W, 1)
        gt = gt[::4,::4,...] / 1000.0
        gt_clone = np.copy(gt)
        gt = np2tensor(gt) # (1, H, W)

        # -- prepare rgb --
        rgb = cv2.imread(self.rgb_files[idx]) # (H, W, 3)
        rgb = rgb[::4,::4,...]
        rgb = np2tensor(rgb) # (3, H, W)

        # -- prepare normals --
        if self.args.use_norm:
            norm = ((cv2.cvtColor(cv2.imread(self.norm_files[idx]), cv2.COLOR_BGR2RGB)/255*2)-1).astype(np.float32) # (H, W, 3)
            norm = norm[::4,::4,...]
            norm = np2tensor(norm) # (3, H, W)

        # -- prepare intrinsics --
        K = self.K.clone()

        # -- prepare mask, which masks out invalid pixels in the groundtruth --
        mask = np.ones_like(gt_clone) # (H, W, 1)
        mask[gt_clone<1e-3] = 0
        mask = np2tensor(mask) # (1, H, W)

        # -- prepare polarization representation -- 
        if self.args.use_pol:
            pol = None
            if self.args.pol_rep == 'grayscale-4':
                pol = np.load(self.pol_files[idx].replace('.png', '.npy')) # (H, W, 4), grayscale of 0, 45, 90, and 135
                pol = pol[::4,::4,...]
                pol = np2tensor(pol) # (4, H, W)
            elif self.args.pol_rep == 'rgb-12':
                pass
            elif self.args.pol_rep == 'leichenyang-7':
                pol = np.load(self.pol_files[idx].replace('.png', '.npy'))
                pol = pol[::4,::4,...]
                vd = self.vd[::4, ::4, ...]
                iun = pol[..., 0:1]
                rho = pol[..., 1:2]
                phi = pol[..., 2:3]
                phi_encode = np.concatenate([np.cos(2 * phi), np.sin(2 * phi)], axis=2)
                pol = np.concatenate([iun, rho, phi_encode, vd], axis=2)
                pol = np2tensor(pol) # (7, H, W)
            elif self.args.pol_rep == 'rgb':
                pol = rgb
    
        # -- apply data augmentation --
        rgb = rgb / 255.0
        t_rgb = T.Compose([
                T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            ])
        rgb = t_rgb(rgb)

        if self.args.use_pol:
            if self.args.pol_rep == 'grayscale-4':
                pol = pol / 255.0
            
        # -- return data --
        # print("--> RGB size {}".format(rgb.shape))
        # print("--> Sparse depth size {}".format(sparse_depth.shape))
        # print("--> Groundtruth size {}".format(gt.shape))
        # print("--> Mask size {}".format(mask.shape))
        output = {'rgb': rgb, \
                    'dep': sparse_depth, \
                    'gt': gt, \
                    'K': K, \
                    'net_mask': mask}
        
        if self.args.use_pol:
            output['pol'] = pol

        if self.args.use_norm:
            output['norm'] = norm

        return output