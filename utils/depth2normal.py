import torch
import numpy as np
import math

def depth2norm(depth, camera_matrix, neighborhood_size=3, computation_mode="row", step_size=50):
    '''
    depth: a torch tensor of depth map, has shape (B, H, W), values should be in the unit of mm
    camera_matrix: a numpy array of the camera intrinsic parameters, has shape (3, 3), note that the units shall be in terms of mm
    neighborhood_size: defines the local neighborhood size used during depth-to-normal conversion, recommended value is 3 or 5, make sure it is odd
    computation_mode: this module contains batch matrix multiplication, if data size is large, we recommend a slower but memory-sufficient computation, modes are:
        1. vectorized: fully vectorized, require the most GPU memory space
        2. row: semi-vectorized, require less GPU space but is slower
        3. row-col: least vectorized, require the least GPU space but is the slowest
    step_size: used only in mode row and row-col
        1. in row mode, step_size shall be scalar and is the number of rows to compute in each loop
        2. in row-col mode, step_size shall be a scalar which is the step size for number of cols to compute in the inner-most loop
    
    return: a torch tensor of shape (B, H, W), representing a rough normal map converted directly from the depth input, via a least-square module, normal vectors are normalized to 1
    '''
    B, H_orig, W_orig = depth.size()
    
    if neighborhood_size > H_orig or neighborhood_size > W_orig or neighborhood_size <= 0:
        raise Exception("invalid neighborhood size, received {}, which should have been smaller than the width and height of the input and been bigger than 0".format(neighborhood_size))
    
    neighborhood_size = 2 * int(neighborhood_size/2) + 1
    patch_size = neighborhood_size
    
    # why this padding? shouldn't padding sum up to patch_size?
    pad_value = math.floor(patch_size/2)
    total_padding = (pad_value, pad_value, pad_value, pad_value)
    depth = torch.nn.functional.pad(depth, pad=total_padding)
    
    depth_patches = depth.unfold(1, patch_size, 1).unfold(2, patch_size, 1)
    B, H, W, P, P = depth_patches.size()
    depth_patches = torch.unsqueeze(depth_patches.reshape((B, H, W, P*P)), 4)
    point_patches = depth_patches.repeat(1, 1, 1, 1, 3)
    
    xx_over_z, yy_over_z = np.meshgrid(np.arange(0, W_orig, 1).astype(np.float32), np.arange(0, H_orig, 1).astype(np.float32)) # (H, W)
    
    xx_over_z = (xx_over_z - camera_matrix[0, 2]) / camera_matrix[0, 0]
    yy_over_z = (yy_over_z - camera_matrix[1, 2]) / camera_matrix[1, 1]
    
    xx_over_z = torch.from_numpy(xx_over_z).to(depth.get_device())
    yy_over_z = torch.from_numpy(yy_over_z).to(depth.get_device())
    
    xx_over_z = torch.nn.functional.pad(xx_over_z, pad=total_padding)
    yy_over_z = torch.nn.functional.pad(yy_over_z, pad=total_padding)
    
    xx_over_z_patches = xx_over_z.unfold(0, patch_size, 1).unfold(1, patch_size, 1) # (H, W, PS, PS)
    yy_over_z_patches = yy_over_z.unfold(0, patch_size, 1).unfold(1, patch_size, 1) # (H, W, PS, PS)
    
    xx_over_z_patches = xx_over_z_patches.reshape((H, W, P*P)) # (H, W, PS*PS)
    yy_over_z_patches = yy_over_z_patches.reshape((H, W, P*P)) # (H, W, PS*PS)
    
    xx_over_z_patches = torch.unsqueeze(xx_over_z_patches, 0).repeat(B, 1, 1, 1) # (B, H, W, PS*PS)
    yy_over_z_patches = torch.unsqueeze(yy_over_z_patches, 0).repeat(B, 1, 1, 1) # (B, H, W, PS*PS)
    z = torch.ones(xx_over_z_patches.size()).to(xx_over_z_patches.get_device())
    
    xyz_preproc = torch.stack([xx_over_z_patches, yy_over_z_patches, z], dim=4)    
    A = point_patches * xyz_preproc # x, y, z points in the camera 3D coordinates, has shape (B, H, W, PS*PS, 3)
    
    if computation_mode == "vectorized":
    ##########################################################################################################################
    # fully vectorized version, should encounter memory explosion for large images, if so use the alternative versions below #
    ##########################################################################################################################
        A_T = torch.transpose(A, dim0=3, dim1=4)
        tmp = torch.matmul(A_T, A)
        # print(tmp.get_device(), tmp.size())
        tmp_det = torch.det(tmp.cpu()) # (B, H, W), compute in cpu
        tmp_det = tmp_det.to(A_T.get_device())
        # print(tmp_det.size())
        tmp_det = torch.unsqueeze(tmp_det, 3).repeat(1, 1, 1, 3) # (B, H, W, 3)
        tmp_det = torch.unsqueeze(tmp_det, 4).repeat(1, 1, 1, 1, 3) # (B, H, W, 3, 3)
        tmp_inversible = torch.where(tmp_det != 0, tmp, torch.eye(3).to(tmp.get_device()))
        norm_unnormalized = torch.matmul(torch.matmul(torch.inverse(tmp_inversible), A_T), torch.ones(patch_size**2).to(A_T.get_device()))
    elif computation_mode == "row-col":
    #############################
    # row-col iteration version #
    #############################
        if step_size <= 0 or step_size > W:
            raise Exception("invalid step_size value in row-col computation mode, received {}, it shall be bigger than 0 and less than or equal to the width of the input".format(step_size))
        
        norm_unnormalized = None
        for i in range(H):
            norm_unnormalized_row = None
            # print(i)
            for j in range(int(W / step_size)):
                if j == int(W / step_size) - 1 and int(W / step_size) < (W / step_size):
                    remained_num_cols = W - int(W / step_size)
                    mat1 = torch.unsqueeze(A[:, i, j*step_size:j*step_size+step_size+remained_num_cols, :, :], 1)
                else:
                    mat1 = torch.unsqueeze(A[:, i, j*step_size:j*step_size+step_size, :, :], 1)
                mat2 = torch.transpose(mat1, dim0=3, dim1=4)
                tmp = torch.matmul(mat2, mat1)    
                
                mat_det = torch.unsqueeze(torch.det(tmp), 3).repeat(1, 1, 1, 3)
                mat_det = torch.unsqueeze(mat_det, 4).repeat(1, 1, 1, 1, 3)
                identity_mat = torch.eye(3).to(tmp.get_device())
                tmp = torch.where(mat_det > 1e-15, tmp, identity_mat) # filter out un-inversible matrices
                result = torch.matmul(torch.matmul(torch.inverse(tmp), mat2), torch.ones(patch_size**2).to(tmp.get_device()))

                if norm_unnormalized_row is None:
                    norm_unnormalized_row = result
                else:
                    norm_unnormalized_row = torch.cat([norm_unnormalized_row, result], 2)
            if norm_unnormalized is None:
                norm_unnormalized = norm_unnormalized_row
            else:
                norm_unnormalized = torch.cat([norm_unnormalized, norm_unnormalized_row], 1)
    elif computation_mode == "row":
    #########################
    # row iteration version #
    #########################
        if step_size <= 0 or step_size > H:
            raise Exception("invalid step_size value in row computation mode, received {}, which shall be bigger than 0 and smaller than the height of the input".format(step_size))
        
        norm_unnormalized = None
        for r in range(int(H/step_size)):
            if r == int(H/step_size) - 1 and int(H/step_size) < H/step_size: # last iteration & there are left over rows
                matA = A[:, r*step_size:H, ...]
            else:
                matA = A[:, r*step_size:(r+1)*step_size, ...] # (B, step_size, W, PS*PS, 3)
            matA_T = torch.transpose(matA, dim0=3, dim1=4) # (B, step_size, W, 3, 3)
            
            # filter non-inversible matrices
            tmp = torch.matmul(matA_T, matA)
            # print("tmp", tmp)
            tmp_det = torch.det(tmp) # (B, step_size, W)
            tmp_det = torch.unsqueeze(tmp_det, 3).repeat(1, 1, 1, 3) # (B, step_size, W, 3)
            tmp_det = torch.unsqueeze(tmp_det, 4).repeat(1, 1, 1, 1, 3) # (B, step_size, W, 3, 3)
            
            tmp_inversible = torch.where(torch.abs(tmp_det) > 1e-80, tmp, torch.eye(3).to(matA.get_device())) # (B, step_size, W, 3, 3)
            tmp_inv = torch.inverse(tmp_inversible)
            # print("tmp_inv", tmp_inv)
            norm_unnormalized_row = torch.matmul(torch.matmul(tmp_inv, matA_T), torch.ones(patch_size**2).to(matA.get_device())) # (B, step_size, W, 3)
            
            if norm_unnormalized is None:
                norm_unnormalized = norm_unnormalized_row
            else:
                norm_unnormalized = torch.cat([norm_unnormalized, norm_unnormalized_row], dim=1)
    else:
        raise Exception("computation mode shall be one of 'vectorized', 'row-col', and 'row' but received {}".format(computation_mode))
    
    # norm_unnormalized has shape (B, H, W, 3)
    norm_normalized = torch.nn.functional.normalize(norm_unnormalized, dim=3)
    
    # # from left-hand coordinate to right-hand one (negate x-axis values)
    norm_normalized[:, :, :, 0] = norm_normalized[:, :, :, 0] * -1
    
    # reshape to conventional (B, 3, H, W)
    norm_normalized = norm_normalized.permute((0, 3, 1, 2))
    
    return norm_normalized
    
    
    
    