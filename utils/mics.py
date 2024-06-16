import cv2
import torch
import numpy as np
import math
from matplotlib import pyplot as plt
import os


def save_output(sample, output, aop, save_dir, input_type, sparse_type, with_norm):
    if input_type == 'polar':
        aop = aop
        if with_norm:
            pred_norm = output['norm'].detach().cpu().float().numpy()
            pred_norm = (np.transpose(pred_norm, (0, 2, 3, 1)) + 1.) * 255./2.

            gt_norm = sample['norm'].detach().cpu().float().numpy()
            gt_norm = (np.transpose(gt_norm, (0, 2, 3, 1)) + 1.) * 255./2.
    base_name = sample['base_name']
    net_mask = torch.logical_and(
                sample['mask'], sample['gt'] > 0)
    
    net_in = sample['input']
    pred_depth_wo_mask = output['pred']
    pred_depth = output['pred'] * net_mask
    gt_depth = sample['gt']
    sparse = sample['dep']

    for i in range(net_in.shape[0]):
        

        test_name = base_name[i]
        
        save_path = os.path.join(save_dir, test_name)
        vis_dir = save_path.split('/set')[0]
        os.makedirs(vis_dir, exist_ok=True)
        
        test_depth = pred_depth[i].permute(1,2,0).detach().cpu().numpy()
        test_depth = np.ascontiguousarray(test_depth, dtype=np.float64)
        save_path_depth = save_path+f'pred_depth_{input_type}.png'
        save_path_depth_16 = save_path+f'pred_depth_{input_type}_16.png'

        # print(test_depth.max())
        # exit()

        cv2.imwrite(save_path_depth_16, (np.clip(test_depth, 0, test_depth.max())*1000).astype("uint16"), [cv2.IMWRITE_PNG_COMPRESSION, 0])

        test_depth_wo_mask = pred_depth_wo_mask[i].permute(1,2,0).detach().cpu().numpy()
        test_depth_wo_mask = np.ascontiguousarray(test_depth_wo_mask, dtype=np.float64)

        save_path_depth_wo_mask_16 = save_path+f'pred_depth_wo_mask_{input_type}_16.png'
        cv2.imwrite(save_path_depth_wo_mask_16, (np.clip(test_depth_wo_mask, 0, test_depth.max())*1000).astype("uint16"), [cv2.IMWRITE_PNG_COMPRESSION, 0])

        
        save_path_depth_wo_mask = save_path+f'pred_depth_wo_mask_{input_type}.png'
        
        test_mask = net_mask[i].permute(1,2,0).detach().cpu().numpy()

        if input_type == 'gray':
            test_gray_scale = net_in[i][0:3].permute(1,2,0).detach().cpu().numpy()
        else:
            test_gray_scale = net_in[i][0:1].permute(1,2,0).detach().cpu().numpy()
            
        test_gray_scale = np.ascontiguousarray(test_gray_scale, dtype=np.float64)
        save_path_gray_scale = save_path+f'input_{input_type}.png'

        gt = gt_depth[i].permute(1,2,0).detach().cpu().numpy()
        gt = np.ascontiguousarray(gt, dtype=np.float64)
        save_path_gt = save_path+'gt.png'
        
        if input_type == 'polar':
            test_dop = net_in[i][1:2].permute(1,2,0).detach().cpu().numpy()
            test_aop = aop[i].permute(1,2,0).detach().cpu().numpy()
            save_path_dop = save_path+'dop.png'
            save_path_aop = save_path+'aop.png'
            if with_norm:
                test_norm_pred = pred_norm[i].astype(np.uint8)
                save_path_norm_pred = save_path+'pred_norm.png'
                test_norm_gt = gt_norm[i].astype(np.uint8)
                save_path_norm_gt = save_path+'gt_norm.png'

        
        test_sparse = sparse[i].permute(1,2,0).detach().cpu().numpy()
        test_sparse = np.ascontiguousarray(test_sparse, dtype=np.float64)
        save_path_sparse = save_path+f'sparse_depth_{sparse_type}.png'
        
        # gt_min = gt.min()
        # gt_max = gt.max()
        
        # test_depth = np.clip(test_depth, gt_min, gt_max)
        # test_depth_wo_mask = np.clip(test_depth_wo_mask, gt_min, gt_max)
        
        vis_min = 0
        vis_max = max(gt.max(), test_depth.max())

        # print('test',test_depth.min(), test_depth.max())
        # print('test_wo',test_depth_wo_mask.min(), test_depth_wo_mask.max())
        # print('gt', gt.min(), gt.max())
        

        diff = np.abs(gt - test_depth)
        save_path_diff = save_path+f'diff_gt_{sparse_type}.png'        

        # print(test_depth.max(), test_sparse.max(), test_depth_wo_mask.max(), test_gray_scale.max(), test_dop.max(), test_aop.max())


        fig1 = plt.figure()
        plt.imshow(test_gray_scale, cmap=plt.cm.gray)
        plt.axis('off')
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        fig1.savefig(save_path_gray_scale, bbox_inches='tight',pad_inches = 0)

        fig2 = plt.figure()
        plt.imshow(diff, cmap=plt.cm.jet, vmin=vis_min, vmax=vis_max/2)   
        plt.axis('off')
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        fig2.savefig(save_path_diff, bbox_inches='tight', pad_inches = 0)

        fig3 = plt.figure()
        plt.imshow(test_depth, cmap=plt.cm.jet, vmin=vis_min, vmax=vis_max)
        plt.axis('off') 
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        fig3.savefig(save_path_depth, bbox_inches='tight', pad_inches = 0)

        fig4 = plt.figure()
        plt.imshow(test_depth_wo_mask, cmap=plt.cm.jet, vmin=vis_min, vmax=vis_max)
        plt.axis('off')
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        fig4.savefig(save_path_depth_wo_mask, bbox_inches='tight', pad_inches = 0)

        fig5 = plt.figure()
        plt.imshow(test_sparse, cmap=plt.cm.jet, vmin=vis_min, vmax=vis_max)
        plt.axis('off')
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        fig5.savefig(save_path_sparse, bbox_inches='tight', pad_inches = 0)

        fig6 = plt.figure()
        plt.imshow(gt, cmap=plt.cm.jet, vmin=vis_min, vmax=vis_max)
        plt.axis('off')
        plt.margins(0,0)
        plt.gca().xaxis.set_major_locator(plt.NullLocator())
        plt.gca().yaxis.set_major_locator(plt.NullLocator())
        fig6.savefig(save_path_gt, bbox_inches='tight', pad_inches = 0)

        if input_type == 'polar':
            fig7 = plt.figure()
            plt.imshow(test_dop, cmap=plt.cm.gray)
            plt.axis('off')
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            fig7.savefig(save_path_dop, bbox_inches='tight', pad_inches = 0)

            fig8 = plt.figure()
            plt.imshow(test_aop, cmap=plt.cm.jet)
            plt.axis('off')
            plt.margins(0,0)
            plt.gca().xaxis.set_major_locator(plt.NullLocator())
            plt.gca().yaxis.set_major_locator(plt.NullLocator())
            fig8.savefig(save_path_aop, bbox_inches='tight', pad_inches = 0)

            if with_norm:
                fig9 = plt.figure()
                plt.imshow(test_norm_pred)
                plt.axis('off')
                plt.margins(0,0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                fig9.savefig(save_path_norm_pred, bbox_inches='tight', pad_inches = 0)

                fig10 = plt.figure()
                plt.imshow(test_norm_gt)
                plt.axis('off')
                plt.margins(0,0)
                plt.gca().xaxis.set_major_locator(plt.NullLocator())
                plt.gca().yaxis.set_major_locator(plt.NullLocator())
                fig10.savefig(save_path_norm_gt, bbox_inches='tight', pad_inches = 0)


        plt.cla()
        plt.close("all")