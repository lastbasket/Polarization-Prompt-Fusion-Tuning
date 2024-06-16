"""
    CompletionFormer

    ======================================================================

    All of the parameters are defined here.
"""


import time
import argparse


parser = argparse.ArgumentParser(description='CompletionFormer')


# Dataset
parser.add_argument('--data_percentage', type=float, default=1.0, help='the percentage of the entire dataset used')

parser.add_argument('--dir_data',
                    type=str,
                    default='./dataset/NYUDepthV2_HDF5',
                    help='path to dataset')
parser.add_argument('--data_name',
                    type=str,
                    default='NYU',
                    # default='KITTIDC',
                    choices=('NYU', 'KITTIDC', 'HAMMER'),
                    help='dataset name')
parser.add_argument('--split_json',
                    type=str,
                    default='../data_json/nyu.json',
                    # default='../data_json/kitti_dc.json',
                    help='path to json file')
parser.add_argument('--patch_height',
                    type=int,
                    default=228,
                    # default=240,
                    help='height of a patch to crop')
parser.add_argument('--patch_width',
                    type=int,
                    default=304,
                    # default=1216,
                    help='width of a patch to crop')
parser.add_argument('--top_crop',
                    type=int,
                    default=0,
                    # default=100,
                    help='top crop size for KITTI dataset')
parser.add_argument('--use_norm',
                    action='store_true',
                    default=False,
                    help='whether or not to use normal maps, available for HAMMER')
parser.add_argument('--use_pol',
                    action='store_true',
                    default=False,
                    help='whether or not to use polarization representations, available for HAMMER')
parser.add_argument('--pol_rep', type=str, default="grayscale-4", choices=['grayscale-4', 'rgb-12', 'leichenyang-7', 'rgb'], help="the polarization representation to be used")
parser.add_argument('--data_txt',
                    type=str,
                    default='',
                    help='the path list file used for HAMMER, the path to the file (that is, this argument) must contain a placeholder string called "MODE", which is for substitute for either "train", "val", or "test" during program execution')


# Hardware
parser.add_argument('--seed',
                    type=int,
                    default=43,
                    help='random seed point')
parser.add_argument('--gpus',
                    type=str,
                    default="0,1,2,3",
                    help='visible GPUs')
parser.add_argument('--port',
                    type=str,
                    default='29500',
                    help='master port')
parser.add_argument('--address',
                    type=str,
                    default='localhost',
                    help='master address')
parser.add_argument('--num_threads',
                    type=int,
                    default=4,
                    help='number of threads')
parser.add_argument('--no_multiprocessing',
                    action='store_true',
                    default=False,
                    help='do not use multiprocessing')


# Network
parser.add_argument('--pretrain_list_file', type=str, default=None, help="the text file containing a list of ckeckpoints to be evaluated in batch")
parser.add_argument('--model',
                    type=str,
                    default='CompletionFormer',
                    help='main model name')
parser.add_argument('--from_scratch',
                    action='store_true',
                    default=False,
                    help='train from scratch')
parser.add_argument('--prop_time',
                    type=int,
                    default=6,
                    help='number of propagation')
parser.add_argument('--prop_kernel',
                    type=int,
                    default=3,
                    help='propagation kernel size')
parser.add_argument('--preserve_input',
                    action='store_true',
                    default=False,
                    help='preserve input points by replacement')
parser.add_argument('--affinity',
                    type=str,
                    default='TGASS',
                    choices=('AS', 'ASS', 'TC', 'TGASS'),
                    help='affinity type (dynamic pos-neg, dynamic pos, '
                         'static pos-neg, static pos, none')
parser.add_argument('--affinity_gamma',
                    type=float,
                    default=0.5,
                    help='affinity gamma initial multiplier '
                         '(gamma = affinity_gamma * number of neighbors')
parser.add_argument('--conf_prop',
                    action='store_true',
                    default=True,
                    help='confidence for propagation')
parser.add_argument('--no_conf',
                    action='store_false',
                    dest='conf_prop',
                    help='no confidence for propagation')
parser.add_argument('--legacy',
                    action='store_true',
                    default=False,
                    help='legacy code support for pre-trained models')
parser.add_argument('--completionformer_mode', type=str, default="rgbd", help='the mode used for completionformer')
parser.add_argument('--prior',
                    action='store_true',
                    default=False,
                    help='whether to use prior')
parser.add_argument('--pre_pvt',
                    action='store_true',
                    default=False,
                    help='whether to use pvt')
parser.add_argument('--pre_res',
                    action='store_true',
                    default=False,
                    help='whether to use a pretrained resnet in completionformer')
parser.add_argument('--direct_cat',
                    action='store_true',
                    default=False,
                    help='whether to use direct concatenation for completionformer')
parser.add_argument('--direct_align',
                    action='store_true',
                    default=False,
                    help='NO IDEA WHAT THIS IS')
parser.add_argument('--align',
                    action='store_true',
                    default=False,
                    help='NO IDEA WHAT THIS IS')
parser.add_argument('--layer0',
                    action='store_true',
                    default=False,
                    help='NO IDEA WHAT THIS IS')
parser.add_argument('--pretrained_completionformer', type=str, default='', \
                    help='path to the pretrained completionformer')

# Training
parser.add_argument('--loss',
                    type=str,
                    default='1.0*L1+1.0*L2',
                    help='loss function configuration')
parser.add_argument('--l1_weight',
                    type=float,
                    default=1.0)
parser.add_argument('--l2_weight',
                    type=float,
                    default=1.0)

parser.add_argument('--epochs',
                    type=int,
                    default=72,
                    # default=100,
                    help='number of epochs to train')
parser.add_argument('--milestones',
                    nargs="+",
                    type=int,
                    default=[36, 48, 56, 64],
                    # default=[50, 60, 70, 80, 90],
                    help='learning rate decay schedule')
parser.add_argument('--opt_level',
                    type=str,
                    default='O0',
                    choices=('O0', 'O1', 'O2', 'O3'))
parser.add_argument('--pretrain',
                    type=str,
                    default=None,
                    help='ckpt path')
parser.add_argument('--resume',
                    action='store_true',
                    help='resume training')
parser.add_argument('--test',
                    action='store_true',
                    help='test only flag')
parser.add_argument('--inference',
                    action='store_true',
                    help='do inference flag')
parser.add_argument('--use_val_set',
                    action='store_true',
                    default=False,
                    help='decides whether to use the validation set for testing (by default is the test set)')
parser.add_argument('--batch_size',
                    type=int,
                    default=12,
                    # default=3,
                    help='input batch size for training')
parser.add_argument('--max_depth',
                    type=float,
                    default=10.0,
                    # default=90.0,
                    help='maximum depth')
parser.add_argument('--augment',
                    type=bool,
                    default=True,
                    help='data augmentation')
parser.add_argument('--no_augment',
                    action='store_false',
                    dest='augment',
                    help='no augmentation')
parser.add_argument('--num_sample',
                    type=int,
                    default=500,
                    help='number of sparse samples')
parser.add_argument('--lidar_lines',
                    type=int,
                    default=64,
                    help='the extracted lidar lines')
parser.add_argument('--test_crop',
                    action='store_true',
                    default=False,
                    help='crop for test')
parser.add_argument('--save_freq',
                    type=int,
                    default=2,
                    # default=3,
                    help='the number of epochs for one model saving')
parser.add_argument('--save_objective', type=str, default='MAE', choices=['MAE', 'RMSE', 'REL'])

# Summary
parser.add_argument('--num_summary',
                    type=int,
                    default=4,
                    help='maximum number of summary images to save')


# Optimizer
parser.add_argument('--lr',
                    type=float,
                    default=0.00024,
                    help='learning rate')
parser.add_argument('--gamma',
                    type=float,
                    default=0.5,
                    help='learning rate multiplicative factors')
parser.add_argument('--optimizer',
                    default='ADAMW',
                    choices=('SGD', 'ADAM', 'ADAMW', 'RMSPROP'),
                    help='optimizer to use (SGD | ADAM | RMSprop | ADAMW)')
parser.add_argument('--scheduler',
                    default='cos',
                    choices=('cos', 'multi_step', 'step', 'exp'),
                    help='optimizer to use (cos | multi step | step | exp)')
parser.add_argument('--momentum',
                    type=float,
                    default=0.9,
                    help='SGD momentum')
parser.add_argument('--betas',
                    type=tuple,
                    default=(0.9, 0.999),
                    help='ADAM | ADAMW beta')
parser.add_argument('--epsilon',
                    type=float,
                    default=1e-8,
                    help='ADAM | ADAMW epsilon for numerical stability')
parser.add_argument('--weight_decay',
                    type=float,
                    default=0.01,
                    help='weight decay')
parser.add_argument('--warm_up',
                    action='store_true',
                    default=True,
                    help='do lr warm up during the 1st epoch')
parser.add_argument('--no_warm_up',
                    action='store_false',
                    dest='warm_up',
                    help='no lr warm up')

parser.add_argument('--step_size',
                    type=int,
                    default=40)

# Logs
parser.add_argument('--log_dir',
                    type=str,
                    default='../experiments/',
                    help='dir for log')
parser.add_argument('--print_freq',
                    type=int,
                    default=1,
                    help='print frequency of tqdm')
parser.add_argument('--save',
                    type=str,
                    default='trial',
                    help='file name to save')
parser.add_argument('--save_full',
                    action='store_true',
                    default=False,
                    help='save optimizer, scheduler and amp in '
                         'checkpoints (large memory)')
parser.add_argument('--save_image',
                    action='store_true',
                    default=False,
                    help='save images for test')
parser.add_argument('--save_result_only',
                    action='store_true',
                    default=False,
                    help='save result images only with submission format')

# transunet configs

parser.add_argument('--trans_n_channels',
                    type=int,
                    default=15)

parser.add_argument('--trans_n_classes',
                    type=int,
                    default=3)

parser.add_argument('--residual_num',
                    type=int,
                    default=8)

parser.add_argument('--trans_dim',
                    type=int,
                    default=64)

parser.add_argument('--trans_bilinear',
                    action='store_true')

parser.add_argument('--skip_res',
                    action='store_true')

parser.add_argument('--dropout',
                    type=float,
                    default=0.)

parser.add_argument('--norm',
                    type=str,
                    default='bn')

parser.add_argument('--pretrain_path_comp',
                    type=str,
                    default='bn')

parser.add_argument('--use_single',
                    action='store_true')

parser.add_argument('--depth_type',
                    type=int,
                    default=0)


args = parser.parse_args()
args.num_gpus = len(args.gpus.split(','))

save_dir = args.log_dir + args.save
args.save_dir = save_dir
