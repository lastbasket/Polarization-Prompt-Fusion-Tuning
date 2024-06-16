# -- dataset imports --
from datasets.hammer import HammerDataset
from datasets.hammer_single_depth import HammerSingleDepthDataset
# -- loss & metric imports --
from losses.l1l2loss import L1L2Loss
from summary.cfsummary import CompletionFormerSummary
from metric.cfmetric import CompletionFormerMetric
# -- pytorch stuff --
import torch
from torch import nn
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
# -- misc. utilities --
import numpy as np
import json
import time
import random
import os
import cv2
import copy
from tqdm import tqdm
from utils.mics import save_output
from utils.metrics import PDNEMetric
from utils.depth2normal import depth2norm
from utils.visualization_utils import *
# -- model imports --
import model
# -- training utilities --
from utils import train_utils
import apex
from apex import amp
from apex.parallel import DistributedDataParallel as DDP

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ---------------
# -- constants --
# ---------------
MODEL_CHOICES = ['PPFT', 'CompletionFormer']

# ------------------------------
# -- user provided parameters --
# ------------------------------
# -- first define the default values --
camera_matrix = np.array([[7.067553100585937500e+02, 0.000000000000000000e+00, 5.456326819328060083e+02],
                [0.000000000000000000e+00, 7.075133056640625000e+02, 3.899299663507044897e+02],
                [0.000000000000000000e+00, 0.000000000000000000e+00, 1.000000000000000000e+00]])
# -- then we allow users to pass-in --
from config import args as args_config

# ------------------------
# -- in-script utlities --
# ------------------------
def init_seed(seed=None):
    if seed is None:
        seed = args_config.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)

def check_args(args):
    new_args = args
    if args.pretrain is not None:
        assert os.path.exists(args.pretrain), \
            "file not found: {}".format(args.pretrain)

        if args.resume:
            checkpoint = torch.load(args.pretrain, map_location='cpu')

            new_args = checkpoint['args']
            new_args.test = args.test
            new_args.pretrain = args.pretrain
            new_args.dir_data = args.dir_data
            new_args.resume = args.resume
            new_args.layer0=False
            new_args.save_freq = 2

    return new_args

def load_pretrain(args, net, ckpt):
    assert os.path.exists(ckpt), \
            "file not found: {}".format(ckpt)

    checkpoint = torch.load(ckpt, map_location='cpu')
    key_m, key_u = net.load_state_dict(checkpoint['net'], strict=False)

    if key_u:
        print('Unexpected keys :')
        print(key_u)

    if key_m:
        print('Missing keys :')
        print(key_m)
        raise KeyError

    print('Checkpoint loaded from {}!'.format(ckpt))

    return net

# -- the training function --
def train(gpu, args):
    # -- initialize distributed training, rank=0 is used for logging --
    dist.init_process_group(backend='nccl', init_method=f'tcp://localhost:{args.port}',
                            world_size=args.num_gpus, rank=gpu)
    torch.cuda.set_device(gpu)

    # -- instantiate the dataloaders --
    print("==> Creating dataset...")
    dataset = HammerDataset(args, "train")

    sampler_train = DistributedSampler(
        dataset, num_replicas=args.num_gpus, rank=gpu)

    batch_size = args.batch_size

    loader_train = DataLoader(
        dataset=dataset, batch_size=batch_size, shuffle=False,
        num_workers=args.num_threads, pin_memory=True, sampler=sampler_train,
        drop_last=True)
    
    loader_val = DataLoader(
        dataset=HammerDataset(args, "val"), batch_size=1, shuffle=False,
        num_workers=args.num_threads, pin_memory=True)
    print("==> Dataset created.")

    # -- instantiate the model --
    print("==> Initializing model...")
    if args.model not in MODEL_CHOICES:
        raise TypeError(args.model, MODEL_CHOICES)
    
    net = getattr(model, args.model)(args)
    net.cuda(gpu)
    print("==> Model initialized.")
    
    # -- load pretrained weights in case of fine-tuning an existing PPFT, e.g. resuming --
    if gpu == 0:
        if args.pretrain is not None:
            assert os.path.exists(args.pretrain), \
                "file not found: {}".format(args.pretrain)

            checkpoint = torch.load(args.pretrain, map_location='cpu')
            net.load_state_dict(checkpoint['net'])

            print('Load network parameters from : {}'.format(args.pretrain))

    # -- instantiate the losses --
    loss = L1L2Loss(args)
    loss.cuda(gpu)

    # -- instantiate the optimizer --
    optimizer, scheduler = train_utils.make_optimizer_scheduler(args, net, len(loader_train))
    net = apex.parallel.convert_syncbn_model(net)
    net, optimizer = amp.initialize(net, optimizer, opt_level=args.opt_level, verbosity=0)
    
    init_epoch = 1
    # -- initialize various parameters in case of resuming --
    if gpu == 0:
        if args.pretrain is not None:
            if args.resume:
                print('Resume:', args.resume)
                try:
                    optimizer.load_state_dict(checkpoint['optimizer'])
                    scheduler.load_state_dict(checkpoint['scheduler'])
                    amp.load_state_dict(checkpoint['amp'])
                    init_epoch = checkpoint['epoch']

                    print('Resume optimizer, scheduler and amp '
                          'from : {}'.format(args.pretrain))
                except KeyError:
                    print('State dicts for resume are not saved. '
                          'Use --save_full argument')

            del checkpoint

    net = DDP(net)

    # -- instantiate the metrics --
    metric = PDNEMetric(args)

    # -- create directories for saving results --
    if gpu == 0:
        os.makedirs(args.save_dir, exist_ok=True)
        os.makedirs(args.save_dir + '/train', exist_ok=True)
        writer_train = SummaryWriter(log_dir=args.save_dir + '/' + 'train')
        total_losses = np.zeros(np.array(loss.loss_name).shape)
        total_metrics = np.zeros(np.array(metric.metric_name).shape)

        with open(args.save_dir + '/args.json', 'w') as args_json:
            json.dump(args.__dict__, args_json, indent=4)
    
    # -- check for training warm-ups --
    if args.warm_up:
        warm_up_cnt = 0.0
        warm_up_max_cnt = len(loader_train)+1.0

    # -- decide the training sample index to track for sanity check --
    rand_idx = np.random.randint(0, args.batch_size)
    tracked_sample = None
    
    # -- training loop starts here --
    print("==> Training begins.")
    best_save_objective_value = 1e10 # we assume save objective is always lower the better (e.g. MAE, RMSE, etc.)
    best_save_objective_epoch = 1
    
    for epoch in range(init_epoch, args.epochs+1):
        net.train()

        sampler_train.set_epoch(epoch)
        if gpu == 0:
            current_time = time.strftime('%y%m%d@%H:%M:%S')

            list_lr = []
            for g in optimizer.param_groups:
                list_lr.append(g['lr'])

        num_sample = len(loader_train) * \
            loader_train.batch_size * args.num_gpus

        if gpu == 0:
            pbar = tqdm(total=num_sample)
            log_cnt = 0.0
            log_loss = 0.0

        # TODO: Check if this exists in the original author's code and if this is appropriate at all
        init_seed(seed=int(time.time()))

        # -- go over batches --
        for batch, sample in enumerate(loader_train):
            sample = {key: val.cuda(gpu) for key, val in sample.items()
                    if (val is not None) and key != 'base_name'}

            sample["input"] = sample["rgb"]
            if tracked_sample is None:
                tracked_sample = copy.deepcopy(sample)

            # -- update learning rates according to the warm-up scheme --
            if epoch == 1 and args.warm_up:
                warm_up_cnt += 1

                for param_group in optimizer.param_groups:
                    lr_warm_up = param_group['initial_lr'] \
                        * warm_up_cnt / warm_up_max_cnt
                    param_group['lr'] = lr_warm_up

            optimizer.zero_grad()

            # -- forward pass --
            output = net(sample)
            
            output['pred'] = output['pred']  * sample['net_mask']
            sample['gt'] = sample['gt'] 

            loss_sum, loss_val = loss(sample, output)
            loss_sum = loss_sum / loader_train.batch_size
            loss_val = loss_val / loader_train.batch_size

            # -- backward pass and parameter update --
            with amp.scale_loss(loss_sum, optimizer) as scaled_loss:
                scaled_loss.backward()
            optimizer.step()

            # -- per iteration logging --
            if gpu == 0:
                for i in range(len(loss.loss_name)):
                    total_losses[i] += loss_val[0][i]

                log_cnt += 1
                log_loss += loss_sum.item()

                e_string = f"{(log_loss/log_cnt):.4f}"
                if batch % args.print_freq == 0:
                    pbar.set_description(e_string)
                    pbar.update(loader_train.batch_size * args.num_gpus)

        # -- per-epoch logging --
        if gpu == 0:
            pbar.close()

            # -- save visualization of the tracked training sample (so that you can check training process qualitatively) --
            output = net(tracked_sample)
            output['pred'] = output['pred'] * tracked_sample['net_mask']

            folder_name = os.path.join(args.save_dir, "train", "epoch-{}".format(str(epoch))) # saved under "./experiments/<experiment_dir>/epoch-<epoch>"
            os.makedirs(folder_name, exist_ok=True)
            
            save_visualization(output["pred"][rand_idx], tracked_sample["gt"][rand_idx], tracked_sample["dep"][rand_idx], folder_name)
            
            # -- save normal map too in case we are interested in --
            if args.use_norm:
                gt_norm_vis = norm_to_colormap(tracked_sample['norm'][rand_idx])
                cv2.imwrite(os.path.join(folder_name, "gt_norm.png"), gt_norm_vis)
                
                pred_norm_vis = norm_to_colormap(normal_from_dep[rand_idx])
                cv2.imwrite(os.path.join(folder_name, "pred_norm.png"), pred_norm_vis)

            # -- log training losses --
            for i in range(len(loss.loss_name)):
                writer_train.add_scalar(
                    loss.loss_name[i], total_losses[i] / len(loader_train), epoch)

            # -- log learning rate --
            writer_train.add_scalar('lr', scheduler.get_last_lr()[0], epoch)

            # -- save the model checkpoint at the right timing --
            if ((epoch) % args.save_freq == 0) or epoch == args.epochs:
                vis_folder = os.path.join(args.save_dir, "val", "epoch-{}".format(str(epoch)))
                os.makedirs(vis_folder, exist_ok=True)
                print("==> Start validation of epoch {}...".format(epoch))
                metric_res = validation(args, net, loader_val, vis_folder, epoch, writer_train)
                print("==> Finished validation of epoch {}.".format(epoch))
                
                is_best = False
                for i, met_name in enumerate(metric.metric_name):
                    if met_name != args.save_objective:
                        continue
                    
                    print("==> Validation result on the save objective {} is {:.5f}".format(args.save_objective, metric_res[i]))
                    if best_save_objective_value > metric_res[i]:
                        best_save_objective_value = metric_res[i]
                        best_save_objective_epoch = epoch
                        is_best = True
                        
                    print('==> Current best model is at epoch-{} with metric value {}: {:.5f}'.format(best_save_objective_epoch, args.save_objective, best_save_objective_value))
                    break
                
                if args.save_full or epoch == args.epochs:
                    state = {
                        'net': net.module.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'amp': amp.state_dict(),
                        'args': args,
                        'epoch': epoch
                    }
                else:
                    state = {
                        'net': net.module.state_dict(),
                        'args': args,
                        'epoch': epoch
                    }

                torch.save(
                    state, '{}/model_{:05d}.pt'.format(args.save_dir, epoch))
                
                if is_best:
                    torch.save(
                    state, '{}/model_best.pt'.format(args.save_dir))
                
        # -- update learning rate --
        scheduler.step()

        if gpu == 0:
            total_losses = np.zeros(np.array(loss.loss_name).shape)
            total_metrics = np.zeros(np.array(metric.metric_name).shape)

    if gpu == 0:
        writer_train.close()

def validation(args, net, loader_val, vis_folder, epoch_idx, summary_writer=None):
    net = nn.DataParallel(net)
    net.eval()
    
    metric = CompletionFormerMetric(args)
    total_metrics = None
    num_sample = len(loader_val)*loader_val.batch_size

    pbar = tqdm(total=num_sample)
    init_seed()

    for batch, sample in enumerate(loader_val):
        sample = {key: val.cuda() for key, val in sample.items()
                  if (val is not None) and key != 'basename'}
        
        with torch.no_grad():
            output = net(sample)

        metric_val = metric.evaluate(sample, output, 'test')

        if total_metrics is None:
            total_metrics = metric_val[0]
        else:
            total_metrics += metric_val[0]

        current_time = time.strftime('%y%m%d@%H:%M:%S')
        error_str = '{} | Test'.format(current_time)
        if batch % args.print_freq == 0:
            pbar.set_description(error_str)
            pbar.update(loader_val.batch_size)
        
        metric_dict = {}
        count = 0
        for m in metric.metric_name:
            metric_dict[m] = metric_val[0][count].detach().cpu().numpy().astype(float).tolist()
            count += 1

    pbar.close()

    metric_avg = total_metrics / num_sample
    if summary_writer is not None:
        for i, metric_name in enumerate(metric.metric_name):
            summary_writer.add_scalar('val/{}'.format(metric_name), metric_avg[i], epoch_idx)
    
    save_visualization(output["pred"][0], sample["gt"][0], sample["dep"][0], vis_folder)
    
    return metric_avg

# -- the tssting functions --
def test_one_model(args, net, loader_test, save_samples, epoch_idx=0, summary_writer=None, result_dict=None, idx=0):
    net = nn.DataParallel(net)

    metric = CompletionFormerMetric(args)

    vis_dir = os.path.join(args.save_dir, "{}".format('all' if (not args.use_single) else ['stereo', 'd-tof', 'i-tof'][args.depth_type]), "epoch-{}".format(str(epoch_idx)), 'visualization')
    try:
        os.makedirs(vis_dir, exist_ok=True)
    except OSError:
        pass
    
    net.eval()

    num_sample = len(loader_test)*loader_test.batch_size

    pbar = tqdm(total=num_sample)

    t_total = 0

    init_seed()
    total_metrics = None

    for batch, sample in enumerate(loader_test):
        sample = {key: val.cuda() for key, val in sample.items()
                  if (val is not None) and key != 'basename'}
        
        t0 = time.time()
        with torch.no_grad():
            output = net(sample)
        t1 = time.time()

        t_total += (t1 - t0)

        metric_val = metric.evaluate(sample, output, 'test')

        if total_metrics is None:
            total_metrics = metric_val[0]
        else:
            total_metrics += metric_val[0]

        current_time = time.strftime('%y%m%d@%H:%M:%S')
        error_str = '{} | Test'.format(current_time)
        if batch % args.print_freq == 0:
            pbar.set_description(error_str)
            pbar.update(loader_test.batch_size)
        
        metric_dict = {}
        count = 0
        for m in metric.metric_name:
            metric_dict[m] = metric_val[0][count].detach().cpu().numpy().astype(float).tolist()
            count += 1
            
        if result_dict is not None:
            result_dict[f's{idx+batch}.png'] = metric_dict

        if batch in save_samples:
            dep = sample['dep'] # in m
            gt = sample['gt'] # in m
            pred = output['pred'] # in m

            pred = pred * sample['net_mask']

            this_vis_dir = os.path.join(vis_dir, 'sample-{}'.format(batch))
            os.makedirs(this_vis_dir, exist_ok=True)
            save_visualization(pred[0], gt[0], dep[0], this_vis_dir)
    
    pbar.close()

    t_avg = t_total / num_sample
    print('Elapsed time : {} sec, '
          'Average processing time : {} sec'.format(t_total, t_avg))

    metric_avg = total_metrics / num_sample
    if summary_writer is not None:
        for i, metric_name in enumerate(metric.metric_name):
            summary_writer.add_scalar('test/{}'.format(metric_name), metric_avg[i], epoch_idx)

    return metric_avg

def test(args):
    # -- instantiate the model --
    # TODO: Share this part of code with that in the training code, avoid copy-pasting
    if args.model not in MODEL_CHOICES:
        raise TypeError(args.model, MODEL_CHOICES)
    
    net = getattr(model, args.model)(args)
    net.cuda()
        
    # -- prepare the dataset --
    # TODO: Make this smarter by combining into a single dataset class --
    if args.use_single:
        data_test = HammerSingleDepthDataset(args, 'test' if not args.use_val_set else 'val')
    else:
        data_test = HammerDataset(args, 'test' if not args.use_val_set else 'val')

    result_dict = {}

    loader_test = DataLoader(dataset=data_test, batch_size=1,
                             shuffle=False, num_workers=args.num_threads)

    # -- test model(s), depending on if one or multiple checkpoints are provided --
    if args.pretrain is not None:
        summary_writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'logs'))

        net = load_pretrain(args, net, args.pretrain)
        save_samples = np.arange(len(loader_test))

        test_one_model(args, net, loader_test, save_samples, result_dict=result_dict, summary_writer=summary_writer)
        summary_writer.close()
    elif args.pretrain_list_file is not None:
        summary_writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'logs'))

        pretrain_list = open(args.pretrain_list_file, 'r').read().split("\n")

        save_samples = np.arange(len(loader_test))
        
        line_idx = 0

        metric = CompletionFormerMetric(args)
        for line in pretrain_list:
            print("==> Testing checkpoint: {}".format(line))
            
            epoch_idx = line.split(" - ")[0]
            ckpt = line.split(" - ")[1]
            net = load_pretrain(args, net, ckpt)
            metric_avg = test_one_model(args, net, loader_test, save_samples, epoch_idx, summary_writer, result_dict=result_dict, idx=line_idx)
            line_idx += 1
            
            result_file_path = os.path.join(args.save_dir, "{}".format('all' if (not args.use_single) else ['stereo', 'd-tof', 'i-tof'][args.depth_type]), 'results.txt')
            result_file = open(result_file_path, 'a')
            
            result_file.write("=============================\nCheckpoint: {} @ Epoch-{}\n".format(ckpt, epoch_idx))
            for i, met_name in enumerate(metric.metric_name):
                result_file.write("{}: {:.6f}\n".format(met_name, metric_avg[i]))
            result_file.write("=============================\n\n")
            print("==> Results written to {}".format(result_file_path))
        
        summary_writer.close()
    else:
        raise Exception("No checkpoint or checkpoint list provided, please provide one for testing")
    
def inference(args):
    # -- instantiate the model --
    if args.model not in MODEL_CHOICES:
        raise TypeError(args.model, MODEL_CHOICES)
    
    net = getattr(model, args.model)(args)
    net.cuda()
        
    # -- prepare the dataset --
    # TODO: Make this smarter by combining into a single dataset class --
    data_val = HammerDataset(args, 'val')
    data_test = HammerDataset(args, 'test')

    result_dict = {}

    loader_val = DataLoader(dataset=data_val, batch_size=1,
                             shuffle=False, num_workers=args.num_threads)
    loader_test = DataLoader(dataset=data_test, batch_size=1,
                             shuffle=False, num_workers=args.num_threads)

    # -- test model(s), depending on if one or multiple checkpoints are provided --
    if args.pretrain_list_file is not None:
        summary_writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'logs'))

        pretrain_list = open(args.pretrain_list_file, 'r').read().split("\n")

        save_samples = np.arange(len(loader_test))
        
        line_idx = 0

        metric = CompletionFormerMetric(args)
        for line in pretrain_list:
            print("==> Running checkpoint: {}".format(line))
            
            epoch_idx = line.split(" - ")[0]
            ckpt = line.split(" - ")[1]
            net = load_pretrain(args, net, ckpt)
            for stage in ['val', 'test']:
                save_dir = os.path.join(args.save_dir, stage)
                metric_avg = inference_one_model(args, save_dir, net, loader_test if stage == 'test' else loader_val, save_samples, epoch_idx, summary_writer, result_dict=result_dict, idx=line_idx)
            line_idx += 1
        
        summary_writer.close()
    else:
        raise Exception("No checkpoint list provided, please provide one for testing")

def inference_one_model(args, save_dir, net, loader_test, save_samples, epoch_idx=0, summary_writer=None, result_dict=None, idx=0):
    net = nn.DataParallel(net)

    metric = CompletionFormerMetric(args)

    raw_data_dir = os.path.join(save_dir, "epoch-{}".format(str(epoch_idx)), 'raw_data')
    try:
        os.makedirs(raw_data_dir, exist_ok=True)
    except OSError:
        pass
    
    net.eval()

    num_sample = len(loader_test)*loader_test.batch_size

    pbar = tqdm(total=num_sample)

    t_total = 0

    init_seed()
    total_metrics = None

    for batch, sample in enumerate(loader_test):
        sample = {key: val.cuda() for key, val in sample.items()
                  if (val is not None) and key != 'basename'}
        
        t0 = time.time()
        with torch.no_grad():
            output = net(sample)
        t1 = time.time()

        t_total += (t1 - t0)

        metric_val = metric.evaluate(sample, output, 'test')

        if total_metrics is None:
            total_metrics = metric_val[0]
        else:
            total_metrics += metric_val[0]

        current_time = time.strftime('%y%m%d@%H:%M:%S')
        error_str = '{} | Test'.format(current_time)
        if batch % args.print_freq == 0:
            pbar.set_description(error_str)
            pbar.update(loader_test.batch_size)
        
        metric_dict = {}
        count = 0
        for m in metric.metric_name:
            metric_dict[m] = metric_val[0][count].detach().cpu().numpy().astype(float).tolist()
            count += 1
            
        if result_dict is not None:
            result_dict[f's{idx+batch}.png'] = metric_dict

        if batch in save_samples:
            dep = sample['dep'] # in m
            gt = sample['gt'] # in m
            pred = output['pred'] # in m

            pred = pred * sample['net_mask']

            this_raw_data_dir = os.path.join(raw_data_dir, 'sample-{}'.format(batch))
            os.makedirs(this_raw_data_dir, exist_ok=True)
            save_raw_data(pred[0], gt[0], dep[0], this_raw_data_dir, pol=sample['pol'], rgb=sample['rgb'])
    
    pbar.close()

    t_avg = t_total / num_sample
    print('Elapsed time : {} sec, '
          'Average processing time : {} sec'.format(t_total, t_avg))

    metric_avg = total_metrics / num_sample
    if summary_writer is not None:
        for i, metric_name in enumerate(metric.metric_name):
            summary_writer.add_scalar('test/{}'.format(metric_name), metric_avg[i], epoch_idx)

    return metric_avg

# -- main --
def main(args):
    init_seed()
    if not args.test and not args.inference:
        if args.no_multiprocessing:
            train(0, args)
        else:
            assert args.num_gpus > 0

            spawn_context = mp.spawn(train, nprocs=args.num_gpus, args=(args,),
                                     join=False)

            while not spawn_context.join():
                pass

            for process in spawn_context.processes:
                if process.is_alive():
                    process.terminate()
                process.join()

        args.pretrain = '{}/model_best.pt'.format(args.save_dir)

    if args.inference:
        inference(args)
        
    if args.test:
        test(args)

# -- main execution --
if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = args_config.gpus
    os.environ["MASTER_ADDR"] = args_config.address
    os.environ["MASTER_PORT"] = args_config.port
    args_main = check_args(args_config)

    print('\n\n=== Arguments ===')
    cnt = 0
    for key in sorted(vars(args_main)):
        print(key, ':',  getattr(args_main, key), end='  |  ')
        cnt += 1
        if (cnt + 1) % 5 == 0:
            print('')
    print('\n')
    time.sleep(5)

    main(args_main)
