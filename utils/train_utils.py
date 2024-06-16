import os
import shutil
import torch.optim as optim
import torch.optim.lr_scheduler as lrs


def make_optimizer_scheduler(args, target, num_batches=-1):
    # optimizer
    kwargs_optimizer = {'lr': args.lr, 'weight_decay': args.weight_decay}

    if args.optimizer == 'SGD':
        optimizer_class = optim.SGD
        kwargs_optimizer['momentum'] = args.momentum
    elif args.optimizer == 'RMSPROP':
        optimizer_class = optim.RMSprop
        kwargs_optimizer['momentum'] = 0 # default
        kwargs_optimizer['weight_decay'] = 0 # default
    elif args.optimizer == 'ADAM':
        optimizer_class = optim.Adam
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    elif args.optimizer == 'ADAMW':
        optimizer_class = optim.AdamW
        kwargs_optimizer['betas'] = args.betas
        kwargs_optimizer['eps'] = args.epsilon
    else:
        raise NotImplementedError

    trainable = target.parameters()
    optimizer = optimizer_class(trainable, **kwargs_optimizer)

    if args.scheduler == 'multi_step':
        scheduler = lrs.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)
    elif args.scheduler == 'exp':
        scheduler = lrs.ExponentialLR(optimizer, gamma=args.gamma)
    elif args.scheduler == 'step':
        scheduler = lrs.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.scheduler == 'cos':
        scheduler = lrs.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=0)

    return optimizer, scheduler