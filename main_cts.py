import torch
import torch.multiprocessing as mp
import torch._dynamo as dynamo
import torch.distributed as dist

import os
import sys
import argparse

import random

import matplotlib
matplotlib.use('Agg') 

def setup_distribute(rank, world_size): 
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank = rank, world_size = world_size)

def cleanup_distribute():
    dist.destroy_process_group()

def dist_wrapper(rank, world_size, func, name: str, args):
    setup_distribute(rank, world_size)
    torch.cuda.set_device(rank)
    set_dynamo_cfg()
    set_non_deterministic()
    try:
        func(rank, world_size, name, args)
    finally:
        cleanup_distribute()

def set_dynamo_cfg():
    dynamo.config.cache_size_limit = 256
    dynamo.config.guard_nn_modules = True

def set_non_deterministic():
    torch.backends.cudnn.benchmark = True

def reset_deterministic(SEED):
    import numpy as np
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(func, name: str, args):
    world_size = torch.cuda.device_count()
    mp.spawn(dist_wrapper, args=(world_size, func, name, args), nprocs=world_size, join=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Run a concrete sparsification experiment.")

    parser.add_argument('name', type=str, help='The base name for the experiment logs and files.')

    parser.add_argument('--model', type=str, default='resnet20',
                        help='Model architecture to use (default: resnet20).')
    parser.add_argument('--no_batchnorm', action='store_true',
                        help='Whether to disable batch normalization (default: False).')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet', 'tiny-imagenet'],
                        help='Dataset to use (default: cifar10).')

    parser.add_argument('--criteria', type=str, default='kldlogit', 
                        choices=['loss', 'deltaloss', 'gradnorm', 'kldlogit', 'msefeature', 'gradmatch'],
                        help='Sparsification criteria (default: kldlogit).')
    parser.add_argument('--when_to_prune', type=str, default='rewind', choices=['init', 'rewind'],
                        help='Sparsity time strategy (init or rewind) (default: rewind).')
    parser.add_argument('--gradstep', type=str, default='gradbalance', choices=['gradbalance', 'lagrange'],
                        help='Gradient approach (default: gradbalance).')
    parser.add_argument('--duration', type=str, default='long', choices=['short', 'long', 'half'],
                        help='Length of concrete optimization (default: long).')
    
    parser.add_argument('--num', type=int, default=1,
                        help='Number of independent experiments to run (default: 1).')
    parser.add_argument('--sparsities', nargs='*', type=float, default=None,
                        help='Optional list of density percentages. If not provided, a default range is used.')


    parser.add_argument('--learning_rate', type=float, default = argparse.SUPPRESS,
                        help='Learning rate for training (overrides dataset default).')
    parser.add_argument('--epochs', type=int, default = argparse.SUPPRESS,
                        help='Number of training epochs (overrides dataset default).')
    parser.add_argument('--warmup_epochs', type=int, default = argparse.SUPPRESS,
                        help='Number of warmup epochs (overrides dataset default).')
    parser.add_argument('--reduce_epochs', nargs='*', type=int, default = argparse.SUPPRESS,
                        help='Epochs to reduce learning rate (overrides dataset default).')
    parser.add_argument('--weight_decay', type=float, default = argparse.SUPPRESS,
                        help='Weight decay for optimizer (overrides dataset default).')
    parser.add_argument('--momentum', type=float, default=argparse.SUPPRESS,
                        help='Momentum for optimizer (overrides default 0.9).')
    parser.add_argument('--rewind_epoch', type=int, default=argparse.SUPPRESS,
                        help='Epoch to rewind to (overrides dataset/model default).')
    parser.add_argument('--optimizer', type=str, default=argparse.SUPPRESS,
                        help='Optimizer to use (overrides default sgd).')
    parser.add_argument('--gradient_clipnorm', type=float, default=argparse.SUPPRESS,
                        help='Gradient clipping norm (overrides default 2.0).')
    parser.add_argument('--scale_loss', type=bool, default=argparse.SUPPRESS,
                        help='Whether to scale loss by world size (overrides default True).')

    parser.add_argument('--batch_size', type=int, default=argparse.SUPPRESS,
                        help='Batch size including all gpus (overrides dataset default).')
    
    parser.add_argument('--save', action='store_true', 
                        help = "Whether to save the final state after training.")
    parser.add_argument('--save_init', action='store_true', 
                        help = "Whether to save the initial state before training.")

    args = parser.parse_args()

    if args.sparsities is not None: args.sparsities = [sp/100 for sp in args.sparsities]
    
    main_kwargs = args
    exp_name = main_kwargs.name
    num_exp = main_kwargs.num
    
    del main_kwargs.name
    del main_kwargs.num

    return exp_name, num_exp, main_kwargs

if __name__ == "__main__":

    from experiments import cts
    
    exp_name, num_exp, main_kwargs = parse_args()

    for exp in range(num_exp):
        print("--------------------------------------------------------------")
        print("EXPERIMENT NUMBER ", exp)
        print("--------------------------------------------------------------") 
        main(cts.main, exp_name + f"_{exp}", main_kwargs)