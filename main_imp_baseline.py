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
    parser = argparse.ArgumentParser(description="Run a IMP training experiment.")

    parser.add_argument('name', type=str, help='The base name for the experiment logs and files.')

    parser.add_argument('--model', type=str, default='resnet20', choices=['resnet20', 'vgg16', 'resnet50'],
                        help='Model architecture to use (default: resnet20).')
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100', 'imagenet'],
                        help='Dataset to use (default: cifar10).')
    parser.add_argument('--time', type=str, default='rewind', choices=['init', 'rewind'],
                        help='Sparsity time strategy (init or rewind) (default: rewind).')
    
    parser.add_argument('--num', type=int, default=1,
                        help='Number of independent experiments to run (default: 1).')
    parser.add_argument('--prune_iterations', type=int, default=1,
                        help='Number of prune iterations to run; i.e. 2 will give maximum prune density of (rate)**2  (default: 1).')

    parser.add_argument('--partitioned_jobs', action='store_true',
                        help='Run each prune iteration in a separate scheduled job.')
    parser.add_argument('--current_iteration', type = int, default = 0, help = "Need to queue iterations 0 through prune_iterations, inclusive.")

    args = parser.parse_args()
    
    main_kwargs = args
    exp_name = main_kwargs.name
    num_exp = main_kwargs.num
    
    del main_kwargs.name
    del main_kwargs.num

    return exp_name, num_exp, main_kwargs

if __name__ == "__main__":

    from training import imp_training
    
    exp_name, num_exp, main_kwargs = parse_args()

    for exp in range(num_exp):
        print("--------------------------------------------------------------")
        print("EXPERIMENT NUMBER ", exp)
        print("--------------------------------------------------------------") 
        main(imp_training.main, exp_name + f"_{exp}", main_kwargs)