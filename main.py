import torch
import torch.multiprocessing as mp
import torch._dynamo as dynamo
import torch.distributed as dist

import os
import sys

import matplotlib
matplotlib.use('Agg') 

def setup_distribute(rank, world_size): 
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank = rank, world_size = world_size)

def cleanup_distribute():
    dist.destroy_process_group()

def dist_wrapper(rank, world_size, func, name: str):
    setup_distribute(rank, world_size)
    torch.cuda.set_device(rank)
    set_deterministic()
    try:
        func(rank, world_size, name)
    finally:
        cleanup_distribute()

def set_dynamo_cfg():
    dynamo.config.capture_scalar_outputs = True
    dynamo.config.cache_size_limit = 256
    dynamo.config.guard_nn_modules = True
    dynamo.config.suppress_errors = True

def set_non_deterministic():
    torch.backends.cudnn.benchmark = True

def set_deterministic():
    torch.manual_seed(42)
    import random
    import numpy as np
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def main(func, name: str):
    world_size = torch.cuda.device_count()
    mp.spawn(dist_wrapper, args=(world_size, func, name), nprocs=world_size, join=True)

if __name__ == "__main__":

    if len(sys.argv) != 2:
        raise ValueError("Must pass exactly one name. Usage should be python main.py $NAME")

    name = sys.argv[1]

    set_dynamo_cfg()

    from training import deterministic_test
    main(deterministic_test.main, name)
