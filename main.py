import torch
import torch.multiprocessing as mp
import torch._dynamo as dynamo
import torch.distributed as dist

import os
import sys

import random


import matplotlib
matplotlib.use('Agg') 

def setup_distribute(rank, world_size): 
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank = rank, world_size = world_size)

def cleanup_distribute():
    dist.destroy_process_group()

def dist_wrapper(rank, world_size, func, name: str, args: list, lock, shared_list):
    setup_distribute(rank, world_size)
    torch.cuda.set_device(rank)
    set_dynamo_cfg()
    set_non_deterministic()
    #reset_deterministic(SEED)
    try:
        func(rank, world_size, name, args, lock, shared_list)
    finally:
        cleanup_distribute()

def set_dynamo_cfg():
    #dynamo.config.capture_scalar_outputs = True
    dynamo.config.cache_size_limit = 256
    dynamo.config.guard_nn_modules = True
    #dynamo.config.suppress_errors = True

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

def main(func, name: str, args: list):#SEED: int):
    mp.set_start_method("spawn", force = True)
    lock = mp.Lock()
    manager = mp.Manager()
    shared_list = manager.list()
    world_size = torch.cuda.device_count()
    mp.spawn(dist_wrapper, args=(world_size, func, name, args, lock, shared_list), nprocs=world_size, join=True)

if __name__ == "__main__":

    if len(sys.argv) != 2 and len(sys.argv) != 4:
        raise ValueError("Must pass exactly one name. Usage should be python main.py $NAME $\"EPS,ITS,SIZE,PLAT,SPARSITY\" $NUM_EXPERIMENTS")

    name = sys.argv[1]

    try: 

        args = [float(item) for item in sys.argv[2].split(",")]
        num_exp = int(sys.argv[3])

    except IndexError:
        print("Wrong number of arguments. Using default arguments.")
        args = [160, 120, 96, 3, 0.012]
        num_exp = 1

    SEED: int = random.randint(0, 2**31)
    print("--------------------------------------------------------------")
    print("NO SEED: ", SEED)
    print("--------------------------------------------------------------")

    #with open(f"./logs/SEEDS/{name}_SEED.txt", "w") as f:
    #    f.write(str(SEED))
    #from training import normal

    #main(normal.main, name, args)
    
    from training import baseline
    for arg in args:
        for exp in range(num_exp):
            print("--------------------------------------------------------------")
            print("EXPERIMENT NUMBER ", exp)
            print("--------------------------------------------------------------")
            main(baseline.main, name + str(exp), [arg])
    