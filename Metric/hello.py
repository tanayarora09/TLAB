import torch
from torch import distributed as dist
import torch.multiprocessing as mp
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from data import get_cifar, setup_distribute, cleanup_distribute, DataAugmentation, ResizeAndNormalize
from Helper import save_individual_image

import torch._dynamo as dynamo
dynamo.config.capture_scalar_outputs = True
dynamo.config.cache_size_limit = 256

from posix import fspath
torch.compiler.allow_in_graph(fspath) # temp solution to graph break in vision dataset

def test_data_device(rank, world_size):

    torch.cuda.set_device(rank)

    setup_distribute(rank, world_size)
    
    dt, dv = get_cifar(rank, world_size) 

    dataAug = torch.jit.script(DataAugmentation())
    preprocess = torch.jit.script(ResizeAndNormalize())

    for step, (x, y) in enumerate(dt):
        x, y = x.to('cuda'), y.to('cuda')
        x = preprocess(x)
        x = dataAug(x)
        if step == 1: 
            print(x.shape, y.shape)
            print(x.device, y.device)
            print(x[0, 0, 0, 140:170])
            for i in range(len(x)):
                save_individual_image(x[i], f"./data_viewing/{i}.jpg")

        if step % 50 == 0:
            print(step)

    for step, (x, y) in enumerate(dt):
        x, y = x.to('cuda'), y.to('cuda')
        x = preprocess(x)
        x = dataAug(x)
        if step == 1: 
            print(x.shape, y.shape)
            print(x.device, y.device)
            print(x[0, 0, 0, 140: 170])
        
        if step % 50 == 0:
            print(step)

    cleanup_distribute()

def main():
    world_size = torch.cuda.device_count()
    torch.multiprocessing.spawn(test_data_device, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()