import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import torch._dynamo as dynamo

from TrainWrappers import VGGPOC

import gc

import matplotlib
matplotlib.use('Agg')

from data import get_cifar, setup_distribute, cleanup_distribute, DataAugmentation, Resize, Normalize, CenterCrop, view_data
from Helper import plot_logs
from VGG import VGG

dynamo.config.capture_scalar_outputs = True
dynamo.config.cache_size_limit = 256

torch.backends.cudnn.benchmark = True

def run(rank, world_size):

    try:

        torch.cuda.set_device(rank)

        setup_distribute(rank, world_size)
    
        dataAug = torch.jit.script(DataAugmentation().to('cuda'))
        resize = torch.jit.script(Resize().to('cuda'))
        normalize = torch.jit.script(Normalize().to('cuda'))
        center_crop = torch.jit.script(CenterCrop().to('cuda'))

        torch._print("Got Data")

        torch.cuda.empty_cache()
        gc.collect()

        dt, dv = get_cifar(rank, world_size)

        view_data(dt, rank, [resize, normalize, dataAug])
        

    finally:
        
        cleanup_distribute()

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(run, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()