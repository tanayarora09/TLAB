import torch
from torch import distributed as dist
import torch._dynamo.config
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils import benchmark
from torch.profiler import profile, record_function, ProfilerActivity
import numpy as np 
import time
import gc

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from data import get_cifar, setup_distribute, cleanup_distribute, DataAugmentation, ResizeAndNormalize
from Helper import save_individual_image
from VGG import VGG19
from TicketModels import TicketCNN

import torch._dynamo as dynamo
"""
dynamo.config.capture_scalar_outputs = True
dynamo.config.cache_size_limit = 256

torch.set_float32_matmul_precision = 'high'
torch.backends.cudnn.benchmark = True

torch.jit.enable_onednn_fusion(True)

#torch._inductor.runtime.hints.TRITON_MAX_BLOCK["X"] =  4096 # tmp solution to triton assert fail; prev 2048"""

dynamo.config.capture_scalar_outputs = True
dynamo.config.cache_size_limit = 256

torch.backends.cudnn.benchmark = True

from posix import fspath
torch.compiler.allow_in_graph(fspath) # temp solution to graph break in vision dataset

def train(rank, world_size):

    try:

        torch.cuda.set_device(rank)

        setup_distribute(rank, world_size)
    
        dataAug = torch.jit.script(DataAugmentation().to('cuda'))
        preprocess = torch.compile(ResizeAndNormalize().to('cuda')) #torch.jit.script(ResizeAndNormalize().to('cuda'))

        torch._print("Got Data")

        model = VGG19()
        
        model.to('cuda')

        model = DDP(model, device_ids = [rank],
                    output_device = rank, 
                    gradient_as_bucket_view = True,
                    find_unused_parameters = True)

        model = torch.compile(model)
        
        torch._print("Got Cuda Model")

        T = TicketCNN(model, rank)

        del model

        T.build(torch.optim.SGD(T.m.module.parameters(), 0.01, momentum = 0.9), 
                data_augmentation_transform = dataAug, preprocess_transform = preprocess, 
                weight_decay = 0.000125, scaler = True, clipnorm = 2.0)


        print(all([param.device == torch.cuda.current_device] for name, param in T.m.named_buffers()))
        print(all([param.device == torch.cuda.current_device] for name, param in T.m.named_parameters()))
        
        if rank == 0:
            T.summary(batch_size = 32)
            torch.cuda.empty_cache()
            gc.collect()
        
        dt, dv = get_cifar(rank, world_size) 

        T.evaluate(dt)
        
        if (rank == 0): print(T.get_eval_results())

        #with profile(activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA],
        #             record_shapes = True) as prof:
            
        #    with record_function("model_training"):

        logs = T.train_one(dt, dv, 5, 391, "TMP")

        T.evaluate(dt)

        if (rank == 0):
            print(T.get_eval_results())

        T.evaluate(dv)

        if (rank == 0):
            print(T.get_eval_results())
            print(logs)
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))

    finally:
        
        cleanup_distribute()

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()