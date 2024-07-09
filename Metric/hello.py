import torch
from torch import distributed as dist
import torch._dynamo.config
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils import benchmark
import numpy as np 
import time

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from data import get_cifar, setup_distribute, cleanup_distribute, DataAugmentation, ResizeAndNormalize
from Helper import save_individual_image
from VGG import VGG19
from TicketModels import TicketCNN

import torch._dynamo as dynamo
dynamo.config.capture_scalar_outputs = True

torch.set_float32_matmul_precision = 'high'
torch.backends.cudnn.benchmark = True

torch.jit.enable_onednn_fusion(True)

# NOTE: if these fail asserts submit a PR to increase them
torch._inductor.runtime.hints.TRITON_MAX_BLOCK["X"] =  4096 # tmp solution to triton assert fail; prev 2048

#from posix import fspath
#torch.compiler.allow_in_graph(fspath) # temp solution to graph break in vision dataset

def temp(rank, world_size):

    try:

        torch.cuda.set_device(rank)

        setup_distribute(rank, world_size)
        
        dt, dv = get_cifar(rank, world_size) 

        dataAug = torch.jit.script(DataAugmentation())
        preprocess = torch.jit.script(ResizeAndNormalize())

        torch._print("Got Data")

        dynamo.reset()

        model = VGG19()
        
        model.cuda()

        model = DDP(model, device_ids = [rank],
                    output_device = rank, 
                    gradient_as_bucket_view = True,
                    find_unused_parameters = True)

        model = torch.compile(model)
        
        torch._print("Got Cuda Model")

        T = TicketCNN(model, rank)

        del model

        T.build(torch.optim.SGD(T.m.module.parameters(), 0.01, momentum = 0.9), 
                data_augmentation_transform = dataAug,
                preprocess_transform = preprocess, 
                weight_decay = 1e-6, scaler = True)


        print(all([param.device == torch.cuda.current_device] for name, param in T.m.named_buffers()))
        print(all([param.device == torch.cuda.current_device] for name, param in T.m.named_parameters()))
        if rank == 0:
            #with torch.profiler.profile as prof:
            dynamo.explain(T.summary)(batch_size = 128)
        
        start = time.time()
        res = T.evaluate(dt)
        if rank == 0: 
            torch._print(str(res))
            torch._print(str(time.time() - start))
            start = time.time()
        res = T.evaluate(dv)
        if rank == 0: 
            torch._print(str(res))
            torch._print(str(time.time() - start))
            start = time.time()
        
        start = time.time()
        res = T.evaluate(dt)
        if rank == 0: 
            torch._print(str(res))
            torch._print(str(time.time() - start))
            start = time.time()
        res = T.evaluate(dv)
        if rank == 0: 
            torch._print(str(res))
            torch._print(str(time.time() - start))
            start = time.time()
        
        start = time.time()
        res = T.evaluate(dt)
        if rank == 0: 
            torch._print(str(res))
            torch._print(str(time.time() - start))
            start = time.time()
        res = T.evaluate(dv)
        if rank == 0: 
            torch._print(str(res))
            torch._print(str(time.time() - start))
            start = time.time()
        
        start = time.time()
        res = T.evaluate(dt)
        if rank == 0: 
            torch._print(str(res))
            torch._print(str(time.time() - start))
            start = time.time()
        res = T.evaluate(dv)
        if rank == 0: 
            torch._print(str(res))
            torch._print(str(time.time() - start))
            start = time.time()
        
        start = time.time()
        res = T.evaluate(dt)
        if rank == 0: 
            torch._print(str(res))
            torch._print(str(time.time() - start))
            start = time.time()
        res = T.evaluate(dv)
        if rank == 0: 
            torch._print(str(res))
            torch._print(str(time.time() - start))
            start = time.time()
        

        """
        timer = benchmark.Timer(
        stmt="T.evaluate(dt); T.evaluate(dv); T.evaluate(dt); T.evaluate(dv);",
        globals={"T": T, "dt": dt, "dv": dv},
        label="Evaluation Time")

        print(timer.blocked_autorange(min_run_time = 240.0))"""

    finally:
        
        cleanup_distribute()

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(temp, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()