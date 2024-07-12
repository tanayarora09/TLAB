import torch
from torch import nn
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import torch._dynamo as dynamo

import time
import gc

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from data import get_cifar, setup_distribute, cleanup_distribute, DataAugmentation, Resize, Normalize, CenterCrop
from Helper import save_individual_image, plot_logs
from VGG import VGG
from TrainWrappers import BaseCNNTrainer

dynamo.config.capture_scalar_outputs = True
dynamo.config.cache_size_limit = 256

torch.backends.cudnn.benchmark = True

def train(rank, world_size):

    try:

        torch.cuda.set_device(rank)

        setup_distribute(rank, world_size)
    
        dataAug = torch.jit.script(DataAugmentation().to('cuda'))
        resize = torch.jit.script(Resize().to('cuda'))
        normalize = torch.jit.script(Normalize().to('cuda'))
        center_crop = torch.jit.script(CenterCrop().to('cuda'))

        torch._print("Got Data")

        model = VGG(19)

        model = DDP(model.to('cuda'), 
                    device_ids = [rank],
                    output_device = rank, 
                    gradient_as_bucket_view = True,
                    find_unused_parameters = True)

        model = torch.compile(model)
        
        torch._print("Got Cuda Model")
        
        T = BaseCNNTrainer(model, rank)

        del model

        T.build(optimizer = torch.optim.SGD(T.m.parameters(), 0.01, momentum = 0.9, weight_decay = 1e-4),
                loss = nn.CrossEntropyLoss(reduction = "summ").to('cuda'),
                collective_transforms = [resize], train_transforms = [dataAug],
                eval_transforms = [center_crop], final_collective_transforms = [normalize],
                scale_loss = True, gradient_clipnorm = 2.0)

        print(all([param.device == torch.cuda.current_device] for name, param in T.m.named_buffers()))
        print(all([param.device == torch.cuda.current_device] for name, param in T.m.named_parameters()))

        torch.cuda.empty_cache()
        gc.collect()

        dt, dv = get_cifar(rank, world_size) 
        
        logs = T.fit(dt, dv, 20, 391, "TMP")

        T.evaluate(dt)

        if (rank == 0): print(T.metric_results())

        T.evaluate(dv)

        if (rank == 0):
            print(T.metric_results())
            plot_logs(logs, 30, "TMP", 391) 
        

    finally:
        
        cleanup_distribute()

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()