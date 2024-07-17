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
from TrainWrappers import BaseCNNWrapper

dynamo.config.capture_scalar_outputs = True
dynamo.config.cache_size_limit = 256

torch.backends.cudnn.benchmark = True

def train(rank, world_size):

    try:

        torch.cuda.set_device(rank)

        setup_distribute(rank, world_size)
    
        dataAug = torch.jit.script(DataAugmentation().to('cuda'))
        resize = torch.jit.script(Resize().to('cuda')) #torch.jit.script(ResizeAndNormalize().to('cuda'))
        normalize = torch.jit.script(Normalize().to('cuda'))
        center_crop = torch.jit.script(CenterCrop().to('cuda'))

        torch._print("Got Data")

        model = VGG19()#.to(memory_format = torch.channels_last)

        #model  = nn.SyncBatchNorm.convert_sync_batchnorm(model) # only necessary if tracking running means (not doing well for eval)

        model = DDP(model.to('cuda'), 
                    device_ids = [rank],
                    output_device = rank, 
                    gradient_as_bucket_view = True,
                    find_unused_parameters = True)

        model = torch.compile(model)
        
        torch._print("Got Cuda Model")
        
        T = TicketCNN(model, rank)

        del model

        T.build(torch.optim.SGD(T.m.module.parameters(), 0.01, momentum = 0.9), #weight_decay = 0.0005),
                loss = nn.CrossEntropyLoss().to('cuda'), 
                data_augmentation_transform = dataAug, resize_transform = resize,
                normalize_transform = normalize, evaluate_transform = center_crop,
                weight_decay = 0.00125, scaler = True, clipnorm = 2.0)

        print(all([param.device == torch.cuda.current_device] for name, param in T.m.named_buffers()))
        print(all([param.device == torch.cuda.current_device] for name, param in T.m.named_parameters()))

        torch.cuda.empty_cache()
        gc.collect()

        dt, dv = get_cifar(rank, world_size) 
        """
        for step, (x, y) in enumerate(dt):
            x, y = x.to('cuda'), y.to('cuda')
            x = dataAug(resize(x))
            normalize(x)
            for i in range(len(x)):
                print(i)
                save_individual_image(x[i], f"./data_viewing/{rank}_{step}_{i}")

            if step > 0:
                break"""

        #T.evaluate(dt)
        
        #if (rank == 0): print(T.get_eval_results())

        #T.evaluate(dv)

        #if (rank == 0): print(T.get_eval_results())
        
        logs = T.train_one(dt, dv, 35, 391, "TMP")

        dt.sampler.set_epoch(0)
        dv.sampler.set_epoch(0)

        T.evaluate(dt)

        if (rank == 0): print(T.get_eval_results())

        T.evaluate(dv)

        if (rank == 0):
            print(T.get_eval_results())
            plot_logs(logs, 35, "TMP", 391) 
        

    finally:
        
        cleanup_distribute()

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()