import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from data.cifar10 import *

from utils.serialization_utils import logs_to_pickle
from utils.training_utils import plot_logs

from training.VGG import VGG_POC
from models.VGG import VGG

import gc

def main(rank, world_size, name: str):

    dataAug = torch.jit.script(DataAugmentation().to('cuda'))
    resize = torch.jit.script(Resize().to('cuda'))
    normalize = torch.jit.script(Normalize().to('cuda'))
    center_crop = torch.jit.script(CenterCrop().to('cuda'))

    model = VGG(19)

    model = DDP(model.to('cuda'), 
                device_ids = [rank],
                output_device = rank, 
                gradient_as_bucket_view = True)

    model = torch.compile(model)
    
    T = VGG_POC(model, rank)

    del model

    T.build(optimizer = torch.optim.SGD(T.m.parameters(), 0.1, momentum = 0.9, weight_decay = 5e-4),
            loss = torch.nn.CrossEntropyLoss(reduction = "sum").to('cuda'),
            collective_transforms = (resize, ), train_transforms = (dataAug,),
            eval_transforms = (center_crop,), final_collective_transforms = (normalize,),#[normalize],
            scale_loss = True, gradient_clipnorm = 2.0)
    T.summary(32)

    torch.cuda.empty_cache()
    gc.collect()

    dt, dv = get_loaders(rank, world_size, iterate = True) 
    
    logs, sparsities = T.TicketIMP(dt, dv, 12, 391, name, 0.8, 6)#T.fit(dt, dv, 3, 391, name)

    T.print(T.grad_captures, 'white', rank = 1)

    T.evaluate(dt)

    if (rank == 0): print(T.metric_results())

    T.evaluate(dv)

    if (rank == 0):
        print(T.metric_results())
        print(T.m.module.sparsity)

    if (rank == 1):
        import json
        with open("./tmp/last_grads.json", "w") as f: 
            json.dump(T.grad_captures, f, ensure_ascii = False, indent = 4)

    del T

    model = VGG(19)

    model = DDP(model.to('cuda'), 
                device_ids = [rank],
                output_device = rank, 
                gradient_as_bucket_view = True)

    model = torch.compile(model)
    
    torch._print("Got Cuda Model")
    
    T = VGG_POC(model, rank)

    del model

    T.build(optimizer = torch.optim.SGD(T.m.parameters(), 0.1, momentum = 0.9, weight_decay = 5e-4),
            loss = torch.nn.CrossEntropyLoss(reduction = "sum").to('cuda'),
            collective_transforms = (resize, ), train_transforms = (dataAug,),
            eval_transforms = (center_crop,), final_collective_transforms = (normalize,),#[normalize],
            scale_loss = True, gradient_clipnorm = 2.0)

    T.load_ckpt(name + f"_IMP_{sparsities[-1]:.1f}", "best")
    T.load_ticket(name + f"_IMP_{sparsities[-1]:.1f}")

    T.evaluate(dt)

    if (rank == 0): 
        print(T.metric_results())
        print(T.m.module.sparsity)

    T.evaluate(dv)

    if (rank == 0):
        print(T.metric_results())
        for i in range(len(logs)):
            plot_logs(logs[i], 12, name + f"_IMP_{((sparsities[i])):.1f}", 391) 
        logs_to_pickle(logs, name)

