import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from data.cifar10 import *

from utils.serialization_utils import logs_to_pickle
from utils.training_utils import plot_logs

from training.VGG import VGG_POC_ACT, VGG_POC_GRAD
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
    
    torch._print("Got Cuda Model")
    
    T = VGG_POC_ACT(model, rank)

    del model

    T.build(optimizer = torch.optim.SGD(T.m.parameters(), 0.1, momentum = 0.9, weight_decay = 1e-3),
            loss = torch.nn.CrossEntropyLoss(reduction = "sum").to('cuda'),
            collective_transforms = (resize, normalize), train_transforms = (dataAug,),
            eval_transforms = (center_crop,), final_collective_transforms = tuple(),#[normalize],
            scale_loss = True, gradient_clipnorm = 2.0)

    print(all([param.device == torch.cuda.current_device] for name, param in T.m.named_buffers()))
    print(all([param.device == torch.cuda.current_device] for name, param in T.m.named_parameters()))

    T.summary(32)

    torch.cuda.empty_cache()
    gc.collect()

    dt, dv = get_loaders(rank, world_size, iterate = True) 
    
    logs = T.TicketIMP(dt, dv, 160, 391, name, 0.8, 19, type = "rewind")

    T.evaluate(dt)

    if (rank == 0): 
        print(T.metric_results())

    T.evaluate(dv)

    if (rank == 0):
        print(T.metric_results())
        print(T.m.module.sparsity)

    elif (rank == 1):
        import json
        with open("./tmp/last_capture.json", "w", encoding = "utf-8") as f: 
            json.dump(T.act_captures, f, ensure_ascii = False, indent = 4)
        logs_to_pickle(T.act_captures, name + "_capture")

    T.load_ckpt(name + f"_IMP_{logs[-1][1]:.1f}", "best")
    T.load_ticket(name + f"_IMP_{logs[-1][1]:.1f}")

    T.evaluate(dt)

    if (rank == 0): 
        print(T.metric_results())
        print(T.m.module.sparsity)

    T.evaluate(dv)

    if (rank == 0):
        print(T.metric_results())
        for i in range(len(logs)):
            plot_logs(logs[i][0], 160, name + f"_IMP_{((logs[i][1])):.1f}", 391) 
        logs_to_pickle(logs, name)