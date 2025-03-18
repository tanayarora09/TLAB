import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from data.cifar10 import *

from utils.serialization_utils import logs_to_pickle, save_tensor
from utils.training_utils import plot_logs

from training.VGG import VGG_POC
from models.VGG import VGG

import json
import pickle
import gc

def main(rank, world_size, name: str, **kwargs):

    EPOCHS = 160
    CARDINALITY = 98

    dataAug = torch.jit.script(DataAugmentation().to('cuda'))
    resize = torch.jit.script(Resize().to('cuda'))
    normalize = torch.jit.script(Normalize().to('cuda'))
    center_crop = torch.jit.script(CenterCrop().to('cuda'))

    model = VGG(depth = 19, rank = rank, world_size = world_size, custom_init = True)


    model = DDP(model.to('cuda'), 
                device_ids = [rank],
                output_device = rank, 
                gradient_as_bucket_view = True)
 
    #model = torch.compile(model)
    
    T = VGG_POC(model, rank, world_size)

    #T.summary(32)

    del model

    torch.cuda.empty_cache()
    gc.collect()
    
    T.build(optimizer = torch.optim.SGD, optimizer_kwargs = {'lr': 0.1, 'momentum': 0.9, 'weight_decay': 1e-3},
        loss = torch.nn.CrossEntropyLoss(reduction = "sum").to('cuda'),
        collective_transforms = (resize, normalize), train_transforms = (dataAug,),
        eval_transforms = (center_crop,), final_collective_transforms = tuple(),
        scale_loss = True, gradient_clipnorm = 2.0)

    dt, dv = get_loaders(rank, world_size, batch_size = 512) 
    

    logs, sparsities_d = T.TicketIMP(dt, dv, EPOCHS, CARDINALITY, name, 0.8, 20, rewind_iter = 1274)

    if (rank == 0):

        with open(f"./tmp/sparsities_{name}.json", "w", encoding = "utf-8") as f:
            json.dump(sparsities_d, f, ensure_ascii = False, indent = 4)
            
        logs_to_pickle(logs, name)        
    
        for i in range(len(logs)):
            plot_logs(logs[i], EPOCHS, name + f"_{(sparsities_d[i] * 100):.2f}", 
                      CARDINALITY, start = (0 if i == 0 else 13)) 