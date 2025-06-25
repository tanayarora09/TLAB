import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from data.cifar10 import *

from utils.serialization_utils import logs_to_pickle, save_tensor
from utils.training_utils import plot_logs

from training.VGG import VGG_CNN, VGG_IMP
from models.VGG import VGG
from models.base import BaseModel

import json
import pickle
import gc
import h5py

def main(rank, world_size, name: str, sp_exp: list, **kwargs):

    #sp = 0.8 ** sp_exp

    EPOCHS = 160
    CARDINALITY = 98
    PRUNE_ITERS = 27
    
    old_name = name

    if rank == 0: h5py.File(f"./logs/TICKETS/{old_name}.h5", "w").close()

    dataAug = torch.jit.script(DataAugmentation().to('cuda'))
    resize = torch.jit.script(Resize().to('cuda'))
    normalize = torch.jit.script(Normalize().to('cuda'))
    center_crop = torch.jit.script(CenterCrop().to('cuda'))

    model = VGG(depth = 16, rank = rank, world_size = world_size, custom_init = True).cuda()
    
    T = VGG_IMP(model, rank, world_size)

    T.build(optimizer = torch.optim.SGD, optimizer_kwargs = {'lr': 0.1, 'momentum': 0.9, 'weight_decay' : 1e-3},
            loss = torch.nn.CrossEntropyLoss(reduction = "sum").to('cuda'),
            collective_transforms = (resize, normalize), train_transforms = (dataAug,),
            eval_transforms = (center_crop,), final_collective_transforms = tuple(),
            scale_loss = True, gradient_clipnorm = 2.0)

    del model

    torch.cuda.empty_cache()
    gc.collect()

    dt, dv = get_loaders(rank, world_size, batch_size = 512) 

    (logs, results), sparsities_d = T.TicketIMP(dt, dv, EPOCHS, CARDINALITY, old_name, 0.8, PRUNE_ITERS, rewind_iter = 13*CARDINALITY, validate = False)

    if rank == 0:
        logs_to_pickle(logs)
        for spe in list(range(PRUNE_ITERS)):
            with open(f"./logs/RESULTS/{old_name}_{spe}.json", "w") as f:
                json.dump(results[spe], f, indent = 6)
                    
    torch.distributed.barrier(device_ids = [rank])
