import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from data.cifar10 import *

from utils.serialization_utils import logs_to_pickle, save_tensor
from utils.training_utils import plot_logs

from training.VGG import VGG_IMP
from models.VGG import VGG

import json
import pickle
import gc

def main(rank, world_size, name: str, **kwargs):

    EPOCHS = 160
    CARDINALITY = 98
    PRUNE_ITERS = 26
    REWIND_ITER = 13 * CARDINALITY

    dataAug = torch.jit.script(DataAugmentation().to('cuda'))
    resize = torch.jit.script(Resize().to('cuda'))
    normalize = torch.jit.script(Normalize().to('cuda'))
    center_crop = torch.jit.script(CenterCrop().to('cuda'))

    model = VGG(depth = 19, rank = rank, world_size = world_size, custom_init = True)


    model = DDP(model.to('cuda'), 
                device_ids = [rank],
                output_device = rank, 
                gradient_as_bucket_view = True)
    
    T = VGG_IMP(model, rank, world_size)

    del model

    torch.cuda.empty_cache()
    gc.collect()
    
    T.build(optimizer = torch.optim.SGD, optimizer_kwargs = {'lr': 0.1, 'momentum': 0.9, 'weight_decay': 1e-3},
        loss = torch.nn.CrossEntropyLoss(reduction = "sum").to('cuda'),
        collective_transforms = (resize, normalize), train_transforms = (dataAug,),
        eval_transforms = (center_crop,), final_collective_transforms = tuple(),
        scale_loss = True, gradient_clipnorm = 2.0)

    dt, dv = get_loaders(rank, world_size, batch_size = 512) 
    

    logs, sparsities_d = T.TicketIMP(dt, dv, EPOCHS, CARDINALITY, name, 0.8, PRUNE_ITERS, rewind_iter = REWIND_ITER)

    if (rank == 0):
   
        logs_to_pickle(logs, name)
    
        for i in range(len(logs)):
            plot_logs(logs[i], EPOCHS, name + f"_{(sparsities_d[i] * 100):.3e}", 
                      CARDINALITY, start = (0 if i == 0 else 13)) 
            
        with open(f"./logs/PICKLES/{name}_best_fitnesses.json", "w", encoding = "utf-8") as f:
            json.dump({}, f, ensure_ascii = False, indent = 4)

        T.evaluate(dt)

        if (rank == 0): 
            train_res = T.metric_results()
            print("Train Results: ", train_res)

        T.evaluate(dv)
        
        if (rank == 0):
            val_res =  T.metric_results()
            print("Validation Results: ", val_res)
            print("Sparsity: ", T.mm.sparsity)
            train_res.update({('val_' + k): v for k, v in val_res.items()})
        
            with open(f"./logs/RESULTS/{old_name}_{spe}.json", "w") as f:
                json.dump(train_res, f, indent = 6)
            
            T.mm.export_ticket(old_name, entry_name = f"{sp * 100:.3e}")
