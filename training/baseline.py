import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from data.cifar10 import *

from utils.serialization_utils import logs_to_pickle, save_tensor
from utils.training_utils import plot_logs

from training.VGG import VGG_CNN
from models.VGG import VGG

import json
import pickle
import gc

def main(rank, world_size, name: str, args: list, lock, shared_list, **kwargs):

    EPOCHS = 160
    CARDINALITY = 391

    dataAug = torch.jit.script(DataAugmentation().to('cuda'))
    resize = torch.jit.script(Resize().to('cuda'))
    normalize = torch.jit.script(Normalize().to('cuda'))
    center_crop = torch.jit.script(CenterCrop().to('cuda'))

    model = VGG(depth = 19, rank = rank)

    sp = 0.8**int(args[0])
    if rank == 0: print(sp)
    
    ticket = None
    pruned = False

    while not pruned:
        model.reset_ticket()
        model.prune_random(sp, distributed = True)

        not_collapse = True
        for layer in model.lottery_layers:
            not_collapse &= torch.any(layer.weight_mask).item()

        if not_collapse: 
            pruned = True
            ticket = model.export_ticket_cpu()

        if rank == 0:
            print(f"Did Not Collapse? {not_collapse}.")

    model.reset_ticket()
    model.set_ticket(ticket, zero_out = False)
    name += "_" + str(args[0])

    model = DDP(model.to('cuda'), 
                device_ids = [rank],
                output_device = rank, 
                gradient_as_bucket_view = True)
 
    #model = torch.compile(model)
    """
    T = VGG_DGTS(model = model, rank = rank, world_size = world_size, 
                 lock = lock, dynamic_list = shared_list)
    
    T.build(sparsity_rate = 0.8, type_of_exp = type_of_exp, experiment_args = args[1:],
            optimizer = torch.optim.SGD(T.m.parameters(), 0.1, momentum = 0.9, weight_decay = 1e-3),
            loss = torch.nn.CrossEntropyLoss(reduction = "sum").to('cuda'),
            collective_transforms = (resize, normalize), train_transforms = (dataAug,),
            eval_transforms = (center_crop,), final_collective_transforms = tuple(),
            scale_loss = True, gradient_clipnorm = 2.0)
    """
    #T.summary(32)

    T = VGG_CNN(model = model, rank = rank, world_size = world_size)

    T.build(torch.optim.SGD(T.m.parameters(), 0.1, momentum = 0.9, weight_decay = 1e-3),
            loss = torch.nn.CrossEntropyLoss(reduction = "sum").to('cuda'),
            collective_transforms = (resize, normalize), train_transforms = (dataAug,),
            eval_transforms = (center_crop,), final_collective_transforms = tuple(),
            scale_loss = True, gradient_clipnorm = 2.0)

    #T.prune_model(ticket)

    del model

    torch.cuda.empty_cache()
    gc.collect()

    dt, dv = get_loaders(rank, world_size, batch_size = 128) 

    #T.load_ckpt("IVGTS_LATE1_200.0,128.0,128.0,52784.0,0.0116", "best")

    logs = T.fit(dt, dv, EPOCHS, CARDINALITY, name, save = False, verbose = False, validate = False)

    #T.mm.prune_by_mg(sp, iteration = 1)
    #T.prune_model(T.mm.export_ticket_cpu())

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
        
        with open(f"./logs/RESULTS/{name}.json", "w") as f:
            json.dump(train_res, f, indent = 6)
    
        logs_to_pickle(logs, name)
        """
        logs_to_pickle(T.fitnesses, name, suffix = "fitnesses")

        #logs_to_pickle(T.prunes, name, suffix = "prunes")

        with open(f"./logs/PICKLES/{name}_prunes.json", "w") as f:
            json.dump(T.prunes, f, indent = 6)"""

        """plot_logs(logs, EPOCHS, name, steps = CARDINALITY, start = 0)"""

    torch.distributed.barrier(device_ids = [rank])