import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data.cifar10 import *

from utils.serialization_utils import logs_to_pickle, save_tensor
from utils.training_utils import plot_logs

from training.VGG import VGG_POC
from models.VGG import VGG

import json
import pickle
import gc

import os

def build_tickets_dict(last_name: str, model: VGG, rank: int):

    out = dict()

    with open(f"./tmp/sparsities_{last_name}.json", "r", encoding = "utf-8") as f:
        sparsities_d = json.load(f)
        print(sparsities_d)
    
    """sparsities_d = [1.0, 0.8, 0.64, 0.5120000076293946, 0.40959999084472654, 0.32768001556396487, 0.2621440315246582, 0.20971523284912108, 0.16777217864990235, 
                    0.13421775817871093, 0.10737419128417969, 0.08589936256408691, 0.06871947765350342, 0.05497560501098633, 0.04398048400878906, 
                    0.035184388160705564, 0.0281475305557251, 0.022518043518066407, 0.01801444411277771, 0.014411544799804688, 0.011529216766357422]"""

    print(sparsities_d)

    for i in range(1, len(sparsities_d)):
        dist.barrier(device_ids = [rank])
        with torch.random.fork_rng(devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"], enabled = True):
            dist.barrier(device_ids = [rank])
            if i > 1: model.set_ticket(model.load_ticket(last_name, 2, f"{(sparsities_d[i-1]*100):.2f}"))
            model.prune_random(sparsities_d[i]/sparsities_d[i-1], distributed = True, root = 2)
            random_ticket = model.export_ticket_cpu()
            imp_ticket = model.load_ticket(last_name, 2, f"{(sparsities_d[i]*100):.2f}").cpu()
        print(f"[rank {rank}] At {model.sparsity} sparsity, {(imp_ticket.logical_xor(random_ticket)).sum()} different pruned.")
        out[i-1] = (imp_ticket, random_ticket)
        model.reset_ticket()

    if rank == 0: os.remove(f"./tmp/sparsities_{last_name}.json")

    return out

def main(rank, world_size, name: str, amts: list, **kwargs):

    last_name = name[:-1] + "f"

    EPOCHS = amts[0]
    CARDINALITY = 98
    PRUNE_ITERS = amts[1] - 1
    REWIND_ITER = amts[2] * CARDINALITY

    dataAug = torch.jit.script(DataAugmentation().to('cuda'))
    resize = torch.jit.script(Resize().to('cuda'))
    normalize = torch.jit.script(Normalize().to('cuda'))
    center_crop = torch.jit.script(CenterCrop().to('cuda'))

    model = VGG(depth = 19, rank = rank, world_size = world_size, custom_init = True)

    ticket_dict = build_tickets_dict(last_name, model, rank)

    #print(torch.rand(1))

    model = DDP(model.to('cuda'), 
                device_ids = [rank],
                output_device = rank, 
                gradient_as_bucket_view = True)
    
    T = VGG_POC(model, rank, world_size)

    del model 

    torch.cuda.empty_cache()
    gc.collect()

    T.build(optimizer = torch.optim.SGD, optimizer_kwargs = {'lr': 0.1, 'momentum': 0.9, 'weight_decay': 1e-3},
            loss = torch.nn.CrossEntropyLoss(reduction = "sum").to('cuda'),
            collective_transforms = (resize, normalize), train_transforms = (dataAug,),
            eval_transforms = (center_crop,), final_collective_transforms = tuple(),
            scale_loss = True, gradient_clipnorm = 2.0, tickets_dict = ticket_dict)


    dt, dv = get_loaders(rank, world_size, batch_size = 512) 
    
    logs, sparsities_d = T.TicketIMP(dt, dv, EPOCHS, CARDINALITY, name, 0.8, PRUNE_ITERS, rewind_iter = REWIND_ITER)

    with open(f"./logs/ACTIVATIONS/activation_log_{name[:-1]}_{rank}.json", "wb") as f:
        pickle.dump(T.activation_log, f, protocol=pickle.HIGHEST_PROTOCOL)
   
    if (rank == 0):    
        
        logs_to_pickle(logs, name)

