import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from data.cifar10 import *

from utils.serialization_utils import logs_to_pickle, save_tensor
from utils.training_utils import plot_logs

from training.VGG import VGG_POC_FULL
from models.VGG import VGG

import json
import pickle
import gc

import h5py
import os

def build_tickets_dict(name:str, last_name: str, model: VGG, rank: int):

    out = dict()
    """
    with open(f"./tmp/sparsities_{last_name}.json", "r", encoding = "utf-8") as f:
        sparsities_d = json.load(f)
        print(sparsities_d)"""
    
    sparsities_d = [0.8**sp for sp in range(31)]

    if rank == 0: open(f"logs/TICKETS/{name}TD.h5",'x').close()

    for i in range(1, len(sparsities_d)):
        dist.barrier(device_ids = [rank])
        with torch.random.fork_rng(devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"], enabled = True):
            model.prune_random(sparsities_d[i], distributed = True, root = 2)
            random_ticket = model.export_ticket_cpu()
            model.reset_ticket()
            dist.barrier(device_ids = [rank])
            imp_ticket = model.load_ticket(last_name, 2, f"{(sparsities_d[i]*100):.2f}").cpu()
            if i > 1: model.set_ticket(model.load_ticket(last_name, 2, f"{(sparsities_d[i-1]*100):.2f}"))
            model.prune_random(sparsities_d[i]/sparsities_d[i-1], distributed = True, root = 2)
            random_d_ticket = model.export_ticket_cpu()
            dist.barrier(device_ids = [rank])
            model.set_ticket(imp_ticket)
            ls = model.export_layerwise_sparsities()
            model.prune_random_given_layerwise(ls, distributed = True, root = 2)
            layerwise_ticket = model.export_ticket_cpu()
        
        hmp = lambda t: (1.0 - ((torch.logical_and(imp_ticket, t)).sum()/imp_ticket.sum()).item())

        #hmps = {'Random': hmp(random_ticket), 'RandomGen': hmp(random_d_ticket), 'Layerwise': hmp(layerwise_ticket)}

        #print(f"[rank {rank}] At {model.sparsity} sparsity, Hamming Percentages: ", hmps)
        out[sparsities_d[i]] =  (imp_ticket, layerwise_ticket, random_d_ticket, random_ticket)

        model.export_ticket(name+"TD", entry_name = f"imp_{i:02d}", given = imp_ticket)
        model.export_ticket(name+"TD", entry_name = f"layer_random_{i:02d}", given = layerwise_ticket)
        model.export_ticket(name+"TD", entry_name = f"close_random_{i:02d}", given = random_d_ticket)
        model.export_ticket(name+"TD", entry_name = f"true_random_{i:02d}", given = random_ticket)

        model.reset_ticket()

    #if rank == 0: os.remove(f"./tmp/sparsities_{last_name}.json")

    #with h5py.File(f"logs/TICKETS/{name}TD.h5", 'r') as f: print(f.keys())

    return out

def main(rank, world_size, name: str, amts: list, **kwargs):

    last_name = ["ProofOfConcept_0_f", "ProofOfConcept_1_f", "ProofOfConceptMG_0_f"][int(name[-1])]

    print(last_name)

    EPOCHS = amts[0]
    CARDINALITY = 98
    PRUNE_ITERS = 0#amts[1] - 1
    REWIND_ITER = amts[2] * CARDINALITY

    dataAug = torch.jit.script(DataAugmentation().to('cuda'))
    resize = torch.jit.script(Resize().to('cuda'))
    normalize = torch.jit.script(Normalize().to('cuda'))
    center_crop = torch.jit.script(CenterCrop().to('cuda'))

    model = VGG(depth = 19, rank = rank, world_size = world_size, custom_init = True)

    ticket_dict = build_tickets_dict(name, last_name, model, rank)

    #print(torch.rand(1))

    model = DDP(model.to('cuda'), 
                device_ids = [rank],
                output_device = rank, 
                gradient_as_bucket_view = True)
    
    T = VGG_POC_FULL(model, rank, world_size)

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

    with open(f"./logs/ACTIVATIONS/activation_log_{name}_{rank}.pickle", "wb") as f:
        pickle.dump(T.activation_log, f, protocol=pickle.HIGHEST_PROTOCOL)
   
    if (rank == 0):    
        
        logs_to_pickle(logs, name)

