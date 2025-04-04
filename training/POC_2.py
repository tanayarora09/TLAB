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
        #print(sparsities_d)
    
    for i in range(1, len(sparsities_d)):
        dist.barrier(device_ids = [rank])
        ticket = model.load_ticket(last_name, 2, f"{(sparsities_d[i]*100):.2f}")
        with torch.random.fork_rng(devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"], enabled = True):
            dist.barrier(device_ids = [rank])
            model.prune_random(sparsities_d[i], distributed = True, root = 2)
        #print(f"[rank {rank}] At {model.sparsity} sparsity, {(ticket.logical_xor(model.get_buffer("MASK"))).sum()} different pruned.")
        out[i-1] = (ticket, model.export_ticket_cpu())
        model.reset_ticket()

    if rank == 0: os.remove(f"./tmp/sparsities_{last_name}.json")

    return out

def main(rank, world_size, name: str, **kwargs):

    EPOCHS = 160
    CARDINALITY = 391

    last_name = name[:-1] + "1"

    dataAug = torch.jit.script(DataAugmentation().to('cuda'))
    resize = torch.jit.script(Resize().to('cuda'))
    normalize = torch.jit.script(Normalize().to('cuda'))
    center_crop = torch.jit.script(CenterCrop().to('cuda'))

    model = VGG(depth = 19, rank = rank)

    ticket_dict = build_tickets_dict(last_name, model, rank)

    model = DDP(model.to('cuda'), 
                device_ids = [rank],
                output_device = rank, 
                gradient_as_bucket_view = True)

    #model = torch.compile(model)
    
    
    T = VGG_POC(model, rank)

    T.build(optimizer = torch.optim.SGD, optimizer_kwargs = {'lr': 0.1, 'momentum': 0.9, 'weight_decay' : 1e-3},
            loss = torch.nn.CrossEntropyLoss(reduction = "sum").to('cuda'),
            collective_transforms = (resize, normalize), train_transforms = (dataAug,),
            eval_transforms = (center_crop,), final_collective_transforms = tuple(),#[normalize],
            scale_loss = True, gradient_clipnorm = 2.0 , tickets_dict = ticket_dict)

    #T.summary(32)

    torch.cuda.empty_cache()
    gc.collect()

    dt, dv = get_loaders(rank, world_size, batch_size = 128) 
    
    logs, sparsities_d = T.TicketIMP(dt, dv, EPOCHS, CARDINALITY, name, 0.8, 19, rewind_iter = 250)

    T.evaluate(dt)

    if (rank == 0): 
        print("Train Results: ", T.metric_results())

    T.evaluate(dv)

    if (rank == 0):

        print("Validation Results: ", T.metric_results())
        print("Sparsity: ", T.mm.sparsity)

        with open(f"./logs/ACTIVATIONS/activation_log_{name[:-1]}.json", "w", encoding = "utf-8") as f:
            json.dump(T.activation_log, f, ensure_ascii = False, indent = 4)
        
        logs_to_pickle(logs, name)

        #for i in range(len(logs)):
        #    plot_logs(logs[i], EPOCHS, name + f"_{((sparsities_d[i])):.2f}", CARDINALITY) 