import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from data.cifar10 import *

from utils.serialization_utils import logs_to_pickle, save_tensor
from utils.training_utils import plot_logs

from training.VGG import VGG_CNN
from search.salient import GraSP_Pruner
from search.genetic import GradientNormSearch
from models.VGG import VGG, BaseModel

import json
import pickle
import gc

import h5py

EPOCHS = 160
CARDINALITY = 98

def ddp_network(rank, world_size, depth = 19):

    model = VGG(depth = depth, rank = rank, world_size = world_size, custom_init = True)
    model = DDP(model.to('cuda'), 
                device_ids = [rank],
                output_device = rank, 
                gradient_as_bucket_view = True)
    
    return model

def get_grasp_tickets(rank, world_size, model, transforms, spr):

    state = model.state_dict()
    
    if rank == 0:
        pruner = GraSP_Pruner(0, 1, model.module)
        pruner.build(spr, transforms, input = None)
        ticket, improved = pruner.grad_mask(improved = "2")
        pruner.finish()
        print(f"GraSP Ticket Sparsity: {100 * (ticket.sum()/ticket.numel()):.2f}")
        print(f"GraSP Improved Ticket Sparsity: {100 * (improved.sum()/improved.numel()):.2f}")


    else:
        ticket = torch.zeros(model.module.num_prunable, dtype = torch.bool, device = 'cuda')
        improved = torch.zeros(model.module.num_prunable, dtype = torch.bool, device = 'cuda')

    torch.distributed.barrier(device_ids = [rank])
    torch.distributed.broadcast(ticket, src = 0)
    torch.distributed.barrier(device_ids = [rank])
    torch.distributed.broadcast(improved, src = 0)

    return state, (ticket, improved)

    pass

def _make_trainer(rank, world_size, state, ticket):

    model = ddp_network(rank, world_size, 16)
    model.load_state_dict(state)
    model.module.set_ticket(ticket)
    
    if (rank == 0):
        print(model.module.sparsity, "\n")

    return VGG_CNN(model, rank, world_size)

def _print_collect_return_fitness(rank, name, fitness):
    print(f"[rank {rank}] {name.capitalize()} Fitness: ", fitness)
    fitness = torch.as_tensor(fitness, dtype = torch.float64, device = 'cuda')
    dist.all_reduce(fitness, op = dist.ReduceOp.AVG)
    dist.barrier(device_ids = [rank])
    return fitness.item()

def run_check_and_export(rank, world_size, state, spr, grasp_ticket, 
                         grasp_improved_ticket, imp_ticket, transforms):

    model = ddp_network(rank, world_size, 16)
    model.load_state_dict(state)

    partial, _ = get_loaders(rank, world_size, batch_size = 512)

    del _

    gc.collect()

    search = GradientNormSearch(rank, world_size, 
                                model.module, partial, 
                                spr, transforms)

    output = dict()

    search.calculate_fitness_given(imp_ticket)

    output["imp"] = _print_collect_return_fitness(rank, "imp", search.calculate_fitness_given(imp_ticket))

    output["grasp"] = _print_collect_return_fitness(rank, "grasp", search.calculate_fitness_given(grasp_ticket))

    output["grasp_improved"] = _print_collect_return_fitness(rank, "grasp_improved", search.calculate_fitness_given(grasp_improved_ticket))

    output["magnitude"] = _print_collect_return_fitness(rank, "magnitude", search.calculate_fitness_magnitude())

    output["random"] = _print_collect_return_fitness(rank, "random", search.calculate_fitness_random())
    
    return output

def main(rank, world_size, name: str, sp_exp: list, **kwargs):

    imp_name = f"imp_vgg16_{name[-1]}"

    dataAug = torch.jit.script(DataAugmentation().to('cuda'))
    resize = torch.jit.script(Resize().to('cuda'))
    normalize = torch.jit.script(Normalize().to('cuda'))
    center_crop = torch.jit.script(CenterCrop().to('cuda'))

    logs = dict()

    for spe in reversed(sp_exp):

        spr = 0.8**spe

        model = ddp_network(rank, world_size, depth = 16)
        ckpt = torch.load(f"./logs/WEIGHTS/init_{imp_name}_100.00.pt", map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}, 
                          weights_only = True)['model']
        model.load_state_dict(ckpt)

        state, (grasp_ticket, grasp_improved_ticket) = get_grasp_tickets(rank, world_size, model, (resize, normalize, center_crop,), spr)

        imp_ticket = model.module.load_ticket(imp_name, root = 0, entry_name = f"{spr*100:.2f}")

        del model

        torch.cuda.empty_cache()
        gc.collect()

        log = run_check_and_export(rank, world_size, state, spr, grasp_ticket, 
                                   grasp_improved_ticket, imp_ticket,  
                                   (resize, normalize, center_crop,))

        if rank == 0: print(f"SPARSITY: {spr * 100 :.2f} || {log.items()}")

        logs[spe] = log

    with open(f"./logs/PICKLES/{name}_comparison.json", "w") as f:
        json.dump(logs, f, indent = 6)
