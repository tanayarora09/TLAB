import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from data.cifar10 import *

from utils.serialization_utils import logs_to_pickle, save_tensor
from utils.training_utils import plot_logs
from utils.search_utils import plot_logs_concrete

from training.VGG import VGG_CNN
from training.ResNet import ResNet_CNN
from search.salient import *
from search.concrete import *
from search.genetic import *
from models.VGG import VGG
from models.base import BaseModel
from models.ResNet import ResNet


import json
import pickle
import gc
from collections import defaultdict

import h5py

EPOCHS = 160
CARDINALITY = 98

def ddp_network(rank, world_size, is_vgg, depth = 16):

    if not is_vgg: depth = 20
    
    if is_vgg:
        model = VGG(depth = depth, rank = rank, world_size = world_size, custom_init = True) 
    else: 
        model =  ResNet(depth = depth, rank = rank, world_size = world_size, custom_init = True)
    
    model = model.cuda()

    model = DDP(model, 
        device_ids = [rank],
        output_device = rank, 
        gradient_as_bucket_view = True)

    return model

def _get_train(rank, world_size):

    dt, _ = get_loaders(rank, world_size, batch_size = 512, validation = False)
    return dt

def get_salient_ticket(rank, world_size, state, is_vgg, transforms, spr):

    model = ddp_network(rank, world_size, is_vgg, depth = 16)
    model.load_state_dict(state)

    if rank == 0:
        pruner = GraSP_Pruner(0, 1, model, )
        pruner.build(spr, transforms, input = None)
        ticket, improved = pruner.grad_mask(improved = "2")
        pruner.finish()


    else:
        ticket = torch.zeros(model.module.num_prunable, dtype = torch.bool, device = 'cuda')
        improved = torch.zeros(model.module.num_prunable, dtype = torch.bool, device = 'cuda')

    torch.distributed.barrier(device_ids = [rank])
    torch.distributed.broadcast(ticket, src = 0)
    torch.distributed.barrier(device_ids = [rank])
    torch.distributed.broadcast(improved, src = 0)

    return state, (ticket, improved)


def get_concrete_ticket_and_imp(rank, world_size, state, is_vgg, dt, transforms, spr, name, imp_name):

    model = ddp_network(rank, world_size, is_vgg, depth = 16)
    model.load_state_dict(state)

    imp_ticket = model.module.load_ticket(imp_name, root = 0, entry_name = f"{spr*100:.2f}")

    search = GraSPConcrete(rank, world_size, model, )# capture_layers = captures, fake_capture_layers = fcaptures)
    
    search.build(spr, torch.optim.Adam, optimizer_kwargs = {'lr': 1e-1}, transforms = transforms)

    logs, ticket = search.optimize_mask(dt, 15, CARDINALITY, dynamic_epochs = False)

    search.finish()

    if rank == 0: plot_logs_concrete(logs, name = name + f"_{100 * spr:.2f}")

    #print(f"Searched Ticket Sparsity: {100 * (ticket.sum()/ticket.numel()):.2f}")

    return state, (ticket, imp_ticket)


def _make_trainer(rank, world_size, state, is_vgg, ticket):

    model = ddp_network(rank, world_size, is_vgg, depth = 16)
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

def run_check_and_export(rank, world_size, state, spr, concrete_ticket,
                         salient_tickets, imp_ticket, is_vgg, dt, transforms):


    model = ddp_network(rank, world_size, is_vgg, depth = 16)
    model.load_state_dict(state)

    
    search = GradientNormSearch(rank, world_size, 
                                model.module, dt, 
                                spr, transforms,)

    output = dict()

    output["imp"] = _print_collect_return_fitness(rank, "imp", search.calculate_fitness_given(imp_ticket))

    output["concrete"] = _print_collect_return_fitness(rank, "concrete", search.calculate_fitness_given(concrete_ticket))

    output["grasp"] = _print_collect_return_fitness(rank, "grasp", search.calculate_fitness_given(salient_tickets[0]))

    output["grasp_improved"] = _print_collect_return_fitness(rank, "grasp_improved", search.calculate_fitness_given(salient_tickets[1]))

    output["dense"] = _print_collect_return_fitness(rank, "dense", search.calculate_fitness_given(torch.ones_like(concrete_ticket)))

    output["magnitude"] = _print_collect_return_fitness(rank, "magnitude", search.calculate_fitness_magnitude())

    output["random"] = _print_collect_return_fitness(rank, "random", search.calculate_fitness_random())
    
    return output

def main(rank, world_size, name: str, sp_exp: list, **kwargs):

    #is_grasp = sp_exp.pop(-1) == 1
    is_vgg = sp_exp.pop(-1) == 1

    imp_name = f"imp_{"vgg16" if is_vgg else "resnet20"}_{name[-1]}"

    dataAug = torch.jit.script(DataAugmentation().to('cuda'))
    resize = torch.jit.script(Resize().to('cuda'))
    normalize = torch.jit.script(Normalize().to('cuda'))
    center_crop = torch.jit.script(CenterCrop().to('cuda'))

    logs = defaultdict(dict)

    dt, _ = get_loaders(rank, world_size, batch_size = 512, validation = False)

    partial = get_partial_train_loader(rank, world_size, 4, batch_size = 512)

    for spe in reversed(sp_exp):

        spr = 0.8**spe

        init_name = "init_" + name

        ckpt = torch.load(f"./logs/WEIGHTS/init_{imp_name}_100.00.pt", map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}, 
                          weights_only = True)['model']

        _, salient_tickets = get_salient_ticket(rank, world_size, ckpt, is_vgg, (resize, normalize, center_crop,), spr)

        _, (concrete_ticket, imp_ticket) = get_concrete_ticket_and_imp(rank, world_size, ckpt, is_vgg, dt, (resize, normalize, center_crop,), spr, init_name, imp_name)

        log = run_check_and_export(rank, world_size, ckpt, spr, concrete_ticket, salient_tickets, 
                                   imp_ticket, is_vgg, partial, (resize, normalize, center_crop,))

        if rank == 0: print(f"SPARSITY: {spr * 100 :.2f} || INIT || {log.items()}")

        logs[spe]['init'] = log
        
        
        rewind_name = "rewind_" + name

        ckpt = torch.load(f"./logs/WEIGHTS/rewind_{imp_name}_100.00.pt", map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}, 
                          weights_only = True)['model']

        _, salient_ticket = get_salient_ticket(rank, world_size, ckpt, is_vgg, (resize, normalize, center_crop,), spr)

        _, (concrete_ticket, imp_ticket) = get_concrete_ticket_and_imp(rank, world_size, ckpt, is_vgg, dt, (resize, normalize, center_crop,), spr, rewind_name, imp_name)

        log = run_check_and_export(rank, world_size, ckpt, spr, concrete_ticket, salient_ticket, 
                                   imp_ticket, is_vgg, partial, (resize, normalize, center_crop,))

        if rank == 0: print(f"SPARSITY: {spr * 100 :.2f} || REWIND || {log.items()}")

        logs[spe]['rewind'] = log
        


    with open(f"./logs/PICKLES/{name}_comparison.json", "w") as f:
        json.dump(logs, f, indent = 6)
