import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from data.index import get_data_object

from utils.serialization_utils import logs_to_pickle, save_tensor
from utils.training_utils import plot_logs
from utils.search_utils import plot_logs_concrete

from training.VGG import VGG_CNN
from training.ResNet import ResNet_CNN
from search.salient import *
from search.concrete import *
from search.genetic import *
from models.VGG import VGG
from models.base import MaskedModel
from models.ResNet import ResNet


import json
import pickle
import gc
from collections import defaultdict

import h5py

EPOCHS = 160


CONCRETE_EXPERIMENTS = {0: ("Loss", SNIPConcrete, SNIP_Pruner, LossSearch),
                        1: ("Gradnorm", GraSPConcrete, GraSP_Pruner, GradientNormSearch),
                        2: ("KldLogit", KldLogit, KldLogit_Pruner, KldLogitSearch),
                        3: ("MseFeature", NormalizedMseFeatures, MSE_Pruner, NormalizedMSESearch),
                        4: ("GradMatch", StepAlignmentConcrete, GradMatch_Pruner, GradMatchSearch),
                        5: ("DeltaLoss", LossChangeConcrete, SNIP_Pruner, DeltaLossSearch), 
                        6: ("OldKld", OldKld, OldKld_Pruner, OldKldSearch),
                        }

def ddp_network(rank, world_size, is_vgg):

    depth = 16 if is_vgg else 20
    
    model = (VGG if is_vgg else ResNet)(depth = depth, rank = rank, world_size = world_size, custom_init = True).cuda()
    
    if world_size > 1:
        model = DDP(model, 
            device_ids = [rank],
            output_device = rank, 
            gradient_as_bucket_view = True)

    return model

def _get_train(rank, world_size, handle):
    dt, _ = handle.get_loaders(rank, world_size, train=True, validation=False)
    return dt

def get_salient_ticket(rank, world_size, steps, state, type_of_salient, is_vgg, handle, spr):

    model = ddp_network(rank, world_size, is_vgg)
    model.load_state_dict(state)

    if rank == 0:

        model_to_inspect = model.module if world_size > 1 else model

        inp_args = {'rank': 0, 'world_size': 1, 'model': model_to_inspect}

        if type_of_salient == 3 or type_of_salient == 6: 
            captures = []
            fcaptures = []
            for bname, block in model_to_inspect.named_children():
                for lname, layer in block.named_children():
                    if lname.endswith("relu"): captures.append(layer)
                    elif lname.endswith("fc"): fcaptures.append((layer, lambda x: torch.softmax(x, dim = 1)))
            inp_args.update({"capture_layers": captures, "fake_capture_layers": fcaptures})

        pruner = CONCRETE_EXPERIMENTS[type_of_salient][2](**inp_args)
        # Get transforms from handle
        tt, et, ft = handle.tef_transforms()
        pruner.build(spr, (tt, et, ft), input = None)
        ticket = pruner.grad_mask(steps = steps)
        pruner.finish()


    else:
        ticket = torch.zeros(model.module.num_prunable, dtype = torch.bool, device = 'cuda')

    if world_size > 1:
        torch.distributed.barrier(device_ids = [rank])
        torch.distributed.broadcast(ticket, src = 0)

    return state, ticket


def get_concrete_ticket_and_imp(rank, world_size, state, type_of_concrete, is_gradnorm, is_vgg, dt, handle, spr, name):

    model = ddp_network(rank, world_size, is_vgg)
    model.load_state_dict(state)

    model_to_inspect = model.module if world_size > 1 else model

    inp_args = {'rank': rank, 'world_size': world_size, 'model': model_to_inspect}

    if type_of_concrete == 3 or type_of_concrete == 6: 
        captures = []
        fcaptures = []
        for bname, block in model_to_inspect.named_children():
            for lname, layer in block.named_children():
                if lname.endswith("relu"): captures.append(layer)
                elif lname.endswith("fc"): fcaptures.append((layer, lambda x: torch.softmax(x, dim = 1)))
        inp_args.update({"capture_layers": captures, "fake_capture_layers": fcaptures})

    search = CONCRETE_EXPERIMENTS[type_of_concrete][1](**inp_args)
    
    # Get transforms from handle
    tt, et, ft = handle.tef_transforms()
    search.build(spr, torch.optim.Adam, optimizer_kwargs = {'lr': 1e-1}, transforms = (tt, et, ft), use_gradnorm_approach = is_gradnorm)

    CARDINALITY = handle.cardinality()
    logs, ticket = search.optimize_mask(dt, 20, CARDINALITY, dynamic_epochs = False)

    search.finish()

    if rank == 0: plot_logs_concrete(logs, name = name + f"_{100 * spr:.3e}")

    return state, ticket

def _print_collect_return_fitness(rank, name, fitness):
    fitness = torch.as_tensor(fitness, dtype = torch.float64, device = 'cuda')
    dist.all_reduce(fitness, op = dist.ReduceOp.AVG)
    dist.barrier(device_ids = [rank])
    if rank == 0: print(f"{name.capitalize()} Fitness: {fitness}")
    return fitness.item()

def run_check_and_export(rank, world_size, state, spr, type_of_search, is_vgg, dt, handle, tickets):


    model = ddp_network(rank, world_size, is_vgg)
    model.load_state_dict(state)

    model_to_inspect = model.module if world_size > 1 else model

    # Get transforms from handle
    tt, et, ft = handle.tef_transforms()
    
    inp_args = {'rank': rank, 'world_size': world_size, 'model': model_to_inspect, 'input': dt, 
                'sparsity_rate': spr, 'transforms': (tt, et, ft)}

    if type_of_search == 3 or type_of_search == 6: 
        captures = []
        fcaptures = []
        for bname, block in model_to_inspect.named_children():
            for lname, layer in block.named_children():
                if lname.endswith("relu"): captures.append(layer)
                elif lname.endswith("fc"): fcaptures.append((layer, lambda x: torch.softmax(x, dim = 1)))
        inp_args.update({"capture_layers": captures, "fake_capture_layers": fcaptures})


    search = CONCRETE_EXPERIMENTS[type_of_search][3](**inp_args)

    output = dict()

    for tname, ticket in tickets:
        output[tname] = _print_collect_return_fitness(rank, tname, search.calculate_fitness_given(ticket))

    output["dense"] = _print_collect_return_fitness(rank, "dense", search.calculate_fitness_given(torch.ones(model_to_inspect.num_prunable, dtype = torch.bool)))

    output["magnitude"] = _print_collect_return_fitness(rank, "magnitude", search.calculate_fitness_magnitude())

    output["random"] = _print_collect_return_fitness(rank, "random", search.calculate_fitness_random())
    
    return output

def main(rank, world_size, name: str, args: list, **kwargs):

    is_vgg = args.pop(-1) == 1
    type_of_concrete = args.pop(-1)

    sp_exp = list(range(2, 43 if is_vgg else 33))

    if rank == 0: print(f"Running IMP Comparison for: {CONCRETE_EXPERIMENTS[type_of_concrete][0]} on {'VGG-16' if is_vgg else 'ResNet-20'}")

    name = CONCRETE_EXPERIMENTS[type_of_concrete][0].lower() + "_" + ("vgg16" if is_vgg else "resnet20") + "_" +  name

    imp_name = f"late_rewind_imp_{"vgg16" if is_vgg else "resnet20"}_{name[-1]}"

    # Create DataHandle for centralized access to data and hparams
    handle = get_data_object("cifar10")

    logs = defaultdict(dict)

    dt, _ = handle.get_loaders(rank, world_size, train=True, validation=False)

    partial = handle.get_partial_train_loader(rank, world_size, data_fraction_factor=4)

    for spe in reversed(sp_exp):

        spr = 0.8**spe

        _ = (VGG if is_vgg else ResNet)(rank, world_size, depth = 16 if is_vgg else 20)
        imp_ticket = _.load_ticket(imp_name, root = 0, entry_name = f"{spr*100:.3e}")

        init_name = "init_" + name

        ckpt = torch.load(f"./logs/WEIGHTS/init_{imp_name}_{100.0:.3e}.pt", map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}, 
                          weights_only = True)['model']
        
        if world_size == 1: ckpt = {k[7:]:v for k,v in ckpt.items()}

        _, salient_ticket = get_salient_ticket(rank, world_size, 1, ckpt, type_of_concrete, is_vgg, handle, spr)

        _, iterative_ticket = get_salient_ticket(rank, world_size, 100, ckpt, type_of_concrete, is_vgg, handle, spr)

        _, gradbalance_ticket = get_concrete_ticket_and_imp(rank, world_size, ckpt, type_of_concrete, True, is_vgg, dt, 
                                                           handle, spr, init_name)

        _, multiplier_ticket = get_concrete_ticket_and_imp(rank, world_size, ckpt, type_of_concrete, False, is_vgg, dt, 
                                                          handle, spr, init_name)

        tickets = [('imp', imp_ticket), ('salient', salient_ticket), ('iterative', iterative_ticket), 
                   ('gradbalance', gradbalance_ticket), ('multiplier', multiplier_ticket)]

        log = run_check_and_export(rank, world_size, ckpt, spr, type_of_concrete, is_vgg, partial, handle, tickets)

        if rank == 0: print(f"SPARSITY: {spr * 100 :.3e} || INIT || {log.items()}")

        logs[spe]['init'] = log
        
        
        rewind_name = "rewind_" + name

        ckpt = torch.load(f"./logs/WEIGHTS/rewind_{imp_name}_{100.0:.3e}.pt", map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}, 
                          weights_only = True)['model']

        if world_size == 1: ckpt = {k[7:]:v for k,v in ckpt.items()}

        _, salient_ticket = get_salient_ticket(rank, world_size, 1, ckpt, type_of_concrete, is_vgg, handle, spr)

        _, iterative_ticket = get_salient_ticket(rank, world_size, 100, ckpt, type_of_concrete, is_vgg, handle, spr)

        _, gradbalance_ticket = get_concrete_ticket_and_imp(rank, world_size, ckpt, type_of_concrete, True, is_vgg, dt, 
                                                           handle, spr, rewind_name)

        _, multiplier_ticket = get_concrete_ticket_and_imp(rank, world_size, ckpt, type_of_concrete, False, is_vgg, dt, 
                                                          handle, spr, rewind_name)

        tickets = [('imp', imp_ticket), ('salient', salient_ticket), ('iterative', iterative_ticket), 
                   ('gradbalance', gradbalance_ticket), ('multiplier', multiplier_ticket)]

        log = run_check_and_export(rank, world_size, ckpt, spr, type_of_concrete, is_vgg, partial, handle, tickets)

        if rank == 0: print(f"SPARSITY: {spr * 100 :.3e} || REWIND || {log.items()}")

        logs[spe]['rewind'] = log
        

    if rank == 0:
        with open(f"./logs/RESULTS/{name}_comparison.json", "w") as f:
            json.dump(logs, f, indent = 6)