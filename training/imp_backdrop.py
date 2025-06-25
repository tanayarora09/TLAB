import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from data.cifar10 import *

from utils.serialization_utils import logs_to_pickle, save_tensor
from utils.training_utils import plot_logs

from training.VGG import VGG_CNN
from search.salient import DGTS
from models.VGG import VGG, BaseModel

import json
import pickle
import gc

import h5py

EPOCHS = 160
CARDINALITY = 98

def get_ticket(rank, world_size):

    pass




def main(rank, world_size, name: str, args: list, lock, shared_list, **kwargs):

    dataAug = torch.jit.script(DataAugmentation().to('cuda'))
    resize = torch.jit.script(Resize().to('cuda'))
    normalize = torch.jit.script(Normalize().to('cuda'))
    center_crop = torch.jit.script(CenterCrop().to('cuda'))

    model = VGG(depth = 16, rank = rank, world_size = world_size).to("cuda")

    model = DDP(model, 
                device_ids = [rank],
                output_device = rank, 
                gradient_as_bucket_view = True)

    gc.collect()
    torch.cuda.empty_cache()

    #T.load_ckpt(f"NAIVE_MIDDLE_TRUE1_160.0,100.0,160.0,31279.0,5.0,0.055_80.0", "rewind", zero_out = False)

    ckpt = torch.load("/u/tanaya_guest/tlab/tanaya/TLAB_SIHL/logs/WEIGHTS/rewind_ProofOfConceptMG_0_f_100.00.pt",#"/u/tanaya_guest/tlab/tanaya/TLAB_SIHL/logs/WEIGHTS/rewind_ProofOfConcept_0_f_100.00.pt", 
                      map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}, weights_only = True)['model']

    model.load_state_dict(ckpt)    

    if not FIRST_:

        T = VGG_CNN(model = model, rank = rank, world_size = world_size)

        #del model

        #torch.cuda.empty_cache()
        #gc.collect()

        dt, dv = get_loaders(rank, world_size, batch_size = 512) 

        T.build(optimizer = torch.optim.SGD, optimizer_kwargs = {'lr': 0.1, 'momentum': 0.9, 'weight_decay' : 1e-3},
                loss = torch.nn.CrossEntropyLoss(reduction = "sum").to('cuda'),
                collective_transforms = (resize, normalize), train_transforms = (dataAug,),
                eval_transforms = (center_crop,), final_collective_transforms = tuple(),
                scale_loss = True, gradient_clipnorm = 2.0)

        logs_init = T.fit(dt, dv, 67, CARDINALITY, name + "break", verbose = False, save = False, validate = True)

        del dt, dv

    else:
        model.load_state_dict(torch.load(f"logs/WEIGHTS/final_STRENGTHEN_20_{EXPERIMENT}break.pt",#"/u/tanaya_guest/tlab/tanaya/TLAB_SIHL/logs/WEIGHTS/rewind_ProofOfConcept_0_f_100.00.pt", 
                      map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}, weights_only = True)['model'])

    if rank == 0:
        with h5py.File("/u/tanaya_guest/tlab/tanaya/TLAB_SIHL/logs/TICKETS/ProofOfConceptMG_0_f.h5", 'r') as f:
            winning_ticket = torch.as_tensor(f[f"{(100 * sp):.2f}"][:], device = 'cuda')
    else: winning_ticket = torch.zeros_like(model.module.get_buffer("MASK"), device = 'cuda')

    dist.barrier(device_ids = [rank])
    dist.broadcast(winning_ticket, src = 0)

    captures = list()
    fcaptures = list()

    for _, block in model.module.named_children():
        for n, layer in block.named_children():
            if n.endswith("relu"): captures.append(layer)
            elif n.endswith("fc"): fcaptures.append((layer, nn.ReLU()))

    if rank == 0: print("\n\n\n")

    GTS = DGTS(rank, world_size, lock, shared_list,
               model.module, captures, fcaptures)
    
    partial_dt = get_partial_train_loader(rank, world_size, 3, batch_size = 512)

    GTS.build(search_iterations = int(args[1]),
              search_size = int(args[2]), possible_children = int(args[3]),
              sparsity_rate = sp, mutation_temperature = float(args[4]),
              final_mutation_temperature = float(args[5]),
              elite_percentage = 0.05,
              init_by_mutate = True, transforms = (resize, normalize, center_crop,),
              input = partial_dt)


    model.module.prune_by_mg(sp, iteration = 1)
    mg_ticket = model.module.export_ticket_cpu()

    model.module.pls()
    model.module.cc()
    model.module.reset_ticket()

    mgfit = torch.as_tensor(GTS.calculate_fitness_given(mg_ticket), dtype = torch.float64, device = 'cuda')
    print(f"[rank {rank}] Magnitude Fitness: ", mgfit.item())
    dist.all_reduce(mgfit, op = dist.ReduceOp.AVG)
    dist.barrier(device_ids = [rank])

    if rank == 0: print(f"Magnitude Fitness: ", mgfit.item())

    winfit = torch.as_tensor(GTS.calculate_fitness_given(winning_ticket), dtype = torch.float64, device = 'cuda')
    print(f"[rank {rank}] Winning Fitness: ", winfit.item()) 
    dist.all_reduce(winfit, op = dist.ReduceOp.AVG)
    dist.barrier(device_ids = [rank])

    if rank == 0: print(f"Winning Fitness: ", winfit.item())

    randfit = torch.as_tensor(GTS.calculate_fitness_random(), dtype = torch.float64, device = 'cuda')
    print(f"[rank {rank}] Random Fitness: ", randfit.item()) 
    dist.all_reduce(randfit, op = dist.ReduceOp.AVG)
    dist.barrier(device_ids = [rank])

    if rank == 0: print(f"Random Fitness: ", randfit.item()) 

    ticket, fitness = GTS.search(winning_ticket)


    if rank == 0: print("Best Fitness: ", fitness, "\n\n\n")

    if rank == 0: 
        with open(f"./logs/PICKLES/{name}_best_fitnesses.json", "w", encoding = "utf-8") as f:
            json.dump({'search':GTS.best_log, 'random':randfit.item(), 'imp':winfit.item(), 
                       'magnitude':mgfit.item()}, f, ensure_ascii = False, indent = 4)

    del partial_dt, GTS

    gc.collect()
    torch.cuda.empty_cache()

    dist.barrier(device_ids = [rank]) 

    model.module.set_ticket(ticket)#T.prune_model(ticket)

    model.module.export_ticket(name)
    
    model.load_state_dict(ckpt)

    #ckpt = torch.load("/u/tanaya_guest/tlab/tanaya/TLAB_SIHL/logs/WEIGHTS/rewind_POC_1e-3WD1_100.00.pt", 
    #                  map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}, weights_only = True)['model']

    #model.load_state_dict(ckpt)    

    del ckpt

    #T.load_ckpt(f"NAIVE_MIDDLE_TRUE0_160.0,100.0,160.0,31279.0,5.0,0.055_80.0", "rewind")

    """T.build(optimizer = torch.optim.SGD(T.m.parameters(), 0.1, momentum = 0.9, weight_decay = 1e-3),
            loss = torch.nn.CrossEntropyLoss(reduction = "sum").to('cuda'),
            collective_transforms = (resize, normalize), train_transforms = (dataAug,),
            eval_transforms = (center_crop,), final_collective_transforms = tuple(),
            scale_loss = True, gradient_clipnorm = 2.0)"""

    dt, dv = get_loaders(rank, world_size, batch_size = 512) 

    T = VGG_CNN(model, rank, world_size)

    T.build(optimizer = torch.optim.SGD, optimizer_kwargs = {'lr': 0.1, 'momentum': 0.9, 'weight_decay' : 1e-3},
            loss = torch.nn.CrossEntropyLoss(reduction = "sum").to('cuda'),
            collective_transforms = (resize, normalize), train_transforms = (dataAug,),
            eval_transforms = (center_crop,), final_collective_transforms = tuple(),
            scale_loss = True, gradient_clipnorm = 2.0)

    logs = T.fit(dt, dv, EPOCHS, CARDINALITY, name, 
                        save = False, verbose = False,
                        sampler_offset = 0,
                        start = 13)

    if (rank == 0):
        
        logs_to_pickle(logs, name)
        if not FIRST_: logs_to_pickle(logs_init, name + "break")
        
        plot_logs(logs, EPOCHS, name, steps = CARDINALITY, start = 13)
        if not FIRST_: plot_logs(logs_init, 67, name + "break", steps = CARDINALITY, start = 0)
