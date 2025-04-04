import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from data.cifar10 import *

from utils.serialization_utils import logs_to_pickle, save_tensor
from utils.training_utils import plot_logs

from training.VGG import VGG_CNN
from search.distributed import DGTS
from models.VGG import VGG, BaseModel

import json
import pickle
import gc

import h5py

def main(rank, world_size, name: str, args: list, lock, shared_list, **kwargs):

    EPOCHS = int(args[0])
    CARDINALITY = 98

    EXPERIMENT = int(name[-1])

    SPIDX = int(args[6])

    sp = 0.8**SPIDX

    name = name[:-1] + f"_{SPIDX}_{EXPERIMENT}"

    dataAug = torch.jit.script(DataAugmentation().to('cuda'))
    resize = torch.jit.script(Resize().to('cuda'))
    normalize = torch.jit.script(Normalize().to('cuda'))
    center_crop = torch.jit.script(CenterCrop().to('cuda'))

    model = VGG(depth = 19, rank = rank, world_size = world_size).to("cuda")

    model = DDP(model, 
                device_ids = [rank],
                output_device = rank, 
                gradient_as_bucket_view = True)

    gc.collect()
    torch.cuda.empty_cache()

    imp_name = ["ProofOfConcept_0_f", "ProofOfConcept_1_f"][EXPERIMENT % 2]

    ckpt = torch.load(f"/u/tanaya_guest/tlab/tanaya/TLAB_SIHL/logs/WEIGHTS/rewind_{imp_name}_100.00.pt",
                      map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}, weights_only = True)['model']

    model.load_state_dict(ckpt)    

    if rank == 0:
        with h5py.File(f"/u/tanaya_guest/tlab/tanaya/TLAB_SIHL/logs/TICKETS/{imp_name}.h5", 'r') as f:
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

    #if rank == 0: print("\n\n\n")

    GTS = DGTS(rank, world_size, lock, shared_list,
               model.module, captures, fcaptures)
    
    partial_dt = get_partial_train_loader(rank, world_size, 3, batch_size = 512)

    GTS.build(search_iterations = int(args[1]),
              search_size = int(args[2]), possible_children = int(args[3]),
              sparsity_rate = sp, mutation_temperature = float(args[4]),
              final_mutation_temperature = float(args[5]),
              elite_percentage = 0.05, init_by_mutate = True, 
              transforms = (resize, normalize, center_crop,),
              input = partial_dt)

    model.module.prune_by_mg(sp, iteration = 1)
    mg_ticket = model.module.export_ticket_cpu()
    #model.module.pls()
    #model.module.cc()
    model.module.reset_ticket()

    mgfit = torch.as_tensor(GTS.calculate_fitness_given(mg_ticket), dtype = torch.float64, device = 'cuda')
    #print(f"[rank {rank}] Magnitude Fitness: ", mgfit.item())
    dist.all_reduce(mgfit, op = dist.ReduceOp.AVG)
    dist.barrier(device_ids = [rank])

    if rank == 0: print(f"Magnitude Fitness: ", mgfit.item())

    winfit = torch.as_tensor(GTS.calculate_fitness_given(winning_ticket), dtype = torch.float64, device = 'cuda')
    #print(f"[rank {rank}] Winning Fitness: ", winfit.item()) 
    dist.all_reduce(winfit, op = dist.ReduceOp.AVG)
    dist.barrier(device_ids = [rank])

    if rank == 0: print(f"Winning Fitness: ", winfit.item())

    randfit = torch.as_tensor(GTS.calculate_fitness_random(), dtype = torch.float64, device = 'cuda')
    #print(f"[rank {rank}] Random Fitness: ", randfit.item()) 
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
    
    for rerun in range(2):

        model.load_state_dict(ckpt)

        dt, dv = get_loaders(rank, world_size, batch_size = 512) 

        T = VGG_CNN(model, rank, world_size)

        T.build(optimizer = torch.optim.SGD, optimizer_kwargs = {'lr': 0.1, 'momentum': 0.9, 'weight_decay' : 1e-3},
                loss = torch.nn.CrossEntropyLoss(reduction = "sum").to('cuda'),
                collective_transforms = (resize, normalize), train_transforms = (dataAug,),
                eval_transforms = (center_crop,), final_collective_transforms = tuple(),
                scale_loss = True, gradient_clipnorm = 2.0)

        logs = T.fit(dt, dv, EPOCHS, CARDINALITY, f"{name}_{rerun}", 
                            save = False, verbose = False,
                            sampler_offset = 0,
                            start = 13)

        if (rank == 0):
            
            logs_to_pickle(logs, f"{name}_{rerun}")
            
            plot_logs(logs, EPOCHS, f"{name}_{rerun}" , steps = CARDINALITY, start = 13)


    """
    while not T._pruned:

        tmp_name = f"{name}_{T.mm.sparsity.item()*0.8:.1f}"

        logs = T.fit(dt, dv, EPOCHS, CARDINALITY, tmp_name, 
                     save = True if (T.mm.sparsity_d.item() == 1.00) else False,
                     rewind_iter = 20000,
                     verbose = False, 
                     sampler_offset = sampler_offset,
                     validate = False)
        
        T.init_capture_hooks()

        ticket, fitness = collect_and_search(dt, T)

        T.remove_handles()

        T.prune_model(ticket)
        T.fitnesses.append((T.mm.sparsity.item(), fitness))

        if T.mm.sparsity_d <= T.desired_sparsity:
            T._pruned = True

        sampler_offset += 1

        if (rank == 0):

            logs_to_pickle(logs, tmp_name)
            T.mm.export_ticket(name, entry_name = f"{T.mm.sparsity.item():.1f}")

        torch.distributed.barrier(device_ids = [rank])

        T.load_ckpt(f"{name[:-5] + "0.055"}_{80.0:.1f}", "rewind")

        T.build(sparsity_rate = 0.8, experiment_args = args[1:], type_of_exp = 2,
            optimizer = torch.optim.SGD(T.m.parameters(), 0.1, momentum = 0.9, weight_decay = 1e-3),
            loss = torch.nn.CrossEntropyLoss(reduction = "sum").to('cuda'),
            collective_transforms = (resize, normalize), train_transforms = (dataAug,),
            eval_transforms = (center_crop,), final_collective_transforms = tuple(),
            scale_loss = True, gradient_clipnorm = 2.0)

        torch.distributed.barrier(device_ids = [rank])

    T.mm.export_ticket(name)

    logs_final = T.fit(dt, dv, EPOCHS, CARDINALITY, name, 
                     save = False, verbose = False,
                     sampler_offset = sampler_offset)

    if (rank == 0):
        
        plot_logs(logs_final, EPOCHS, name, steps = CARDINALITY, start = 0)
        
        logs_to_pickle(logs_final, name)

    torch.distributed.barrier(device_ids = [rank])"""