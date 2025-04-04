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

    CARDINALITY = 98

    EXPERIMENT = int(name[-1])

    name = name[:-1] + f"_{EXPERIMENT}"

    dataAug = torch.jit.script(DataAugmentation().to('cuda'))
    resize = torch.jit.script(Resize().to('cuda'))
    normalize = torch.jit.script(Normalize().to('cuda'))
    center_crop = torch.jit.script(CenterCrop().to('cuda'))

    model = VGG(depth = 19, rank = rank, world_size = world_size,
                custom_init = True).to("cuda")

    model = DDP(model, 
                device_ids = [rank],
                output_device = rank, 
                gradient_as_bucket_view = True)

    gc.collect()
    torch.cuda.empty_cache()

    model.load_state_dict(torch.load(f"logs/WEIGHTS/final_MGSEARCH_17_{EXPERIMENT}break.pt",#"/u/tanaya_guest/tlab/tanaya/TLAB_SIHL/logs/WEIGHTS/rewind_ProofOfConcept_0_f_100.00.pt", 
                      map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}, weights_only = True)['model'])
    
    model.eval()

    captures = list()
    fcaptures = list()

    for _, block in model.module.named_children():
        for n, layer in block.named_children():
            if n.endswith("relu"): captures.append(layer)
            elif n.endswith("fc"): fcaptures.append((layer, nn.ReLU()))

    output = dict()

    tickets = dict()

    for spidx in range(1, 31):
        tickets[spidx] = list()
        for _ in range(3):
            model.module.prune_random(0.8**spidx, distributed = True)
            tickets[spidx].append(model.module.export_ticket_cpu())
            print(model.module.sparsity)
            model.module.reset_ticket()


    for batch_count in range(2, 99, 3):    

        print(f"[rank {rank}] Testing {batch_count} batches.")    

        output[batch_count] = dict()

        GTS = DGTS(rank, world_size, lock, shared_list,
                model.module, captures, fcaptures)
    
        partial_dt = get_partial_train_loader(rank, world_size, batch_count = batch_count, batch_size = 512,)

        GTS.build(search_iterations = 1,
                search_size = 1, possible_children = 0,
                sparsity_rate = 1.0, mutation_temperature = 0.5,
                final_mutation_temperature = 0.4,
                elite_percentage = 0.05, init_by_mutate = True, 
                transforms = (resize, normalize, center_crop,),
                input = partial_dt)


        for spidx in range(1, 31):
            output[batch_count][spidx] = list()
            for i in range(2):
                if rank == 0: 
                    fitness_list = [torch.zeros(1, device = 'cuda', dtype = torch.float64) for _ in range(world_size)]
                else: fitness_list = None
                fitness = torch.as_tensor([GTS.calculate_fitness_given(tickets[spidx][i])], dtype = torch.float64, device = 'cuda')
                dist.gather(fitness, fitness_list, dst = 0)
                if rank == 0: output[batch_count][spidx].append([fitness.item() for fitness in fitness_list])
            
            print(f"[rank {rank}] Tested {(100 * 0.8 ** spidx):.3f} Sparsity. {output[batch_count][spidx]}")

            
    if rank == 0: 
        with open(f"./logs/PICKLES/{name}_data_fitnesses.json", "w", encoding = "utf-8") as f:
            json.dump(output, f, ensure_ascii = False, indent = 4)

    del partial_dt, GTS

    gc.collect()
    torch.cuda.empty_cache()

    dist.barrier(device_ids = [rank]) 
