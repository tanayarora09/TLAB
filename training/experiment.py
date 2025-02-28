import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from data.cifar10 import *

from utils.serialization_utils import logs_to_pickle, save_tensor
from utils.training_utils import plot_logs

from training.VGG import VGG_DGTS
from models.VGG import VGG

import json
import pickle
import gc

def main(rank, world_size, name: str, args: list, lock, shared_list, **kwargs):

    EPOCHS = int(args[0])
    CARDINALITY = 391
    
    name += "_" + ",".join([str(item) for item in args])

    dataAug = torch.jit.script(DataAugmentation().to('cuda'))
    resize = torch.jit.script(Resize().to('cuda'))
    normalize = torch.jit.script(Normalize().to('cuda'))
    center_crop = torch.jit.script(CenterCrop().to('cuda'))

    model = VGG(depth = 16, rank = rank).to("cuda")

    model = DDP(model, 
                device_ids = [rank],
                output_device = rank, 
                gradient_as_bucket_view = True)
 
    T = VGG_DGTS(model = model, rank = rank, world_size = world_size, 
                 lock = lock, dynamic_list = shared_list)

    del model

    torch.cuda.empty_cache()
    gc.collect()

    dt, dv = get_loaders(rank, world_size, batch_size = 128) 

    T.build(sparsity_rate = 0.8, experiment_args = args[1:], type_of_exp = 2,
            optimizer = torch.optim.SGD(T.m.parameters(), 0.1, momentum = 0.9, weight_decay = 1e-4),
            loss = torch.nn.CrossEntropyLoss(reduction = "sum").to('cuda'),
            collective_transforms = (resize, normalize), train_transforms = (dataAug,),
            eval_transforms = (center_crop,), final_collective_transforms = tuple(),
            scale_loss = True, gradient_clipnorm = 2.0)
    
    T.fit(dt, dv, EPOCHS, CARDINALITY, name + "break", save = False, verbose = False, validate = False)


    T.init_capture_hooks()
    with torch.no_grad():
        sampler_offset = 0
        while not T._pruned:
            
            x, y = custom_fetch_data(dt, sampler_offset = sampler_offset)
            x, y = x.to('cuda'), y.to('cuda')

            for tr in T.cT: x = tr(x)
            for tr in T.eT: x = tr(x)
            for tr in T.fcT: x = tr(x)

            T.mm(x)

            ticket, fitness = T.search(x, y)
            T.prune_model(ticket)
            T.fitnesses.append((T.mm.sparsity.item(), fitness))
            
            sampler_offset += 1

            if T.mm.sparsity_d <= T.desired_sparsity:
                T._pruned = True

    T.remove_handles()

    """
    logs = T.fit(dt, dv, EPOCHS, CARDINALITY, name, save = False, validate = True, verbose = False)
    
    T.evaluate(dt)

    if rank == 0:

        print(f"Train Set: {T.metric_results()}.")

        plot_logs(logs, EPOCHS, name, CARDINALITY)
        logs_to_pickle(logs, name)
        logs_to_pickle(T.fitnesses, name, "fitnesses")
        with open(f"./logs/PICKLES/{name}_prunes.json", "w") as f:
            json.dump(T.prunes, f, indent = 6)
    
    torch.distributed.barrier(device_ids = [rank])

    """

    T.mm.export_ticket(name)

    logs_final = T.fit(dt, dv, EPOCHS, CARDINALITY, name, 
                     save = False, verbose = False)

    if (rank == 0):
        
        plot_logs(logs_final, EPOCHS, name, steps = CARDINALITY, start = 0)
        
        logs_to_pickle(logs_final, name)

    torch.distributed.barrier(device_ids = [rank])