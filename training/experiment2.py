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
    

    dataAug = torch.jit.script(DataAugmentation().to('cuda'))
    resize = torch.jit.script(Resize().to('cuda'))
    normalize = torch.jit.script(Normalize().to('cuda'))
    center_crop = torch.jit.script(CenterCrop().to('cuda'))

    model = VGG(depth = 19, rank = rank).to("cuda")

    #if name.endswith("0"): model.set_ticket(model.load_ticket("REWINDA_EQUAL0_160.0,90.0,160.0,10.0,5.0,0.055"))

    name += "_" + ",".join([str(item) for item in args])

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

    T.build(sparsity_rate = 0.8, experiment_args = args[1:],
            optimizer = torch.optim.SGD, optimizer_kwargs = {'lr': 0.1, 'momentum': 0.9, 'weight_decay' : 1e-3},
            loss = torch.nn.CrossEntropyLoss(reduction = "sum").to('cuda'),
            collective_transforms = (resize, normalize), train_transforms = (dataAug,),
            eval_transforms = (center_crop,), final_collective_transforms = tuple(),
            scale_loss = True, gradient_clipnorm = 2.0)
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
    sampler_offset = 0

    while not T._pruned:

        tmp_name = f"{name}_{T.mm.sparsity.item()*0.8:.1f}"

        logs = T.fit(dt, dv, EPOCHS, CARDINALITY, tmp_name, 
                     save = False,
                     verbose = False, 
                     sampler_offset = sampler_offset,
                     validate = False)

        sampler_offset += 1
        
        if (rank == 0):

            print("Sparsity: ", T.mm.sparsity)
            #print("\n\n\n")
            logs_to_pickle(T.fitnesses, tmp_name, suffix = "fitnesses")

            #logs_to_pickle(T.prunes, name, suffix = "prunes")

            #with open(f"./logs/PICKLES/{tmp_name}_prunes.json", "w") as f:
            #    json.dump(T.prunes, f, indent = 6)

            T.fitnesses.clear()
            T.prunes.clear()

            logs_to_pickle(logs, tmp_name)
            T.mm.export_ticket(name, entry_name = f"{T.mm.sparsity.item():.1f}")

        torch.distributed.barrier(device_ids = [rank])

        T.load_ckpt(tmp_name, "rewind")

        T.build(sparsity_rate = 0.8, experiment_args = args[1:],
            optimizer = torch.optim.SGD, optimizer_kwargs = {'lr': 0.1, 'momentum': 0.9, 'weight_decay' : 1e-3},
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

    torch.distributed.barrier(device_ids = [rank])