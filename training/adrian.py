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

    model = VGG(depth = 19, rank = rank)

    model = DDP(model.to('cuda'), 
                device_ids = [rank],
                output_device = rank, 
                gradient_as_bucket_view = True)
 
    #model = torch.compile(model)
    
    T = VGG_DGTS(model = model, rank = rank, world_size = world_size, 
                 lock = lock, dynamic_list = shared_list)
    
    T.build(sparsity_rate = 0.8,
            optimizer = torch.optim.SGD(T.m.parameters(), 0.1, momentum = 0.9, weight_decay = 1e-3),
            loss = torch.nn.CrossEntropyLoss(reduction = "sum").to('cuda'),
            collective_transforms = (resize, normalize), train_transforms = (dataAug,),
            eval_transforms = (center_crop,), final_collective_transforms = tuple(),
            scale_loss = True, gradient_clipnorm = 2.0, experiment_args = args[1:])

    #T.summary(32)

    del model

    torch.cuda.empty_cache()
    gc.collect()

    dt, dv = get_loaders(rank, world_size, batch_size = 128) 

    logs = T.fit(dt, dv, EPOCHS, CARDINALITY, name, save = False, verbose = False)

    T.evaluate(dt)

    if (rank == 0): 
        print("Train Results: ", T.metric_results())

    T.evaluate(dv)

    if (rank == 0):

        print("Validation Results: ", T.metric_results())
        print("Sparsity: ", T.mm.sparsity)

        logs_to_pickle(logs, name)

        logs_to_pickle(T.fitnesses, name, suffix = "fitnesses")

        #logs_to_pickle(T.prunes, name, suffix = "prunes")

        with open(f"./logs/PICKLES/{name}_prunes.json", "w") as f:
            json.dump(T.prunes, f, indent = 6)

        plot_logs(logs, EPOCHS, name, steps = CARDINALITY)

    torch.distributed.barrier(device_ids = [rank])