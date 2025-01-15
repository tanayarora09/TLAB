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

def main(rank, world_size, oname: str, argstr: str, lock, shared_list, **kwargs):

    CARDINALITY = 391
    SPARSITY_RATE = 0.8

    dataAug = torch.jit.script(DataAugmentation().to('cuda'))
    resize = torch.jit.script(Resize().to('cuda'))
    normalize = torch.jit.script(Normalize().to('cuda'))
    center_crop = torch.jit.script(CenterCrop().to('cuda'))

    if argstr is None: args = [90, 120, 3 * CARDINALITY, 6, SPARSITY_RATE**18, 0, 160]
    else: args = [float(item) for item in argstr.split(",")] # Iterations, Size, Spacing, PlateauStopEps, Sparsity, Strength Percentage, Epochs
    for idx, item in enumerate(args):
        if item.is_integer():  args[idx] = int(item)

    EPOCHS = args[-1]

    name = oname + "_" + argstr

    model = VGG(depth = 19, rank = rank)

    model = DDP(model.to('cuda'), 
                device_ids = [rank],
                output_device = rank, 
                gradient_as_bucket_view = True,
                broadcast_buffers = False)

    
    T = VGG_DGTS(model = model, rank = rank, world_size = world_size, 
                 lock = lock, dynamic_list = shared_list)
    
    T.build(sparsity_rate = SPARSITY_RATE,
            optimizer = torch.optim.SGD, optim_kwargs = {"lr": 0.1, "momentum" : 0.9, "weight_decay" : 1e-3},
            loss = torch.nn.CrossEntropyLoss(reduction = "sum").to('cuda'),
            collective_transforms = (resize, normalize), train_transforms = (dataAug,),
            eval_transforms = (center_crop,), final_collective_transforms = tuple(),
            scale_loss = True, gradient_clipnorm = 2.0)
    
    T.build_experiment(args)

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

        with open(f"./logs/PICKLES/{name}_prunes.json", "w") as f:
            json.dump(T.prunes, f, indent = 6)

        plot_logs(logs, EPOCHS, name, steps = CARDINALITY)

    torch.distributed.barrier(device_ids = [rank])