import torch

from data.index import get_data_handle

from training.hparams import get_training_hparams
from utils.serialization_utils import logs_to_pickle

from training.index import get_trainer, build_trainer, get_training_hparams
from models.index import get_model

import json
import copy
import h5py

DEFAULT_NAIVE_SPARSITY_RANGE = [1.00]

def main(rank, world_size, name: str, args, **kwargs):

    args.rank = rank
    args.world_size = world_size    

    tParams = get_training_hparams(args)

    name = f"{args.type}_{args.model}_{args.dataset}_{name}"
    old_name = name

    if rank == 0: h5py.File(f"./logs/TICKETS/{old_name}.h5", "w").close()

    data_handle = get_data_handle(args)
    data_handle.load_transforms(device = 'cuda')
    dt, dv = data_handle.get_loaders(rank, world_size) 

    sps = DEFAULT_NAIVE_SPARSITY_RANGE if args.sparsities is None else args.sparsities

    for spr in sps:

        if args.type != "none": name = old_name + f"_{spr * 100:.3e}"

        model = get_model(args)

        model_to_inspect = model.module if world_size > 1 else model
        if args.type == "random": model_to_inspect.prune_random(spr, distributed = True)
        elif args.type == "magnitude": model_to_inspect.prune_by_mg(spr, 1, root = 0)
        elif args.type != "none": raise ValueError(f"Unknown pruning type {args.type}")

        T = get_trainer(args, model, 'cnn')
        build_trainer(T, args, data_handle)

        logs = T.fit(dt, dv, tParams.epochs, data_handle.cardinality(), 
                     name, save = args.save, save_init = args.save_init, 
                     verbose = False, validate = False)

        if rank == 0: logs_to_pickle(logs, name)
           
        T.evaluate(dt)

        if (rank == 0): 
            train_res = T.metric_results()
            print("Train Results: ", train_res)

        T.evaluate(dv)
        
        if (rank == 0):
            val_res =  T.metric_results()
            print("Validation Results: ", val_res)
            print("Sparsity: ", T.mm.sparsity)
            train_res.update({('val_' + k): v for k, v in val_res.items()})
        
            with open(f"./logs/RESULTS/{old_name}_{spr*100:.3e}.json", "w") as f:
                json.dump(train_res, f, indent = 6)
        
        torch.distributed.barrier(device_ids = [rank])