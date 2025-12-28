import torch

from data.index import get_data_handle

from training.hparams import get_training_hparams
from utils.serialization_utils import logs_to_pickle

from training.index import get_trainer, build_trainer, get_training_hparams
from models.index import get_model
from models.hparams import return_model_name

import json
import copy
import os

def _state_convert(args, state):
    if args.world_size > 1: return state
    return {k[7:]:v for k,v in state.items()}

def get_parent_state(args):
    """
    (args.parent, args.dataset) in save
    """
    parent_path = lambda name, dataset: f"./logs/PARENTS/{name}_{dataset}.pt"
    
    if not os.path.exists(parent_path(return_model_name(args.parent), args.dataset)):
        raise ValueError(f"No parent state found for model {args.parent} on dataset {args.dataset}")
    
    return torch.load(parent_path(return_model_name(args.parent), args.dataset), 
                      map_location = {'cuda:%d' % 0: 'cuda:%d' % args.rank},
                      weights_only = True)['model']

def main(rank, world_size, name: str, args, **kwargs):

    args.rank = rank
    args.world_size = world_size

    tParams = get_training_hparams(args)

    name = f"{args.parent}_distill_{args.model}_{args.dataset}_{name}"

    data_handle = get_data_handle(args)
    data_handle.load_transforms(device = 'cuda')
    dt, dv = data_handle.get_loaders(rank, world_size) 

    model = get_model(args)

    parent_args = copy.copy(args)
    parent_args.model = args.parent
    parent = get_model(parent_args)
    parent.load_state_dict(_state_convert(args, get_parent_state(args)))
    
    T = get_trainer(args, model, 'cnn')
    
    build_trainer(T, args, data_handle)

    logs = T.distill(dt, dv, tParams.epochs, data_handle.cardinality(), name,
                     parent, args.alpha, args.temperature,
                     save = args.save, save_init = args.save_init, 
                     verbose = False, validate = True)

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
    
        with open(f"./logs/RESULTS/{name}.json", "w") as f:
            json.dump(train_res, f, indent = 6)
    
    torch.distributed.barrier(device_ids = [rank])