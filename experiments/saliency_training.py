import torch

from data.index import get_data_handle

from utils.serialization_utils import logs_to_pickle
from utils.training_utils import plot_logs

from training.index import get_trainer, build_trainer, get_training_hparams
from models.index import get_model

from search.salient import get_salient

import json
import gc
import copy
import time

DEFAULT_SALIENCY_SPARSITY_RANGE = [0.01]

def micro_batchsize(args):
    return {"cifar10": 256, "cifar100": 256, "imagenet": 64}[args.dataset] # based on performance 

def _update_mse_feature(args, model, inp_args):
    
    if args.criteria != "msefeature": return

    captures = []
    fcaptures = []
    for bname, block in model.named_children():
        for lname, layer in block.named_children():
            if lname.endswith("relu"): captures.append(layer)
            elif lname.endswith("fc"): fcaptures.append((layer, lambda x: torch.softmax(x, dim = 1)))
    inp_args.update({"capture_layers": captures, "fake_capture_layers": fcaptures})

def _run_parallel_salient(model, data_handle, args):
    

def _run_sp_salient(model, spr, data_handle, args):

    model_to_inspect = model.module if args.world_size > 1 else model

    if args.rank == 0:
        inp_args = {"rank": 0, "world_size": 1, "model": model_to_inspect}
        _update_mse_feature(args, model_to_inspect, inp_args)

        pruner = (get_salient(args.criteria))(**inp_args)
        pruner.build(spr, data_handle, input = None)
        ticket = pruner.grad_mask(steps = args.steps, micro_batch_size = micro_batchsize(args))
        pruner.finish()

    else:
        ticket = torch.zeros(model_to_inspect.num_prunable, dtype = torch.bool, device = 'cuda')

    if args.world_size > 1:
        torch.distributed.barrier(device_ids = [args.rank])
        torch.distributed.broadcast(ticket, src = 0)


def run_salient(name, args, spr, data_handle, state = None): 
    
    args_copy = copy.copy(args)
    args_copy.bn_track = (args.criteria in ["synflow", "gradmatch"])

    model = get_model(args_copy, state)
    state = model.state_dict()
    
    if args.dataset == "imagenet": ticket = _run_parallel_salient(model, data_handle, args)

    else: ticket = _run_sp_salient(model, data_handle, args)

    return state, ticket

def _make_trainer(args, data_handle, state = None, ticket = None):

    model = get_model(args, state, ticket)

    trainer = get_trainer(args, model, 'cnn')

    build_trainer(trainer, args, data_handle)

    return trainer

def run_start_train(name, args, 
                    dt, dv, 
                    data_handle,
                    training_Hparams): 

    T = _make_trainer(args, data_handle)

    T.fit(dt, dv, training_Hparams.rewind_epoch, data_handle.cardinality(), 
          name + "_pretrain", save = False, save_init = False, validate = False)

    T.evaluate(dt)

    if (args.rank == 0): print("Train Results: ", T.metric_results())

    T.evaluate(dv)
    
    if (args.rank == 0): print("Validation Results: ", T.metric_results()) 

    return T.m.state_dict()

def run_fit_and_export(name, old_name,
                       args, state, ticket, 
                       spr, dt, dv, 
                       data_handle,
                       training_Hparams): 

    T = _make_trainer(args, data_handle, state, ticket)

    T.mm.export_ticket(old_name, entry_name = f"{spr * 100:.3e}", root = 0)

    T.evaluate(dt)

    if (args.rank == 0): 
        orig_train_res = T.metric_results()
        print("Train Results: ", orig_train_res)

    T.evaluate(dv)
    
    if (args.rank == 0):
        orig_val_res =  T.metric_results()
        print("Validation Results: ", orig_val_res)
        print("Sparsity: ", T.mm.sparsity)
        orig_train_res.update({('val_' + k): v for k, v in orig_val_res.items()})

    logs = T.fit(dt, dv, training_Hparams.epochs, data_handle.cardinality(), name, 
                 validate = True, save = False, save_init = False, start = training_Hparams.rewind_epoch)

    T.evaluate(dt)

    if (args.rank == 0): 
        train_res = T.metric_results()
        print("Train Results: ", train_res)

    T.evaluate(dv)
    
    if (args.rank == 0):
        val_res =  T.metric_results()
        print("Validation Results: ", val_res)
        print("Sparsity: ", T.mm.sparsity)
        train_res.update({('val_' + k): v for k, v in val_res.items()})
    
        with open(f"./logs/RESULTS/{old_name}_{spr*100:.3e}.json", "w") as f:
            json.dump({"original": orig_train_res, "finetuned": train_res}, f, indent = 6)
        
        logs_to_pickle(logs, name)

        plot_logs(logs, training_Hparams.epochs, name, data_handle.cardinality(), start = training_Hparams.rewind_epoch)

def main(rank, world_size, name: str, args, **kwargs):

    args.rank = rank
    args.world_size = world_size

    if rank == 0: print(f"Model: {args.model}\nTime: {args.time}\nType: {args.criteria}\nSteps: {args.steps}")

    sps = DEFAULT_SALIENCY_SPARSITY_RANGE if args.sparsities is None else args.sparsities

    name = f"{args.criteria}_{args.model}_{args.dataset}_{args.time}_{args.steps}_{name}"

    old_name = name

    handle = get_data_handle(args) #updates potential batch_size
    handle.load_transforms(device = 'cuda')

    dt, dv = handle.get_loaders(rank, world_size)

    tParams = get_training_hparams(args)

    start_start = time.time()

    if args.when_to_prune == "rewind": 
        ostate = run_start_train(name = name, args = args,
                                 dt = dt, dv = dv, 
                                handle = handle,
                                training_Hparams = tParams)
    else: 
        ostate = None

    start_end = time.time()
    start_train_time = (start_end - start_start)

    for spr in sps:

        prune_train_start = time.time()

        name = old_name + f"_{spr:03e}"

        state, ticket = run_salient(name = name, args = args, 
                                     state = ostate, spr = spr,
                                     data_handle = handle,)

        torch.cuda.empty_cache()
        gc.collect()
        
        run_fit_and_export(name = name, old_name = old_name, args = args, 
                           state = state, ticket = ticket, spr = spr, dt = dt, dv = dv, 
                           data_handle = handle, training_Hparams = tParams)

        
        torch.cuda.empty_cache()
        gc.collect()

        prune_train_end = time.time()

        total_time = (prune_train_end - prune_train_start) + start_train_time
        with open(f"logs/TIMES/{name}.txt", "w") as f:
            f.write(f"{total_time:.2f}")

        if world_size > 1: torch.distributed.barrier(device_ids = [rank])
