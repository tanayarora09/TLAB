import torch
from data.index import get_data_handle

from utils.serialization_utils import logs_to_pickle
from utils.training_utils import plot_logs
from utils.search_utils import plot_logs_concrete


from training.index import get_trainer, build_trainer, get_training_hparams

from models.index import get_model

from search.concrete import get_concrete

import json
import gc
import time


DEFAULT_CONCRETE_SPARSITY_RANGE = [0.01]


def concrete_epochs(args, training_Hparams):
    concrete_epoch_ratio = {"short": 0.125, "half": 0.5, "long": 1.0}[args.duration]
    return int(concrete_epoch_ratio * training_Hparams.epochs)

def _make_trainer(args, data_handle, state = None, ticket = None):

    model = get_model(args, state, ticket)

    trainer = get_trainer(args, model, 'cnn')

    build_trainer(trainer, args, data_handle)

    return trainer

def _update_mse_feature(args, model, inp_args):

    if args.criteria != "msefeature": return

    captures = []
    fcaptures = []
    model_to_inspect = model.module if args.world_size > 1 else model
    for bname, block in model_to_inspect.named_children():
        for lname, layer in block.named_children():
            if lname.endswith("relu"): captures.append(layer)
            elif lname.endswith("fc"): fcaptures.append((layer, lambda x: torch.softmax(x, dim = 1)))
    inp_args.update({"capture_layers": captures, "fake_capture_layers": fcaptures})


def run_concrete(name, args,
                 state, spr, 
                 dt, data_handle,
                 training_Hparams):
    
    if spr == 1.0: return state, None

    model = get_model(args, state)
    state = model.state_dict()

    inp_args = {"rank": args.rank, "world_size": args.world_size, "model": model,}

    _update_mse_feature(args, model, inp_args)
    search = get_concrete(args.criteria)(**inp_args)

    search.build(spr, torch.optim.Adam, optimizer_kwargs = {'lr': training_Hparams.learning_rate}, data_handle = data_handle, gradbalance = (args.gradstep != "lagrange"))

    cepochs = concrete_epochs(args, training_Hparams)

    logs, ticket = search.optimize_mask(dt, cepochs, data_handle.cardinality(), reduce_epochs = [int(0.88 * cepochs)])
    
    search.finish()

    if args.rank == 0: plot_logs_concrete(logs, name = name)

    return state, ticket


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
                       training_Hparams): #Transforms via DataHandle

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
                 validate = True, save = args.save, save_init = args.save_init, 
                 start = training_Hparams.rewind_epoch)

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

    if rank == 0: print(f"Model: {args.model}\nGradient Step: {args.gradstep}\nTime: {args.time}\nType: {args.criteria}\nDuration: {args.duration}")

    sps = DEFAULT_CONCRETE_SPARSITY_RANGE if args.sparsities is None else args.sparsities

    name = f"{args.criteria}_{args.model}_{args.dataset}_{args.time}_{args.duration}_{name}"

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

        name = old_name + f"_{spr * 100:.3e}"

        state, ticket = run_concrete(name = name, args = args, 
                                     state = ostate, spr = spr, dt = dt,  
                                    handle = handle,
                                    training_Hparams = tParams)

        torch.cuda.empty_cache()
        gc.collect()
        
        run_fit_and_export(name = name, old_name = old_name, args = args, 
                           state = state, ticket = ticket, spr = spr, dt = dt, dv = dv, 
                           handle = handle,
                           training_Hparams = tParams)
        
        torch.cuda.empty_cache()
        gc.collect()

        prune_train_end = time.time()

        total_time = (prune_train_end - prune_train_start) + start_train_time
        with open(f"logs/TIMES/{name}.txt", "w") as f:
            f.write(f"{total_time:.2f}")

        if world_size > 1: torch.distributed.barrier(device_ids = [rank])