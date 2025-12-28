import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from data.index import get_data_handle

from utils.serialization_utils import logs_to_pickle

from training.index import get_training_hparams, get_trainer, build_trainer
from models.index import get_model

import json
import time
import pickle
import os
import math

IS_ORCA = False
SAVE_DIRECTORY = "/scratch/tarora_pdx-imagenet/save" if IS_ORCA else "/stash/tlab/tanaya/save"


def prune_rate(args):
    return {"vgg16": 0.8, "resnet20": 0.8, "resnet50": 0.46415888336}[args.model]

def rewind_iteration(training_Hparams, data_handle):
    return int(training_Hparams.rewind_epoch * data_handle.cardinality())

def _make_trainer(args, data_handle, state = None, ticket = None):

    model = get_model(args, state, ticket)

    trainer = get_trainer(args, model, 'imp')

    build_trainer(trainer, args, data_handle)

    return trainer

def main(rank, world_size, name: str, args, **kwargs):

    args.rank = rank
    args.world_size = world_size

    if rank == 0:
        with open(f"{SAVE_DIRECTORY}/{name}_status.txt", "w") as f: f.write("Running.")

    resume_file = f"{SAVE_DIRECTORY}/{name}_resume.pkl"
    resume_data = dict()
    if os.path.exists(resume_file):
        if rank == 0:
            with open(resume_file, "rb") as f:
                resume_data = [pickle.load(f)]
        else:
            resume_data = [dict()]
        torch.distributed.broadcast_object_list(resume_data, src = 0)
        resume_data = resume_data[0]
    resume_data["start_iter"] = args.current_iteration

    data_handle = get_data_handle(args) 
    data_handle.load_transforms(device = 'cuda')

    dt, dv = data_handle.get_loaders(args.rank, args.world_size)

    tParams = get_training_hparams(args)

    old_name = name

    start_time = time.time()

    T = _make_trainer(args, data_handle, tParams)

    time_prep = time.time() - start_time

    output = T.TicketIMP(
        dt, dv, tParams.epochs, data_handle.cardinality(),
        old_name, prune_rate(args), args.prune_iterations,
        rewind_iter=rewind_iteration(tParams, data_handle), 
        validate=False,
        job_data=resume_data,
        partitioned_jobs=args.partitioned_jobs
        )

    if args.partitioned_jobs and (args.current_iteration < args.prune_iterations):

        total_logs, results, sparsities_d = output
        
        if rank == 0:
            with open(resume_file, "wb") as f:
                pickle.dump({"total_logs": total_logs, "results": results, "sparsities_d": sparsities_d}, 
                            f, protocol = pickle.HIGHEST_PROTOCOL)

            with open(f"{SAVE_DIRECTORY}/{name}_status.txt", "w") as f:
                f.write("Waiting.")

        return

    elif rank == 0:

        (logs, results), sparsities_d = output

        for spe in list(range(len(results))):
            with open(f"./logs/RESULTS/{old_name}_{spe}.json", "w") as f:
                json.dump(results[spe][0], f, indent = 6)
            with open(f"./logs/TIMES/{old_name}_{spe}.txt", "w") as f:
                f.write(f"{time_prep + results[spe][1]:.2f}")
        logs_to_pickle(logs, name)

        with open(f"{SAVE_DIRECTORY}/{name}_status.txt", "w") as f:
            f.write("Done.")

        


    torch.distributed.barrier(device_ids = [rank])
