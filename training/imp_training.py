import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from data import cifar10, cifar100, imagenet, tiny_imagenet

from utils.serialization_utils import logs_to_pickle

from training.base import BaseIMP
from training.hparams import build_experiment_hparams
from models.vgg import vgg
from models.resnet import resnet

import json
import time
import pickle
import gc
import os
import h5py
from functools import lru_cache

IS_ORCA = False
SAVE_DIRECTORY = "/scratch/tarora_pdx-imagenet/save" if IS_ORCA else "/stash/tlab/tanaya/save"


@lru_cache(maxsize=None)
def _cached_hparams(model: str, dataset: str, time: str):
    return build_experiment_hparams(model, dataset, time=time, pipeline="imp")


def hparams(args):
    return _cached_hparams(args.model, args.dataset, getattr(args, "time", "rewind"))


def momentum(args):
    return hparams(args).momentum

def weight_decay(args):
    return hparams(args).weight_decay

def prune_rate(args):
    return hparams(args).prune_rate

def total_epochs(args):
    return hparams(args).total_epochs

def batchsize(args):
    return hparams(args).batch_size

def datasize(args):
    return hparams(args).train_size

def cardinality(args):
    return hparams(args).cardinality

def learning_rate(args):
    return hparams(args).learning_rate

def start_epochs(args):
    return hparams(args).start_epoch

def rewind_iteration(args):
    return int(start_epochs(args) * cardinality(args))

def warmup_eps(args):
    return hparams(args).warmup_epochs

def reduce_eps(args):
    return hparams(args).lr_milestones

def ddp_network(args):
    hp = hparams(args)

    kwargs = {"outfeatures": hp.num_classes,
              "rank": args.rank,
              "world_size": args.world_size,
              "custom_init": True}

    model = None
    if args.model == "vgg16": model = vgg(depth = hp.model_depth, **kwargs)
    if args.model == "resnet20": model = resnet(depth = hp.model_depth, **kwargs)
    if args.model == "resnet50": model = resnet(depth = hp.model_depth, **kwargs)

    model = model.cuda()

    if args.world_size > 1:
        model = DDP(model, 
            device_ids = [args.rank],
            output_device = args.rank, 
            gradient_as_bucket_view = True)

    return model


def _make_trainer(args, state = None, ticket = None):

    model = ddp_network(args)

    if state is not None: model.load_state_dict(state)
    model_to_inspect = model.module if args.world_size > 1 else model
    if ticket is not None: model_to_inspect.set_ticket(ticket)

    if (args.rank == 0):
        print(f"Training with sparsity {(model_to_inspect.sparsity.item()):.3e}% \n")

    return BaseIMP(model, args.rank, args.world_size, warmup_epochs = warmup_eps(args), reduce_epochs = reduce_eps(args))

def main(rank, world_size, name: str, args, **kwargs):

    args.rank = rank
    args.world_size = world_size

    if rank == 0:
        with open(f"{SAVE_DIRECTORY}/{name}_status.txt", "w") as f:
            f.write("Running.")

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

    data_path = {"cifar10": cifar10, "cifar100": cifar100, "imagenet": imagenet, "tiny-imagenet": tiny_imagenet}[args.dataset]

    dataAug = torch.jit.script(data_path.DataAugmentation().to('cuda'))
    normalize = torch.jit.script(data_path.Normalize().to('cuda'))
    center_crop = torch.jit.script(data_path.CenterCrop().to('cuda'))

    dt, dv = data_path.get_loaders(args.rank, args.world_size, batch_size = batchsize(args))

    old_name = name

    start_time = time.time()

    T = _make_trainer(args)

    T.build(optimizer = torch.optim.SGD, optimizer_kwargs = {'lr': learning_rate(args), 'momentum': momentum(args), 'weight_decay': weight_decay(args)},
            loss = torch.nn.CrossEntropyLoss(reduction = "sum").to('cuda'),
            collective_transforms = tuple(), train_transforms = (dataAug,),
            eval_transforms = (center_crop,), final_collective_transforms = (normalize, ),
            scale_loss = True, gradient_clipnorm = 2.0)

    time_prep = time.time() - start_time

    output = T.TicketIMP(
        dt, dv, total_epochs(args), cardinality(args),
        old_name, prune_rate(args), args.prune_iterations,
        rewind_iter=rewind_iteration(args), validate=False,
        job_data=resume_data,
        partitioned_jobs=getattr(args, "partitioned_jobs", False)
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
