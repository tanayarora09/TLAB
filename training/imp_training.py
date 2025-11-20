import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from data import cifar10, cifar100, imagenet

from utils.serialization_utils import logs_to_pickle

from training.VGG import VGG_IMP
from training.ResNet import ResNet_IMP, ResNet50_IMP
from models.vgg import vgg
from models.resnet import resnet

import json
import time
import pickle
import gc
import os
import h5py
import math

IS_ORCA = False
SAVE_DIRECTORY = "/scratch/tarora_pdx-imagenet/save" if IS_ORCA else "/stash/tlab/tanaya/save"

def momentum(args):
    return 0.9

def weight_decay(args):
    return 1e-4 if args.model == "resnet50" else 1e-3

def prune_rate(args):
    return {"vgg16": 0.8, "resnet20": 0.8, "resnet50": 0.31622776601}[args.model]

def total_epochs(args):
    return {"resnet20": 160, "vgg16": 160, "resnet50": 90}[args.model]

def batchsize(args):
    return 1024 if args.dataset == "imagenet" else 512

def datasize(args):
    return 1281167 if args.dataset == "imagenet" else 50000

def cardinality(args):
    return math.ceil(datasize(args)/batchsize(args))

def learning_rate(args):
    return {"resnet20": 1e-1, "vgg16": 1e-1, "resnet50": 4e-1}[args.model]

def start_epochs(args):
    start_epochs = {"resnet20": 9, "vgg16": 15, "resnet50": 18}[args.model]
    if args.time == "init": start_epochs = 0
    return start_epochs

def rewind_iteration(args):
    return int(start_epochs(args) * cardinality(args))

def ddp_network(args):
    
    classes = {"cifar10": 10,
               "cifar100": 100,
               "imagenet": 1000}[args.dataset]
    
    kwargs = {"outfeatures": classes,
              "rank": args.rank,
              "world_size": args.world_size,
              "custom_init": True}

    model = None
    if args.model == "vgg16": model = vgg(depth = 16, **kwargs)
    if args.model == "resnet20": model = resnet(depth = 20, **kwargs)
    if args.model == "resnet50": model = resnet(depth = 50, **kwargs)

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

    return {"vgg16": VGG_IMP, "resnet20": ResNet_IMP, "resnet50": ResNet50_IMP}[args.model](model, args.rank, args.world_size)

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

    data_path = {"cifar10": cifar10, "cifar100": cifar100, "imagenet": imagenet}[args.dataset]

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
