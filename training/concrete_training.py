import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from data import cifar10, cifar100, imagenet, tiny_imagenet

from utils.serialization_utils import logs_to_pickle, save_tensor
from utils.training_utils import plot_logs
from utils.search_utils import plot_logs_concrete


from training.base import BaseCNNTrainer

from models.vgg import vgg
from models.resnet import resnet
from training.hparams import build_experiment_hparams

from search.salient import *
from search.concrete import * 
from search.genetic import *

import json
import pickle
import gc
import time
from functools import lru_cache
 
CONCRETE_EXPERIMENTS = {"loss": SNIPConcrete,
                        "gradnorm": GraSPConcrete,
                        "kldlogit": KldLogit,
                        "msefeature": NormalizedMseFeatures,
                        "gradmatch": StepAlignmentConcrete,
                        "deltaloss": LossChangeConcrete}

def momentum(args):
    return hparams(args).train.momentum


@lru_cache(maxsize=None)
def _cached_hparams(model: str, dataset: str, time: str):
    return build_experiment_hparams(model, dataset, time=time, pipeline="concrete")


def hparams(args):
    return _cached_hparams(args.model, args.dataset, getattr(args, "time", "rewind"))


def weight_decay(args):
    return hparams(args).train.weight_decay

def total_epochs(args):
    return hparams(args).train.total_epochs

def batchsize(args):
    return hparams(args).train.batch_size

def datasize(args):
    return hparams(args).data.train_size

def cardinality(args):
    return hparams(args).cardinality

def learning_rate(args):
    return hparams(args).train.learning_rate

def prune_rate(args):
    return hparams(args).search.prune_rate

def start_epochs(args):
    return hparams(args).train.start_epoch

def warmup_eps(args):
    return hparams(args).train.warmup_epochs

def reduce_eps(args):
    return hparams(args).train.lr_milestones

def concrete_epochs(args):
    concrete_epoch_ratio = {"short": 0.125, "half": 0.5, "long": 1.0}[args.duration]
    return int(concrete_epoch_ratio * total_epochs(args))

def ddp_network(args):
    hp = hparams(args)

    kwargs = {"outfeatures": hp.data.num_classes,
              "rank": args.rank,
              "world_size": args.world_size,
              "custom_init": True}

    model = None
    if args.model == "vgg16": model = vgg(depth = hp.model.depth, **kwargs)
    if args.model == "resnet20": model = resnet(depth = hp.model.depth, **kwargs)
    if args.model == "resnet50": model = resnet(depth = hp.model.depth, **kwargs)

    model = model.cuda()

    if args.world_size > 1:
        model = DDP(model, 
            device_ids = [args.rank],
            output_device = args.rank, 
            gradient_as_bucket_view = True)

    return model

def run_concrete(name, args,
                 state, spr, 
                 dt, transforms,):
    
    if spr == 1.0: return state, None

    model = ddp_network(args)
    if state is not None: model.load_state_dict(state)
    state = model.state_dict()

    inp_args = {"rank": args.rank, "world_size": args.world_size, "model": model,}

    if args.criteria == "msefeature":
        captures = []
        fcaptures = []
        model_to_inspect = model.module if args.world_size > 1 else model
        for bname, block in model_to_inspect.named_children():
            for lname, layer in block.named_children():
                if lname.endswith("relu"): captures.append(layer)
                elif lname.endswith("fc"): fcaptures.append((layer, lambda x: torch.softmax(x, dim = 1)))
        inp_args.update({"capture_layers": captures, "fake_capture_layers": fcaptures})

    search = CONCRETE_EXPERIMENTS[args.criteria](**inp_args)

    search.build(spr, torch.optim.Adam, optimizer_kwargs = {'lr': learning_rate(args)}, transforms = transforms, gradbalance = (args.gradstep != "lagrange"))

    cepochs = concrete_epochs(args)
    logs, ticket = search.optimize_mask(dt, cepochs, cardinality(args), dynamic_epochs = False, reduce_epochs = [int(0.88 * cepochs)])
    
    search.finish()

    if args.rank == 0: plot_logs_concrete(logs, name = name)

    return state, ticket


def _make_trainer(args, state = None, ticket = None):

    model = ddp_network(args)

    if state is not None: model.load_state_dict(state)
    model_to_inspect = model.module if args.world_size > 1 else model
    if ticket is not None: model_to_inspect.set_ticket(ticket)

    if (args.rank == 0):
        print(f"Training with sparsity {(model_to_inspect.sparsity.item()):.3e}% \n")

    return BaseCNNTrainer(model, args.rank, args.world_size, warmup_epochs = warmup_eps(args), reduce_epochs = reduce_eps(args))



def run_start_train(name, args, 
                    dt, dv, transforms,): 

    T = _make_trainer(args)

    T.build(optimizer = torch.optim.SGD, optimizer_kwargs = {'lr': learning_rate(args), 'momentum': momentum(args), 'weight_decay': weight_decay(args)},
            loss = torch.nn.CrossEntropyLoss(reduction = "sum").to('cuda'),
            collective_transforms = tuple(), train_transforms = (transforms[0],),
            eval_transforms = (transforms[1],), final_collective_transforms = (transforms[2], ),
            scale_loss = True, gradient_clipnorm = 2.0)

    T.fit(dt, dv, start_epochs(args), cardinality(args), name + "_pretrain", save = False, save_init = False, validate = False)

    T.evaluate(dt)

    if (args.rank == 0): 
        train_res = T.metric_results()
        print("Train Results: ", train_res)

    T.evaluate(dv)
    
    if (args.rank == 0):
        val_res =  T.metric_results()
        print("Validation Results: ", val_res) 

    return T.m.state_dict(), T.optim.state_dict()


def run_fit_and_export(name, old_name,
                       args, state, ticket, 
                       spr, dt, dv, transforms,): #Transforms: DataAug, CenterCrop, Normalize

    T = _make_trainer(args, state, ticket)

    T.mm.export_ticket(old_name, entry_name = f"{spr * 100:.3e}", root = 0)

    T.build(optimizer = torch.optim.SGD, optimizer_kwargs = {'lr': learning_rate(args), 'momentum': momentum(args), 'weight_decay': weight_decay(args)},
            loss = torch.nn.CrossEntropyLoss(reduction = "sum").to('cuda'),
            collective_transforms = tuple(), train_transforms = (transforms[0],),
            eval_transforms = (transforms[1],), final_collective_transforms = (transforms[2], ),
            scale_loss = True, gradient_clipnorm = 2.0)

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

    logs = T.fit(dt, dv, total_epochs(args), cardinality(args), name, validate = True, save = False, save_init = False, start = start_epochs(args))

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

        plot_logs(logs, total_epochs(args), name, cardinality(args), start = start_epochs(args))

def main(rank, world_size, name: str, args, **kwargs):

    args.rank = rank
    args.world_size = world_size

    if rank == 0: print(f"Model: {args.model}\nGradient Step: {args.gradstep}\nTime: {args.time}\nType: {args.criteria}\nDuration: {args.duration}")

    sparsity_range = {"vgg16": [prune_rate(args) ** x for x in range(2, 43, 2)],
                      "resnet20": [prune_rate(args) ** x for x in range(2, 43, 2)],
                      "resnet50": [prune_rate(args) ** x for x in range(1, 7)]}

    sps =  sparsity_range[args.model] if args.sparsities is None else args.sparsities

    name = f"{args.criteria}_{args.model}_{args.dataset}_{args.time}_{args.duration}_{name}"

    DISTRIBUTED = world_size > 1

    old_name = name

    data_path = {"cifar10": cifar10, "cifar100": cifar100, "imagenet": imagenet, "tiny-imagenet": tiny_imagenet}[args.dataset]

    dataAug = torch.jit.script(data_path.DataAugmentation().to('cuda'))
    normalize = torch.jit.script(data_path.Normalize().to('cuda'))
    center_crop = torch.jit.script(data_path.CenterCrop().to('cuda'))

    dt, dv = data_path.get_loaders(args.rank, args.world_size, batch_size = batchsize(args))

    start_start = time.time()

    if args.time == "rewind": 
        ostate, _ = run_start_train(name = name, dt = dt, dv = dv, 
                                    transforms = (dataAug, center_crop, normalize,),
                                    args = args)
    else: 
        ostate = None

    start_end = time.time()
    start_train_time = (start_end - start_start)

    for spr in sps:

        prune_train_start = time.time()

        name = old_name + f"_{spr * 100:.3e}"

        state, ticket = run_concrete(name = name, args = args, 
                                     state = ostate, spr = spr, dt = dt, 
                                     transforms = (center_crop, normalize,),)

        torch.cuda.empty_cache()
        gc.collect()
        
        run_fit_and_export(name = name, old_name = old_name, args = args, 
                           state = state, ticket = ticket, spr = spr, dt = dt, dv = dv, 
                           transforms = (dataAug, center_crop, normalize,))
        
        torch.cuda.empty_cache()
        gc.collect()

        prune_train_end = time.time()

        total_time = (prune_train_end - prune_train_start) + start_train_time
        with open(f"logs/TIMES/{name}.txt", "w") as f:
            f.write(f"{total_time:.2f}")

        if DISTRIBUTED: torch.distributed.barrier(device_ids = [rank])
