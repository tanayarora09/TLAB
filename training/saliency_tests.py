import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from data.cifar10 import *

from utils.serialization_utils import logs_to_pickle, save_tensor
from utils.training_utils import plot_logs

from training.VGG import VGG_CNN
from models.VGG import VGG

from search.salient import MSE_Pruner

import json
import pickle
import gc

EPOCHS = 160
CARDINALITY = 98

def ddp_network(rank, world_size, depth = 16):

    model = VGG(depth = depth, rank = rank, world_size = world_size, custom_init = True)
    model = DDP(model.to('cuda'), 
                device_ids = [rank],
                output_device = rank, 
                gradient_as_bucket_view = True)
    
    return model

def run_saliency(rank, world_size, name, old_name, spe, spr, transforms, ckpt = None): #ONLY ON ROOT

    model = ddp_network(rank, world_size, 16)

    if ckpt is not None: model.load_state_dict(ckpt)
    #ckpt = torch.load("", )
    #model.load_state_dict(ckpt)
    
    state = model.state_dict()
    
    captures = list()
    for _, block in model.module.named_children():
        for n, layer in block.named_children():
            if n.endswith("relu"): captures.append(layer)

    
    if rank == 0:
        pruner = MSE_Pruner(0, 1, model.module, captures, [])
        #partial = get_partial_train_loader(rank, world_size, 10)
        pruner.build(spr, transforms, input = None)#partial)
        ticket = pruner.grad_mask()
        pruner.finish()

    else:
        ticket = torch.zeros(model.module.num_prunable, dtype = torch.bool, device = 'cuda')

    torch.distributed.barrier(device_ids = [rank])
    torch.distributed.broadcast(ticket, src = 0)

    return state, ticket

def _make_trainer(rank, world_size, state = None, ticket = None):

    model = ddp_network(rank, world_size, 16)
    if state is not None: model.load_state_dict(state)
    if ticket is not None: model.module.set_ticket(ticket)
    
    if (rank == 0):
        print(model.module.sparsity, "\n")

    return VGG_CNN(model, rank, world_size)

def pretrain_dict(rank, world_size, name, transforms):

    T = _make_trainer(rank, world_size)

    T.build(optimizer = torch.optim.SGD, optimizer_kwargs = {'lr': 0.1, 'momentum': 0.9, 'weight_decay': 1e-3},
            loss = torch.nn.CrossEntropyLoss(reduction = "sum").to('cuda'),
            collective_transforms = (transforms[1], transforms[2]), train_transforms = (transforms[0],),
            eval_transforms = (transforms[3],), final_collective_transforms = tuple(),
            scale_loss = True, gradient_clipnorm = 2.0)

    dt, dv = get_loaders(rank, world_size, batch_size = 512)

    T.fit(dt, dv, 5, CARDINALITY, name + "_pretrain", save = False, validate = False)

    return T.m.state_dict()

def run_fit_and_export(rank, world_size, name, old_name, state, ticket, spe, spr, transforms,): #Transforms: DataAug, Resize, Normalize, Crop

    T = _make_trainer(rank, world_size, state, ticket)

    T.build(optimizer = torch.optim.SGD, optimizer_kwargs = {'lr': 0.1, 'momentum': 0.9, 'weight_decay': 1e-3},
            loss = torch.nn.CrossEntropyLoss(reduction = "sum").to('cuda'),
            collective_transforms = (transforms[1], transforms[2]), train_transforms = (transforms[0],),
            eval_transforms = (transforms[3],), final_collective_transforms = tuple(),
            scale_loss = True, gradient_clipnorm = 2.0)
    
    dt, dv = get_loaders(rank, world_size, batch_size = 512) 

    logs = T.fit(dt, dv, EPOCHS, CARDINALITY, name, validate = False, save = False, start = 0)

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
    
        with open(f"./logs/RESULTS/{old_name}_{spe}.json", "w") as f:
            json.dump(train_res, f, indent = 6)
        
        logs_to_pickle(logs, name)
        
    T.mm.export_ticket(old_name, entry_name = f"{spr * 100:.3e}", root = 0)

    del dt, dv

def main(rank, world_size, name: str, sp_exp: list, **kwargs):

    old_name = name

    dataAug = torch.jit.script(DataAugmentation().to('cuda'))
    resize = torch.jit.script(Resize().to('cuda'))
    normalize = torch.jit.script(Normalize().to('cuda'))
    center_crop = torch.jit.script(CenterCrop().to('cuda'))

    for spe in reversed(sp_exp):

        spr = 0.8**spe
        name = old_name + f"_{spe:02d}"

        #ckpt = pretrain_dict(rank, world_size, name,
        #                    (dataAug, resize, normalize, center_crop,))

        torch.cuda.empty_cache()
        gc.collect()

        state, ticket = run_saliency(rank, world_size, name, old_name, spe, spr, 
                                   (resize, normalize, center_crop,), ckpt = None)

        torch.cuda.empty_cache()
        gc.collect()
        
        run_fit_and_export(rank, world_size, name, old_name, state, ticket,
                spe, spr, (dataAug, resize, normalize, center_crop,))

        
        torch.cuda.empty_cache()
        gc.collect()

        torch.distributed.barrier()
