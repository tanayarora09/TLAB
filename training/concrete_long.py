import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from data.cifar10 import *

from utils.serialization_utils import logs_to_pickle, save_tensor
from utils.training_utils import plot_logs
from utils.search_utils import plot_logs_concrete


from training.VGG import VGG_CNN
from models.VGG import VGG

from training.ResNet import ResNet_CNN
from models.ResNet import ResNet

from search.salient import *
from search.concrete import * 
from search.genetic import *

import json
import pickle
import gc

EPOCHS = 160
CARDINALITY = 98
START_EPOCHS = 0#10
CONCRETE_EPOCHS = 160#25 


def ddp_network(rank, world_size, is_vgg, depth = 16):

    if not is_vgg: depth = 20
    
    if is_vgg:
        model = VGG(depth = depth, rank = rank, world_size = world_size, custom_init = True) 
    else: 
        model =  ResNet(depth = depth, rank = rank, world_size = world_size, custom_init = True)
    
    model = model.cuda()

    if world_size > 1:
        model = DDP(model, 
            device_ids = [rank],
            output_device = rank, 
            gradient_as_bucket_view = True)

    return model

def run_concrete(rank, world_size, name, is_vgg, state, spe, spr, dt, transforms, tmp_is_step_alignment: bool = False, opt_state = None): #ONLY ON ROOT

    model = ddp_network(rank, world_size, is_vgg, depth = 16)
    if state is not None: model.load_state_dict(state)

    """captures = []
    fcaptures = []
    for bname, block in model.module.named_children():
        for lname, layer in block.named_children():
            if lname.endswith("relu")"""

    search = SNIPConcrete(rank, world_size, model)#KldLogit(rank, world_size, model)
    #if not tmp_is_step_alignment: search = SNIPConcrete(rank, world_size, model)
    #else: search = StepAlignmentConcrete(rank, world_size, model, optimizer_state = opt_state)

    search.build(spr, torch.optim.Adam, optimizer_kwargs = {'lr': 1e-1}, transforms = transforms)

    logs, ticket = search.optimize_mask(dt, CONCRETE_EPOCHS, CARDINALITY, dynamic_epochs = False, reduce_epochs = [120])

    search.finish()

    if rank == 0: plot_logs_concrete(logs, name = name)

    return state, ticket


def _make_trainer(rank, world_size, is_vgg, state = None, ticket = None):

    model = ddp_network(rank, world_size, is_vgg, 16)
    if state is not None: model.load_state_dict(state)#{k[7:]:v for k,v in state.items()} if world_size == 1 else state)
    if ticket is not None and world_size == 1: model.set_ticket(ticket)#.module.set_ticket(ticket)
    elif ticket is not None: model.module.set_ticket(ticket)

    if (rank == 0):
        if world_size == 1: print(f"Training with sparsity {(100 - model.sparsity.item()):.3f}% \n")
        else: print(f"Training with sparsity {(100 - model.module.sparsity.item()):.3f}% \n")


    return VGG_CNN(model, rank, world_size) if is_vgg else ResNet_CNN(model, rank, world_size)

def run_start_train(rank, world_size, name, is_vgg, dt, dv, transforms,): 

    T = _make_trainer(rank, world_size, is_vgg)

    T.build(optimizer = torch.optim.SGD, optimizer_kwargs = {'lr': 0.1, 'momentum': 0.9, 'weight_decay': 1e-3},
            loss = torch.nn.CrossEntropyLoss(reduction = "sum").to('cuda'),
            collective_transforms = (transforms[1], transforms[2]), train_transforms = (transforms[0],),
            eval_transforms = (transforms[3],), final_collective_transforms = tuple(),
            scale_loss = True, gradient_clipnorm = 2.0)

    T.fit(dt, dv, START_EPOCHS, CARDINALITY, name + "_pretrain", save = False, save_init = False, validate = False)

    T.evaluate(dt)

    if (rank == 0): 
        train_res = T.metric_results()
        print("Train Results: ", train_res)

    T.evaluate(dv)
    
    if (rank == 0):
        val_res =  T.metric_results()
        print("Validation Results: ", val_res) 

    return T.m.state_dict(), T.optim.state_dict()

def run_fit_and_export(rank, world_size, name, old_name, state, ticket, is_vgg, spe, spr, dt, dv, transforms,): #Transforms: DataAug, Resize, Normalize, Crop

    T = _make_trainer(rank, world_size, is_vgg, state, ticket)

    T.mm.export_ticket(old_name, entry_name = f"{spr * 100:.2f}", root = 0)

    T.build(optimizer = torch.optim.SGD, optimizer_kwargs = {'lr': 0.1, 'momentum': 0.9, 'weight_decay': 1e-3},
            loss = torch.nn.CrossEntropyLoss(reduction = "sum").to('cuda'),
            collective_transforms = (transforms[1], transforms[2]), train_transforms = (transforms[0],),
            eval_transforms = (transforms[3],), final_collective_transforms = tuple(),
            scale_loss = True, gradient_clipnorm = 2.0)

    T.evaluate(dt)

    if (rank == 0): 
        orig_train_res = T.metric_results()
        print("Train Results: ", orig_train_res)

    T.evaluate(dv)
    
    if (rank == 0):
        orig_val_res =  T.metric_results()
        print("Validation Results: ", orig_val_res)
        print("Sparsity: ", T.mm.sparsity)
        orig_train_res.update({('val_' + k): v for k, v in orig_val_res.items()})

    logs = T.fit(dt, dv, EPOCHS, CARDINALITY, name, validate = True, save = False, save_init = False, start = START_EPOCHS)

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
            json.dump({"results": {"original": orig_train_res, "finetuned": train_res}, "sparsity": T.mm.sparsity.item()}, f, indent = 6)
        
        logs_to_pickle(logs, name)

        plot_logs(logs, EPOCHS, name, CARDINALITY, start = START_EPOCHS)

    del dt, dv

def main(rank, world_size, name: str, sp_exp: list, **kwargs):

    #is_grasp = True
    is_vgg = sp_exp.pop(-1) == 1 # 1 is yes, 0 is no
    if rank == 0: print(f"VGG: {is_vgg}")

    DISTRIBUTED = world_size > 1

    old_name = name

    dataAug = torch.jit.script(DataAugmentation().to('cuda'))
    resize = torch.jit.script(Resize().to('cuda'))
    normalize = torch.jit.script(Normalize().to('cuda'))
    center_crop = torch.jit.script(CenterCrop().to('cuda'))

    dt, dv = get_loaders(rank, world_size, batch_size = 512)

    #state, opt_state = run_start_train(rank, world_size, name, is_vgg, dt, dv,
    #                        (dataAug, resize, normalize, center_crop,))
    state = None

    for spe in reversed(sp_exp):

        spr = 0.8**spe
        name = old_name + f"_{spe:02d}"
        
        """
        _, ticket = run_concrete(rank, world_size, "step_" + name, is_vgg, state, spe,
                                    spr, dt, (resize, normalize, center_crop,),
                                    tmp_is_step_alignment = True, opt_state = opt_state)

        torch.cuda.empty_cache()
        gc.collect()
        
        run_fit_and_export(rank, world_size, "step_" + name, "step_" + old_name, state, ticket,
                 is_vgg, spe, spr, dt, dv, (dataAug, resize, normalize, center_crop,))

        
        torch.cuda.empty_cache()
        gc.collect()

        if DISTRIBUTED: torch.distributed.barrier(device_ids = [rank])
        """

        _, ticket = run_concrete(rank, world_size, name, is_vgg, state, spe,
                                    spr, dt, (resize, normalize, dataAug,))

        torch.cuda.empty_cache()
        gc.collect()
        
        run_fit_and_export(rank, world_size, name, old_name, state, ticket,
                 is_vgg, spe, spr, dt, dv, (dataAug, resize, normalize, center_crop,))

        
        torch.cuda.empty_cache()
        gc.collect()

        if DISTRIBUTED: torch.distributed.barrier(device_ids = [rank])
