import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from data.cifar10 import *

from utils.serialization_utils import logs_to_pickle, save_tensor
from utils.training_utils import plot_logs

from training.VGG import VGG_CNN
from models.VGG import VGG

from training.ResNet import ResNet_CNN
from models.ResNet import ResNet

from search.salient import *

import json
import pickle
import gc

EPOCHS = 160
CARDINALITY = 98

TYPES = [("SNIP", SNIP_Pruner), ("GraSP", GraSP_Pruner), ("SynFlow", SynFlow_Pruner), 
         ("KldLogit", KldLogit_Pruner), ("MseFeature", MSE_Pruner), ("GradMatch", GradMatch_Pruner)]


def ddp_network(rank, world_size, is_vgg, bn_track = False):

    depth = 16 if is_vgg else 20
    
    model = (VGG if is_vgg else ResNet)(depth = depth, rank = rank, world_size = world_size, custom_init = True, bn_track = bn_track).cuda()
    
    if world_size > 1:

        if bn_track: model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)

        model = DDP(model, 
                device_ids = [rank],
                output_device = rank, 
                gradient_as_bucket_view = True)
    
    return model

def run_salient(rank, world_size, name, old_name, type_of_salient, steps_of_salient, is_vgg, spe, spr, transforms, state = None): #ONLY ON ROOT

    model = ddp_network(rank, world_size, is_vgg, bn_track = (type_of_salient == 2 and type_of_salient == 5))
    if state is not None: model.load_state_dict(state)
    state = model.state_dict()
    
    if rank == 0:
        inp_args = {}
        if type_of_salient == 4: 
            captures = []
            fcaptures = []
            model_to_inspect = model.module if world_size > 1 else model
            for bname, block in model_to_inspect.named_children():
                for lname, layer in block.named_children():
                    if lname.endswith("relu"): captures.append(layer)
                    elif lname.endswith("fc"): fcaptures.append((layer, lambda x: torch.softmax(x, dim = 1)))
            inp_args.update({"capture_layers": captures, "fake_capture_layers": fcaptures})

        pruner = (TYPES[type_of_salient][1])(0, 1, model.module if world_size > 1 else model, **inp_args)
        pruner.build(spr, transforms, input = None)
        ticket = pruner.grad_mask(steps = steps_of_salient)
        pruner.finish()

    else:
        ticket = torch.zeros(model.module.num_prunable, dtype = torch.bool, device = 'cuda')

    if world_size > 1:
        torch.distributed.barrier(device_ids = [rank])
        torch.distributed.broadcast(ticket, src = 0)

    return state, ticket

def _make_trainer(rank, world_size, is_vgg, state = None, ticket = None):

    model = ddp_network(rank, world_size, is_vgg)
    if state is not None: model.load_state_dict(state, strict = False) # SynFlow requires finding tickets with running batchnorm.
    model_to_inspect = model.module if world_size > 1 else model
    if ticket is not None: model_to_inspect.set_ticket(ticket)

    if (rank == 0):
        print(model_to_inspect.sparsity, "\n")

    return (VGG_CNN if is_vgg else ResNet_CNN)(model, rank, world_size)

def run_fit_and_export(rank, world_size, name, old_name, state, ticket, is_init, is_vgg, spe, spr, transforms, dt, dv): #Transforms: DataAug, Resize, Normalize, Crop

    T = _make_trainer(rank, world_size, is_vgg, state, ticket, )

    T.build(optimizer = torch.optim.SGD, optimizer_kwargs = {'lr': 0.1, 'momentum': 0.9, 'weight_decay': 1e-3},
            loss = torch.nn.CrossEntropyLoss(reduction = "sum").to('cuda'),
            collective_transforms = (transforms[1], transforms[2]), train_transforms = (transforms[0],),
            eval_transforms = (transforms[3],), final_collective_transforms = tuple(),
            scale_loss = True, gradient_clipnorm = 2.0)
    
    start_epochs = 0 if is_init else (5 if is_vgg else 3)

    logs = T.fit(dt, dv, EPOCHS, CARDINALITY, name, validate = False, save = False, save_init = False, start = start_epochs)

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

def run_start_train(rank, world_size, 
                    name, is_vgg, start_epochs, 
                    dt, dv, transforms,): 

    T = _make_trainer(rank, world_size, is_vgg)

    T.build(optimizer = torch.optim.SGD, optimizer_kwargs = {'lr': 0.1, 'momentum': 0.9, 'weight_decay': 1e-3},
            loss = torch.nn.CrossEntropyLoss(reduction = "sum").to('cuda'),
            collective_transforms = (transforms[1], transforms[2]), train_transforms = (transforms[0],),
            eval_transforms = (transforms[3],), final_collective_transforms = tuple(),
            scale_loss = True, gradient_clipnorm = 2.0)

    T.fit(dt, dv, start_epochs, CARDINALITY, name + "_pretrain", save = False, save_init = False, validate = False)

    T.evaluate(dt)

    if (rank == 0): 
        train_res = T.metric_results()
        print("Train Results: ", train_res)

    T.evaluate(dv)
    
    if (rank == 0):
        val_res =  T.metric_results()
        print("Validation Results: ", val_res) 

    return T.m.state_dict(), T.optim.state_dict()

def main(rank, world_size, name: str, sp_exp: list, **kwargs):

    is_vgg = sp_exp.pop(-1) == 1 # 1 is yes, 0 is no
    is_init = sp_exp.pop(-1) == 1 
    type_of_salient = sp_exp.pop(-1) # SNIP, GraSP, SynFlow, KldLogit, MseFeature, GradMatch
    steps_of_salient = sp_exp.pop(-1) # 1, 100 are common
    print(f"TYPE: {TYPES[type_of_salient][0]} | VGG: {is_vgg}")

    DISTRIBUTED = world_size > 1

    old_name = name

    dataAug = torch.jit.script(DataAugmentation().to('cuda'))
    resize = torch.jit.script(Resize().to('cuda'))
    normalize = torch.jit.script(Normalize().to('cuda'))
    center_crop = torch.jit.script(CenterCrop().to('cuda'))


    dt, dv = get_loaders(rank, world_size, batch_size = 512) 

    ckpt = None
    if not is_init:
        start_epochs = 5 if is_vgg else 3
        ckpt, _ = run_start_train(rank, world_size, name + "_pretrain", is_vgg, start_epochs, dt, dv, (dataAug, resize, normalize, center_crop,))

    for spe in reversed(sp_exp):

        spr = 0.8**spe
        name = old_name + f"_{spe:02d}"

        state, ticket = run_salient(rank, world_size, name, old_name, type_of_salient, steps_of_salient,
                                    is_vgg, spe, spr, (resize, normalize, center_crop,), state = ckpt)

        torch.cuda.empty_cache()
        gc.collect()
        
        run_fit_and_export(rank, world_size, name, old_name, state, ticket, is_init, 
                 is_vgg, spe, spr, (dataAug, resize, normalize, center_crop,), dt, dv)

        
        torch.cuda.empty_cache()
        gc.collect()

        if DISTRIBUTED: torch.distributed.barrier(device_ids = [rank])
