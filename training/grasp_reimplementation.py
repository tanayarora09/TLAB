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


def ddp_network(rank, world_size, is_vgg, depth = 16):

    if not is_vgg: depth = 20
    
    if is_vgg:
        model = VGG(depth = depth, rank = rank, world_size = world_size, custom_init = True) 
    else: 
        model =  ResNet(depth = depth, rank = rank, world_size = world_size, custom_init = True)
    
    model = model.cuda()
    #print(type(model))
    
    if world_size > 1:
        model = DDP(model, 
                device_ids = [rank],
                output_device = rank, 
                gradient_as_bucket_view = True)
    
    return model

def run_grasp(rank, world_size, name, old_name, is_grasp, is_vgg, spe, spr, transforms): #ONLY ON ROOT

    model = ddp_network(rank, world_size, is_vgg, 16)
    state = model.state_dict()
    
    if rank == 0:
        if is_grasp: pruner = GraSP_Pruner(0, 1, model.module)
        else: pruner = SynFlow_Pruner(0, 1, model.module)
        #else: pruner = SNIP_Pruner(0, 1, model)#.module)
        #partial = get_partial_train_loader(rank, world_size, 10)
        pruner.build(spr, transforms, input = None)#partial)
        ticket = pruner.grad_mask()
        pruner.finish()

    else:
        ticket = torch.zeros(model.module.num_prunable, dtype = torch.bool, device = 'cuda')

    torch.distributed.barrier(device_ids = [rank])
    torch.distributed.broadcast(ticket, src = 0)

    return state, ticket

def _make_trainer(rank, world_size, state, ticket, is_vgg):

    model = ddp_network(rank, world_size, is_vgg, 16)
    model.load_state_dict(state)
    model.module.set_ticket(ticket)
    #model.set_ticket(ticket)

    if (rank == 0):
        print(model.module.sparsity, "\n")
        #print(model.sparsity, "\n")

    return VGG_CNN(model, rank, world_size) if is_vgg else ResNet_CNN(model, rank, world_size)

def run_fit_and_export(rank, world_size, name, old_name, state, ticket, is_vgg, spe, spr, transforms,): #Transforms: DataAug, Resize, Normalize, Crop

    T = _make_trainer(rank, world_size, state, ticket, is_vgg)

    T.build(optimizer = torch.optim.SGD, optimizer_kwargs = {'lr': 0.1, 'momentum': 0.9, 'weight_decay': 1e-3},
            loss = torch.nn.CrossEntropyLoss(reduction = "sum").to('cuda'),
            collective_transforms = (transforms[1], transforms[2]), train_transforms = (transforms[0],),
            eval_transforms = (transforms[3],), final_collective_transforms = tuple(),
            scale_loss = True, gradient_clipnorm = 2.0)
    
    dt, dv = get_loaders(rank, world_size, batch_size = 512) 

    logs = T.fit(dt, dv, EPOCHS, CARDINALITY, name, validate = False, save = False, save_init = False)

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
        
    T.mm.export_ticket(old_name, entry_name = f"{spr * 100:.2f}", root = 0)

    del dt, dv

def main(rank, world_size, name: str, sp_exp: list, **kwargs):

    is_grasp = False #sp_exp.pop(-1) == 1
    is_vgg = sp_exp.pop(-1) == 1 # 1 is yes, 0 is no
    print(f"GRASP: {is_grasp} | VGG: {is_vgg}")

    DISTRIBUTED = world_size > 1

    old_name = name

    dataAug = torch.jit.script(DataAugmentation().to('cuda'))
    resize = torch.jit.script(Resize().to('cuda'))
    normalize = torch.jit.script(Normalize().to('cuda'))
    center_crop = torch.jit.script(CenterCrop().to('cuda'))

    for spe in sp_exp:

        spr = 0.8**spe
        name = old_name + f"_{spe:02d}"

        state, ticket = run_grasp(rank, world_size, name, old_name, is_grasp, is_vgg, 
                                  spe, spr, (resize, normalize, center_crop,))

        torch.cuda.empty_cache()
        gc.collect()
        
        run_fit_and_export(rank, world_size, name, old_name, state, ticket,
                 is_vgg, spe, spr, (dataAug, resize, normalize, center_crop,))

        
        torch.cuda.empty_cache()
        gc.collect()

        if DISTRIBUTED: torch.distributed.barrier(device_ids = [rank])
