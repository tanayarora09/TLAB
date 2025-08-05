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


CONCRETE_EXPERIMENTS = {0: ("Loss", SNIPConcrete),
                        1: ("Gradnorm", GraSPConcrete),
                        2: ("KldLogit", KldLogit),
                        3: ("MseFeature", NormalizedMseFeatures),
                        4: ("GradMatch", StepAlignmentConcrete),
                        5: ("DeltaLoss", LossChangeConcrete)}

def ddp_network(rank, 
                world_size, 
                is_vgg):

    depth = 16 if is_vgg else 20
    
    model = (VGG if is_vgg else ResNet)(depth = depth, rank = rank, world_size = world_size, custom_init = True).cuda()

    if world_size > 1:
        model = DDP(model, 
            device_ids = [rank],
            output_device = rank, 
            gradient_as_bucket_view = True)

    return model

def run_concrete(rank, world_size, 
                 name, is_vgg,  
                 type_of_concrete,
                 type_of_sanity,
                 is_gradnorm, 
                 concrete_epochs,
                 state, spe, spr, 
                 dt, transforms,):

    model = ddp_network(rank, world_size, is_vgg)
    model_to_inspect = model.module if world_size > 1 else model
    if state is not None: model.load_state_dict(state)
    state = model.state_dict()

    inp_args = {"rank": rank, "world_size": world_size, "model": model,}

    if type_of_concrete == 3:
        captures = []
        fcaptures = []
        for bname, block in model_to_inspect.named_children():
            for lname, layer in block.named_children():
                if lname.endswith("relu"): captures.append(layer)
                elif lname.endswith("fc"): fcaptures.append((layer, lambda x: torch.softmax(x, dim = 1)))
        inp_args.update({"capture_layers": captures, "fake_capture_layers": fcaptures})

    search = CONCRETE_EXPERIMENTS[type_of_concrete][1](**inp_args)

    search.build(spr, torch.optim.Adam, optimizer_kwargs = {'lr': 1e-1}, transforms = transforms, use_gradnorm_approach = is_gradnorm)

    logs, ticket = search.optimize_mask(dt, concrete_epochs, CARDINALITY, dynamic_epochs = False, reduce_epochs = [120], invert_mask = (type_of_sanity == 1))

    search.finish()

    if type_of_sanity == 0:
        model_to_inspect.set_ticket(ticket)
        model_to_inspect.prune_random_given_layerwise(model_to_inspect.export_layerwise_sparsities(), distributed = True)
        ticket = model_to_inspect.export_ticket_cpu()

    if rank == 0: plot_logs_concrete(logs, name = name)

    return state, ticket


def _make_trainer(rank, world_size, is_vgg, state = None, ticket = None):

    model = ddp_network(rank, world_size, is_vgg)
    if state is not None: model.load_state_dict(state)
    model_to_inspect = model.module if world_size > 1 else model
    if ticket is not None: model_to_inspect.set_ticket(ticket)

    if (rank == 0):
        if world_size == 1: print(f"Training with sparsity {(model.sparsity.item()):.3e}% \n")
        else: print(f"Training with sparsity {(model.module.sparsity.item()):.3e}% \n")

    return (VGG_CNN if is_vgg else ResNet_CNN)(model, rank, world_size)



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




def run_fit_and_export(rank, world_size, 
                       name, old_name, 
                       state, ticket, 
                       is_vgg, start_epochs,
                       spe, spr, type_of_sanity,
                       dt, dv, transforms,): #Transforms: DataAug, Resize, Normalize, Crop

    T = _make_trainer(rank, world_size, is_vgg, None if type_of_sanity == 2 else state, ticket)

    T.mm.export_ticket(old_name, entry_name = f"{spr * 100:.3e}", root = 0)

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

    logs = T.fit(dt, dv, EPOCHS, CARDINALITY, name, validate = True, save = False, save_init = False, start = start_epochs)

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
            json.dump({"original": orig_train_res, "finetuned": train_res}, f, indent = 6)
        
        logs_to_pickle(logs, name)

        plot_logs(logs, EPOCHS, name, CARDINALITY, start = start_epochs)

def main(rank, world_size, name: str, args: list, **kwargs):

    is_vgg = args.pop(-1) == 1 # 1 is yes, 0 is no
    is_gradnorm = True # 1 is yes, 0 is no
    is_short = args.pop(-1) == 1 # 1 is yes, 0 is no
    is_init = args.pop(-1) == 1 # 1 is yes, 0 is no
    type_of_concrete = 0 if is_init else 2
    type_of_sanity = args.pop(-1) #0-2, Shuffle Layerwise, Invert Score, Reinit Weights
    sanity_name = ("Shuffled" if type_of_sanity == 0 else ("Inverted" if type_of_sanity == 1 else "Reinit"))
    if rank == 0: print(f"VGG: {is_vgg} | GradBalance: {is_gradnorm} | INIT: {is_init} | TYPE: {CONCRETE_EXPERIMENTS[type_of_concrete][0]} | SANITY: {sanity_name}")

    if type_of_sanity == 2 and not is_init:
        print("Reinit not compatible with rewind. Exiting.")
        exit()

    sp_exp = list(range(2, 43 if is_vgg else 33, 2)) if len(args) == 0 else args

    name = f"{CONCRETE_EXPERIMENTS[type_of_concrete][0].lower()}_{sanity_name.lower()}_{'gradbalance' if is_gradnorm else 'multiplier'}_{'init' if is_init else 'rewind'}_{'short' if is_short else 'long'}_{'vgg16' if is_vgg else 'resnet20'}_{name}" 

    start_epochs = 0 if is_init else (5 if is_vgg else 3)
    start_epochs *= 3
    concrete_epochs = 20 if is_short else 160

    DISTRIBUTED = world_size > 1

    old_name = name

    dataAug = torch.jit.script(DataAugmentation().to('cuda'))
    resize = torch.jit.script(Resize().to('cuda'))
    normalize = torch.jit.script(Normalize().to('cuda'))
    center_crop = torch.jit.script(CenterCrop().to('cuda'))

    dt, dv = get_loaders(rank, world_size, batch_size = 512)

    if not is_init: 
        ostate, _ = run_start_train(rank = rank, world_size = world_size, 
                                            name = name, is_vgg = is_vgg, start_epochs = start_epochs,
                                            dt = dt, dv = dv, transforms = (dataAug, resize, normalize, center_crop,))
    else: 
        ostate = None

    for spe in sp_exp:

        spr = 0.8**spe
        name = old_name + f"_{spe:02d}"

        state, ticket = run_concrete(rank = rank, world_size = world_size, name = name, 
                                     is_vgg = is_vgg, type_of_concrete = type_of_concrete, type_of_sanity = type_of_sanity,
                                     is_gradnorm = is_gradnorm, concrete_epochs = concrete_epochs, state = ostate, 
                                     spe = spe, spr = spr, dt = dt, transforms = (resize, normalize, center_crop,),)

        torch.cuda.empty_cache()
        gc.collect()
        
        run_fit_and_export(rank = rank, world_size = world_size, name = name, old_name = old_name, 
                           state = state, ticket = ticket, is_vgg = is_vgg, start_epochs = start_epochs,
                           spe = spe, spr = spr, type_of_sanity = type_of_sanity, dt = dt, dv = dv, 
                           transforms = (dataAug, resize, normalize, center_crop,))
        
        torch.cuda.empty_cache()
        gc.collect()

        if DISTRIBUTED: torch.distributed.barrier(device_ids = [rank])
