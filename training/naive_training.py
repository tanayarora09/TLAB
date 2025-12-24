import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from data.cifar10 import *

from utils.serialization_utils import logs_to_pickle, save_tensor
from utils.training_utils import plot_logs

from training.VGG import VGG_CNN
from models.vgg import VGG
from training.ResNet import ResNet_CNN
from models.resnet import ResNet
from models.base import MaskedModel

import json
import pickle
import gc
import h5py

def main(rank, world_size, name: str, sp_exp: list, **kwargs):

    is_vgg = sp_exp.pop(-1) == 1 # VGG vs ResNet
    is_rand = sp_exp.pop(-1) == 1 # Random Pruning vs Magnitude Pruning (pre init)

    EPOCHS = 160
    CARDINALITY = 98

    old_name = name

    if rank == 0: h5py.File(f"./logs/TICKETS/{old_name}.h5", "w").close()

    dt, dv = get_loaders(rank, world_size, batch_size = 512) 

    for spe in reversed(sp_exp):

        name = old_name + f"_{spe:02d}"

        sp = 0.8 ** spe

        dataAug = torch.jit.script(DataAugmentation().to('cuda'))
        resize = torch.jit.script(Resize().to('cuda'))
        normalize = torch.jit.script(Normalize().to('cuda'))
        center_crop = torch.jit.script(CenterCrop().to('cuda'))

        depth = 16 if is_vgg else 20
        model = (VGG if is_vgg else ResNet)(rank, world_size, depth = depth, custom_init = True).cuda()
        
        if world_size > 1:

            model = DDP(model, 
                    device_ids = [rank],
                    output_device = rank, 
                    gradient_as_bucket_view = True)

        model_to_inspect = model.module if world_size > 1 else model
        if is_rand: model_to_inspect.prune_random(sp, distributed = True)
        else: model_to_inspect.prune_by_mg(sp, 1, root = 0)

        T = (VGG_CNN if is_vgg else ResNet_CNN)(model = model, rank = rank, world_size = world_size)

        T.build(optimizer = torch.optim.SGD, optimizer_kwargs = {'lr': 0.1, 'momentum': 0.9, 'weight_decay' : 1e-3},
                loss = torch.nn.CrossEntropyLoss(reduction = "sum").to('cuda'),
                collective_transforms = (resize, normalize), train_transforms = (dataAug,),
                eval_transforms = (center_crop,), final_collective_transforms = tuple(),
                scale_loss = True, gradient_clipnorm = 2.0)

        torch.cuda.empty_cache()
        gc.collect()

        logs = T.fit(dt, dv, EPOCHS, CARDINALITY, name, save = False, save_init = False, verbose = False, validate = False)

        if rank == 0: 
            logs_to_pickle(logs, name)
           
        T.m.eval()

        T.evaluate(dt)

        if (rank == 0): 
            train_res = T.metric_results()
            print("Train Results: ", train_res)

            #T.mm.export_ticket(old_name, entry_name = f"{sp * 100:.3e}")

        T.evaluate(dv)
        
        if (rank == 0):
            val_res =  T.metric_results()
            print("Validation Results: ", val_res)
            print("Sparsity: ", T.mm.sparsity)
            train_res.update({('val_' + k): v for k, v in val_res.items()})
        
            with open(f"./logs/RESULTS/{old_name}_{spe}.json", "w") as f:
                json.dump(train_res, f, indent = 6)
        
        torch.distributed.barrier(device_ids = [rank])