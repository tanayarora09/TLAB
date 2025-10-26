import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from data import cifar10, cifar100

from utils.serialization_utils import logs_to_pickle, save_tensor
from utils.training_utils import plot_logs

from training.VGG import VGG_CNN, VGG_IMP
from training.ResNet import ResNet_CNN, ResNet_IMP
from models.vgg import VGG
from models.resnet import ResNet
from models.base import BaseModel

import json
import pickle
import gc
import h5py

def main(rank, world_size, name: str, sp_exp: list, **kwargs):

    EPOCHS = 160
    CARDINALITY = 98
    PRUNE_ITERS = sp_exp[0] #should only be one argument
    is_cifar100 = sp_exp.pop(-1) == 1
    is_vgg = sp_exp.pop(-1) == 1

    data_path = cifar100 if is_cifar100 else cifar10
    outfeatures = 100 if is_cifar100 else 10

    REWIND_EPOCH = 15 if is_vgg else 9

    if len(sp_exp) != 1: raise ValueError()

    old_name = name

    if rank == 0: h5py.File(f"./logs/TICKETS/{old_name}.h5", "w").close()

    dataAug = torch.jit.script(data_path.DataAugmentation().to('cuda'))
    resize = torch.jit.script(data_path.Resize().to('cuda'))
    normalize = torch.jit.script(data_path.Normalize().to('cuda'))
    center_crop = torch.jit.script(data_path.CenterCrop().to('cuda'))

    depth = 16 if is_vgg else 20
    model = (VGG if is_vgg else ResNet)(rank = rank, world_size = world_size, depth = depth, output_channels = outfeatures, custom_init = True).cuda() 

    if world_size > 1:
        model = DDP(model, 
                device_ids = [rank],
                output_device = rank, 
                gradient_as_bucket_view = True)

    T = (VGG_IMP if is_vgg else ResNet_IMP)(model, rank, world_size)

    T.build(optimizer = torch.optim.SGD, optimizer_kwargs = {'lr': 0.1, 'momentum': 0.9, 'weight_decay' : 1e-3},
            loss = torch.nn.CrossEntropyLoss(reduction = "sum").to('cuda'),
            collective_transforms = (resize, normalize), train_transforms = (dataAug,),
            eval_transforms = (center_crop,), final_collective_transforms = tuple(),
            scale_loss = True, gradient_clipnorm = 2.0)

    del model

    torch.cuda.empty_cache()
    gc.collect()

    dt, dv = data_path.get_loaders(rank, world_size, batch_size = 512)

    (logs, results), sparsities_d = T.TicketIMP(dt, dv, EPOCHS, CARDINALITY, old_name, 0.8, PRUNE_ITERS, rewind_iter = REWIND_EPOCH*CARDINALITY, validate = False)

    if rank == 0:

        for spe in list(range(len(results))):
            with open(f"./logs/RESULTS/{old_name}_{spe}.json", "w") as f:
                json.dump(results[spe], f, indent = 6)

        logs_to_pickle(logs, name)

    torch.distributed.barrier(device_ids = [rank])
