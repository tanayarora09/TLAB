import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from data.cifar10 import *

from utils.serialization_utils import logs_to_pickle, save_tensor
from utils.training_utils import plot_logs

from training.VGG import VGG_CNN
from models.VGG import VGG

import json
import pickle
import gc

def main(rank, world_size, name: str, lock, shared_list, **kwargs):

    EPOCHS = 5
    CARDINALITY = 782

    dataAug = torch.jit.script(DataAugmentation().to('cuda'))
    resize = torch.jit.script(Resize().to('cuda'))
    normalize = torch.jit.script(Normalize().to('cuda'))
    center_crop = torch.jit.script(CenterCrop().to('cuda'))

    model = VGG(depth = 19, rank = rank)

    model.prune_random(0.02, distributed = True, root = 0)

    model = DDP(model.to('cuda'), 
                device_ids = [rank],
                output_device = rank, 
                gradient_as_bucket_view = True,
                broadcast_buffers = False)

    #model = torch.compile(model)
    
    T = VGG_CNN(model = model, rank = rank, world_size = world_size)
    
    T.build(optimizer = torch.optim.SGD, optim_kwargs = {"lr": 0.1, "momentum" : 0.9, "weight_decay" : 1e-3},
            loss = torch.nn.CrossEntropyLoss(reduction = "sum").to('cuda'),
            collective_transforms = (resize, normalize), train_transforms = (dataAug,),
            eval_transforms = (center_crop,), final_collective_transforms = tuple(),
            scale_loss = False, gradient_clipvalue = 0.025)

    #T.summary(32)

    del model

    torch.cuda.empty_cache()
    gc.collect()

    dt, dv = get_loaders(rank, world_size, batch_size = 64) 
    """
    logs1 = T.fit(dt, dv, EPOCHS, CARDINALITY, name + "dense", save = True, rewind_iter = 0, verbose = False)

    T.evaluate(dt)

    if (rank == 0): 
        print("Train Results: ", T.metric_results())

    T.evaluate(dv)

    if (rank == 0):

        print("Validation Results: ", T.metric_results())
        print("Sparsity: ", T.mm.sparsity)

        logs_to_pickle(logs1, name + "dense")

        plot_logs(logs1, EPOCHS, name + "dense", steps = CARDINALITY)


    torch.distributed.barrier(device_ids = [rank])

    T.load_ckpt(name + "dense", prefix = "rewind")
    """
    T.mm.migrate_to_sparse()
    T.reset_optimizer(reset_lr = True)

    del dt, dv

    dt, dv = get_loaders(rank, world_size, batch_size = 64)

    logs2 = T.fit(dt, dv, EPOCHS, CARDINALITY, name + "sparse", save = True, rewind_iter = 0, verbose = False)

    T.evaluate(dt)

    if (rank == 0): 
        print("Train Results: ", T.metric_results())

    T.evaluate(dv)

    if (rank == 0):

        print("Validation Results: ", T.metric_results())
        print("Sparsity: ", T.mm.sparsity)

        logs_to_pickle(logs2, name + "sparse")

        plot_logs(logs2, EPOCHS, name + "sparse", steps = CARDINALITY)

    torch.distributed.barrier(device_ids = [rank])