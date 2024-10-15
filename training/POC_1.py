import torch
from torch.nn.parallel import DistributedDataParallel as DDP

from data.cifar10 import *

from utils.serialization_utils import logs_to_pickle, save_tensor
from utils.training_utils import plot_logs

from training.VGG import VGG_POC
from models.VGG import VGG

import json
import pickle
import gc

def main(rank, world_size, name: str, **kwargs):

    EPOCHS = 2
    CARDINALITY = 391

    dataAug = torch.jit.script(DataAugmentation().to('cuda'))
    resize = torch.jit.script(Resize().to('cuda'))
    normalize = torch.jit.script(Normalize().to('cuda'))
    center_crop = torch.jit.script(CenterCrop().to('cuda'))

    model = VGG(19)

    model = DDP(model.to('cuda'), 
                device_ids = [rank],
                output_device = rank, 
                gradient_as_bucket_view = True)

    model = torch.compile(model)
    
    torch._print(f"[rank {rank}] Got Cuda Model")
    
    T = VGG_POC(model, rank)

    T.build(optimizer = torch.optim.SGD(T.m.parameters(), 0.1, momentum = 0.9, weight_decay = 1e-3),
            loss = torch.nn.CrossEntropyLoss(reduction = "sum").to('cuda'),
            collective_transforms = (resize, normalize), train_transforms = (dataAug,),
            eval_transforms = (center_crop,), final_collective_transforms = tuple(),#[normalize],
            scale_loss = True, gradient_clipnorm = 2.0)

    #T.summary(32)

    del model

    torch.cuda.empty_cache()
    gc.collect()

    dt, dv = get_loaders(rank, world_size, iterate = True) 
    
    logs, sparsities_d = T.TicketIMP(dt, dv, EPOCHS, CARDINALITY, name, 0.8, 2, type = "rewind")

    T.evaluate(dt)

    if (rank == 0): 
        print("Train Results: ", T.metric_results())

    T.evaluate(dv)

    if (rank == 0):

        print("Validation Results: ", T.metric_results())
        print("Sparsity: ", T.mm.sparsity)

        with open("./tmp/last_sparsities.json", "w", encoding = "utf-8") as f:
            json.dump(sparsities_d, f, ensure_ascii = False, indent = 4)
            
        #with open(f"./tmp/{name}_IMPs.pickle", 'wb') as file:
        #    pickle.dump(logs, file, protocol=pickle.HIGHEST_PROTOCOL)

        #with open("./tmp/last_activation_log.json", "w", encoding = "utf-8") as f:
        #    json.dump(T.activation_log, f, ensure_ascii = False, indent = 4)
        
    
        for i in range(len(logs)):
            plot_logs(logs[i], EPOCHS, name + f"_{(sparsities_d[i] * 100):.1f}", CARDINALITY) 