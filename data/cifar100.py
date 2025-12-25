import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler, Subset
import torchvision
from torchvision.transforms.v2 import functional as TF
from PIL import Image

import random

from typing import List

from utils.data_utils import jitToList2D
from .hparams import DATASET_HPARAMS
from .base import make_distributed_loader, make_singleprocess_loader, make_subset, BaseModule
import os
import shutil
import subprocess
from filelock import FileLock


IS_ORCA = False

dataset_path = "/tmp/CIFAR100/" if IS_ORCA else "/u/tanaya_guest/tlab/datasets/CIFAR100/"

DATA_HPARAMS = DATASET_HPARAMS["cifar100"]


class DataAugmentation(nn.Module):
    
    def __init__(self):
        super(DataAugmentation, self).__init__()

    def forward(self, x: torch.Tensor):
        
        device = x.device
        batch_size = x.size(0)

        flip_mask = torch.rand(batch_size, device=device) > 0.5
        x[flip_mask] = TF.hflip(x[flip_mask])

        x = nn.functional.pad(x, (4, 4, 4, 4), mode="reflect")
        
        i = torch.randint(0, 40 - 32 + 1, (batch_size,), device=device)
        j = torch.randint(0, 40 - 32 + 1, (batch_size,), device=device)

        x = torch.stack([img[:, i_: i_ + 32, j_: j_ + 32] for img, i_, j_ in zip(x, i, j)])

        return x

class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = TF.normalize(x, (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761), inplace = True)
        return x


class ScriptedToTensor(nn.Module):

    def __init__(self):
        super(ScriptedToTensor, self).__init__()

    def forward(self, x: Image.Image) -> torch.Tensor:
        x = TF.pil_to_tensor(x)
        x = TF.to_dtype(x, dtype=torch.float32, scale = True)
        return x

DEFAULT_DATA_MODULE = BaseModule(
    DATA_HPARAMS,
    train_transforms=(DataAugmentation,),
    eval_transforms=tuple(),
    final_transforms=(Normalize,),
)



def get_loaders(rank, world_size, batch_size = DATA_HPARAMS.default_batch_size, train = True, validation = True):
    """
    Iterate if there are weird behaviors with sample counts
    """
    
    if IS_ORCA: _use_scratch_orca()

    dt, dv = None, None
    
    if train:

        train_data = torchvision.datasets.CIFAR100(dataset_path, train = True, download = False,
                                                transform = ScriptedToTensor())

        dt = make_distributed_loader(train_data, rank, world_size, batch_size)

    if validation:

        test_data = torchvision.datasets.CIFAR100(dataset_path, train = False, download = False,
                                            transform = ScriptedToTensor())

        dv = make_distributed_loader(test_data, rank, world_size, batch_size)
    
    return dt, dv

def get_partial_train_loader(rank, world_size, data_fraction_factor: float = None, batch_count: float = None, batch_size = DATA_HPARAMS.default_batch_size):
    
    if IS_ORCA: _use_scratch_orca()

    train_data = torchvision.datasets.CIFAR100(dataset_path, train = True, download = False,
                                              transform = ScriptedToTensor())
    
    size = len(train_data)
    if batch_count is None and data_fraction_factor is None: raise ValueError 
    if batch_count is None: target_size = size//data_fraction_factor
    else: target_size = min(size, (batch_size * batch_count))
    train_data = make_subset(train_data, target_size=target_size)

    dt = make_distributed_loader(train_data, rank, world_size, batch_size, drop_last = True)

    return dt

def get_sp_loaders(batch_size = 128, train = True, validation = True):
    """
    Iterate if there are weird behaviors with sample counts
    """
    
    if IS_ORCA: _use_scratch_orca()

    dt, dv = None, None
    
    if train:

        train_data = torchvision.datasets.CIFAR100(dataset_path, train = True, download = False,
                                                transform = ScriptedToTensor())

        dt = make_singleprocess_loader(train_data, batch_size = batch_size, shuffle = True)

    if validation:

        test_data = torchvision.datasets.CIFAR100(dataset_path, train = False, download = False,
                                            transform = ScriptedToTensor())

        dv = make_singleprocess_loader(test_data, batch_size = batch_size, shuffle = True)
    
    return dt, dv

def custom_fetch_data(dataloader, amount, samples=10, classes=10, sampler_offset=None):
    
    if samples == 0: return None
    
    if sampler_offset is not None:
        dataloader.sampler.set_epoch(sampler_offset)
    
    results = []  # Stores (X, Y) pairs for each iteration
    data_pool = [[] for _ in range(classes)]
    label_pool = [[] for _ in range(classes)]
    
    finish_loop = False
    
    for inputs, targets, *_ in dataloader:  # Single iteration through dataloader
        if finish_loop: continue
        for idx in range(inputs.shape[0]):
            x, y = inputs[idx:idx+1], targets[idx:idx+1]
            category = y.item()
            if len(data_pool[category]) < amount * samples:
                data_pool[category].append(x)
                label_pool[category].append(y)
        
        if all(len(data_pool[c]) >= amount * samples for c in range(classes)):
            finish_loop = True
    
    for i in range(amount):
        datas = [data_pool[c][i * samples:(i + 1) * samples] for c in range(classes)]
        labels = [label_pool[c][i * samples:(i + 1) * samples] for c in range(classes)]
        X = torch.cat([torch.cat(_, 0) for _ in datas])
        Y = torch.cat([torch.cat(_) for _ in labels]).view(-1)
        results.append((X, Y))

    return results

def _use_scratch_orca():
    source_dir = os.path.expanduser("~/datasets/CIFAR100")
    target_dir = "/tmp/CIFAR100"
    lock_path = "/tmp/CIFAR100.lock"

    def is_non_empty_dir(path):
        return os.path.isdir(path) and len(os.listdir(path)) > 0

    with FileLock(lock_path):
        if not is_non_empty_dir(target_dir):
            print(f"[{os.getpid()}] Copying CIFAR-100 dataset with rsync...")
            if os.path.exists(target_dir):
                shutil.rmtree(target_dir)
            os.makedirs(target_dir, exist_ok=True)
            subprocess.run(["rsync", "-a", f"{source_dir}/", target_dir], check=True)
            print(f"[{os.getpid()}] Copy complete.")
        else:
            print(f"[{os.getpid()}] Dataset already available.")

