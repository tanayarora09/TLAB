import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler, Subset
import torchvision
from torchvision.transforms.v2 import functional as TF
from PIL import Image

import random

from typing import List

from utils.data_utils import jitToList2D

import os
import shutil
import subprocess
from filelock import FileLock


IS_ORCA = False

dataset_path = "/tmp/CIFAR100/" if IS_ORCA else "/u/tanaya_guest/tlab/datasets/Cifar100/"

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

        # Perform batched cropping
        x = torch.stack([img[:, i_: i_ + 32, j_: j_ + 32] for img, i_, j_ in zip(x, i, j)])

        return x

class Resize(nn.Module):

    def __init__(self):
        super(Resize, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x = TF.resize(x, [224, 224], interpolation=TF.InterpolationMode.BICUBIC)
        return x

class CenterCrop(nn.Module):
    def __init__(self):
        super(CenterCrop, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #x = TF.center_crop(x, [200, 200])
        #x = TF.resize(x, [224, 224], interpolation = TF.InterpolationMode.BICUBIC)
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


def get_loaders(rank, world_size, batch_size = 128, train = True, validation = True):
    """
    Iterate if there are weird behaviors with sample counts
    """
    
    if IS_ORCA: _use_scratch_orca()

    dt, dv = None, None
    
    if train:

        train_data = torchvision.datasets.CIFAR100(dataset_path, train = True, download = False,
                                                transform = ScriptedToTensor())

        dt = DataLoader(train_data, batch_size = batch_size//world_size, 
                        sampler = DistributedSampler(train_data, rank = rank,
                                                    num_replicas = world_size,),
                        pin_memory = True, num_workers = 8, 
                        persistent_workers = True)

    if validation:

        test_data = torchvision.datasets.CIFAR100(dataset_path, train = False, download = False,
                                            transform = ScriptedToTensor())

        dv = DataLoader(test_data, batch_size = batch_size//world_size, 
                        sampler = DistributedSampler(test_data, rank = rank,
                                                    num_replicas = world_size), 
                        pin_memory = True, num_workers = 8, 
                        persistent_workers = True)
    
    return dt, dv

def get_partial_train_loader(rank, world_size, data_fraction_factor: float = None, batch_count: float = None, batch_size = 128):
    
    if IS_ORCA: _use_scratch_orca()

    train_data = torchvision.datasets.CIFAR10(dataset_path, train = True, download = False,
                                              transform = ScriptedToTensor())
    
    size = len(train_data)
    if batch_count is None and data_fraction_factor is None: raise ValueError 
    if batch_count is None: indices = torch.randperm(size)[:(size//data_fraction_factor)]
    else: indices = torch.randperm(size)[:min(size, (batch_size * batch_count))]
    train_data = Subset(train_data, indices)

    dt = DataLoader(train_data, batch_size = batch_size//world_size, 
                    sampler = DistributedSampler(train_data, rank = rank,
                                                 num_replicas = world_size,),
                    pin_memory = True, num_workers = 8, 
                    persistent_workers = True, drop_last = True)

    return dt

def custom_fetch_data(dataloader, amount, samples=10, classes=100, sampler_offset=None):
    
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