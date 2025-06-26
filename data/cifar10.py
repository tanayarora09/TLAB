import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler, Subset
import torchvision
from torchvision.transforms.v2 import functional as TF
from PIL import Image

import random

from typing import List

from utils.data_utils import jitToList2D

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

'''
class DataAugmentation(nn.Module):

    """
    Performs Random Addition of:

    Gaussian Noise
    Brightness
    Perspective
    Rotation
    Zoom

    """
    
    def __init__(self):
        super(DataAugmentation, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        batch_size = x.size(0)

        # Horizontal Flip
        flip_mask = torch.rand(batch_size, device=device) > 0.5
        x[flip_mask] = TF.hflip(x[flip_mask])

        # Gaussian Noise
        noise = torch.normal(0.0, 0.625, x.shape, device=device)
        x += noise

        # Brightness
        brightness_factors = torch.empty(batch_size, device=device).uniform_(0.85, 1.15)
        brightness_factors = brightness_factors.view(-1, 1, 1, 1)
        x *= brightness_factors

        # Perspective Transformation
        top_left = torch.randint(0, 29, (batch_size, 2), device=device)
        top_right = torch.cat([torch.randint(195, 224, (batch_size, 1), device=device), torch.randint(0, 29, (batch_size, 1), device=device)], dim=1)
        bottom_right = torch.randint(195, 224, (batch_size, 2), device=device)
        bottom_left = torch.cat([torch.randint(0, 29, (batch_size, 1), device=device), torch.randint(195, 224, (batch_size, 1), device=device)], dim=1)

        startpoints = torch.tensor([[0, 0], [223, 0], [223, 223], [0, 223]], device=device).repeat(batch_size, 1, 1)
        endpoints = torch.stack([top_left, top_right, bottom_right, bottom_left], dim=1)

        # Convert to List[List[int]] compatible with TorchScript
        startpoints_list: List[List[List[int]]] = startpoints.tolist()
        endpoints_list: List[List[List[int]]] = endpoints.tolist()

        x = torch.stack([TF.perspective(img, start, end, interpolation=TF.InterpolationMode.BILINEAR) for img, start, end in zip(x, startpoints_list, endpoints_list)])

        # Rotate
        angles = torch.randint(-16, 17, (batch_size,), device=device)
        x = torch.stack([TF.rotate(img, angle.item(), interpolation=TF.InterpolationMode.BILINEAR) for img, angle in zip(x, angles)])

        # Crop
        i = torch.randint(0, 224 - 190 + 1, (batch_size,), device=device)
        j = torch.randint(0, 224 - 190 + 1, (batch_size,), device=device)
        x = torch.stack([img[:, i_: i_ + 190, j_: j_ + 190] for img, i_, j_ in zip(x, i, j)])

        # Resize
        x = TF.resize(x, [224, 224], interpolation=TF.InterpolationMode.BICUBIC)
        
        return x
'''
"""    @torch.jit.ignore
    def apply_perspective(self, x: torch.Tensor, startpoints: torch.Tensor, endpoints: torch.Tensor) -> torch.Tensor:
        batch_size = x.size(0)
        x_list: List[torch.Tensor] = []
        for i in range(batch_size):
            img = x[i]
            start = startpoints[i].tolist()
            end = endpoints[i].tolist()
            transformed_img = TF.perspective(img, start, end, interpolation=TF.InterpolationMode.BILINEAR)
            x_list.append(transformed_img)
        return torch.stack(x_list)"""

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
        x = TF.normalize(x, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), inplace = True)
        return x


class ScriptedToTensor(nn.Module):

    def __init__(self):
        super(ScriptedToTensor, self).__init__()

    def forward(self, x: Image.Image) -> torch.Tensor:
        x = TF.pil_to_tensor(x)
        x = TF.to_dtype(x, dtype=torch.float32, scale = True)
        return x


def get_loaders(rank, world_size, batch_size = 512):
    """
    Iterate if there are weird behaviors with sample counts
    """

    train_data = torchvision.datasets.CIFAR10("/u/tanaya_guest/tlab/datasets/CIFAR10/", train = True, download = False,
                                              transform = ScriptedToTensor())

    test_data = torchvision.datasets.CIFAR10("/u/tanaya_guest/tlab/datasets/CIFAR10/", train = False, download = False,
                                        transform = ScriptedToTensor())

    dt = DataLoader(train_data, batch_size = batch_size//world_size, 
                    sampler = DistributedSampler(train_data, rank = rank,
                                                 num_replicas = world_size,),
                    pin_memory = True, num_workers = 8, 
                    persistent_workers = True)

    dv = DataLoader(test_data, batch_size = batch_size//world_size, 
                    sampler = DistributedSampler(test_data, rank = rank,
                                                 num_replicas = world_size), 
                    pin_memory = True, num_workers = 8, 
                    persistent_workers = True)
    
    return dt, dv

def get_partial_train_loader(rank, world_size, data_fraction_factor: float = None, batch_count: float = None, batch_size = 128):
    
    train_data = torchvision.datasets.CIFAR10("/u/tanaya_guest/tlab/datasets/CIFAR10/", train = True, download = False,
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