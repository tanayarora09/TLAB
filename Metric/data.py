import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch import distributed as dist
import torchvision
from torchvision.transforms.v2 import functional as TF
import os
from typing import Tuple
from PIL import Image

class DataAugmentation(nn.Module):
    
    def __init__(self):
        super(DataAugmentation, self).__init__()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1) > 0.5:
            TF.hflip(x) 
        angle = torch.randint(-15, 16, (1,)).item() # Rand Rotate
        x = TF.rotate(x, angle, interpolation=TF.InterpolationMode.BILINEAR)
        i = torch.randint(0, 224 - 200 + 1, (1, )).item()
        j = torch.randint(0, 224 - 200 + 1, (1, )).item()
        x = TF.crop(x, i, j, 200, 200)
        x = TF.resize(x, [224, 224], interpolation = TF.InterpolationMode.BICUBIC)
        return x

class ResizeAndNormalize(nn.Module):
    def __init__(self):
        super(ResizeAndNormalize, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = TF.resize(x, [224, 224], interpolation=TF.InterpolationMode.BICUBIC)
        x = TF.normalize(x, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), inplace = True)
        return x


class ScriptedToTensor(nn.Module):

    def __init__(self):
        super(ScriptedToTensor, self).__init__()

    
    def forward(self, x: Image.Image) -> torch.Tensor:
        x = TF.pil_to_tensor(x)
        #x = x.to('cuda') # to prefetch to gpu?
        x = TF.to_dtype(x, dtype=torch.float32, scale = True)
        #x = x.to(memory_format = torch.channels_last)
        return x
"""
class ScriptedOneHot(nn.Module):

    def __init__(self):
        super(ScriptedOneHot, self).__init__()

    def forward(self, x: int) -> torch.Tensor:
        return F.one_hot(torch.as_tensor(x), num_classes = 10).float()"""

class DisributedLoader(DataLoader):

    def __init__(self, data, batch_size, num_workers, rank, world_size, shuffle = False, prefetch_factor = 4):
        self.rank = rank
        self.world_size = world_size
        self.sampler = DistributedSampler(data, num_replicas=world_size, rank=rank, shuffle=shuffle)
        super().__init__(data, batch_size=batch_size, sampler=self.sampler, 
                         num_workers=num_workers, pin_memory=True, pin_memory_device = 'cuda',
                         prefetch_factor=prefetch_factor, persistent_workers=True)
    
    @torch.jit.ignore
    def __iter__(self):
        return super().__iter__()

def setup_distribute(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank = rank, world_size = world_size)

def cleanup_distribute():
    dist.destroy_process_group()

def get_cifar(rank, world_size, batch_size = 64):

    train_data = torchvision.datasets.CIFAR10("/u/tanaya_guest/tlab/datasets/CIFAR10/", train = True, download = False, # "/u/tanaya_guest/tlab/datasets/CIFAR10/"
                                              transform = torch.compile(ScriptedToTensor()))

    test_data = torchvision.datasets.CIFAR10("/u/tanaya_guest/tlab/datasets/CIFAR10/", train = False, download = False,
                                        transform = torch.compile(ScriptedToTensor()))

    dt = DisributedLoader(
        train_data,
        batch_size//world_size,
        4,
        rank,
        world_size
    )

    dv = DisributedLoader(
        test_data,
        batch_size//world_size,
        4,
        rank,
        world_size
    )

    return dt, dv
