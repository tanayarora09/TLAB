import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
import torchvision
from torchvision.transforms.v2 import functional as TF
from PIL import Image

from typing import List

from utils.data_utils import jitToList2D

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
        x = TF.resize(x, [224, 224], interpolation=TF.InterpolationMode.BICUBIC)
        return x

class CenterCrop(nn.Module):
    def __init__(self):
        super(CenterCrop, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = TF.center_crop(x, [200, 200])
        x = TF.resize(x, [224, 224], interpolation = TF.InterpolationMode.BICUBIC)
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

class CIFAR10WithID(torchvision.datasets.CIFAR10):
    
    def __getitem__(self, index: int):
        data, target = super().__getitem__(index)
        unique_id = str(index)
        return data, target, unique_id

class DisributedLoader(DataLoader):

    def __init__(self, data, batch_size, num_workers, rank, world_size, shuffle = False, prefetch_factor = 4):
        self.rank = rank
        self.world_size = world_size
        self.sampler = DistributedSampler(data, num_replicas=world_size, rank=rank, shuffle=shuffle, seed = 42)
        super().__init__(data, batch_size=batch_size, sampler=self.sampler, 
                         num_workers=num_workers, pin_memory=True, pin_memory_device = 'cuda',
                         prefetch_factor=prefetch_factor, persistent_workers=True)
    
    @torch.jit.ignore
    def __iter__(self):
        return super().__iter__()


def get_loaders(rank, world_size, batch_size = 128, iterate: bool = False):
    """
    Iterate if there are weird behaviors with sample counts
    """

    train_data = CIFAR10WithID("/u/tanaya_guest/tlab/datasets/CIFAR10/", train = True, download = False,
                                              transform = torch.compile(ScriptedToTensor()))

    test_data = CIFAR10WithID("/u/tanaya_guest/tlab/datasets/CIFAR10/", train = False, download = False,
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
    
    if not iterate: 

        return dt, dv

    for step, (x, y, id) in enumerate(dt):
        x, y = x.to('cuda'), y.to('cuda')
    
    for step, (x, y, id) in enumerate(dv):
        x, y = x.to('cuda'), y.to('cuda')
        continue

    return dt, dv
