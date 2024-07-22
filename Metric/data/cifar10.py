import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
import torchvision
from torchvision.transforms.v2 import functional as TF
from PIL import Image

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

        if torch.rand(1) > 0.5: # Horizontal Flip
            TF.hflip(x)
        
        noise = torch.normal(0.0, 0.25, x.shape, device = device) # Gaussian Noise
        x += noise

        brightness_factor = torch.empty((1, )).uniform_(0.85, 1.15).item() # Brightness
        x = TF.adjust_brightness(x, brightness_factor)

        topleft = [ int(torch.randint(0, 26, size=(1,), device = device).item()), # Perspective
            int(torch.randint(0, 26, size=(1,), device = device).item()),] # 12 = int(0.1 * 112) + 1
        topright = [ int(torch.randint(198, 224, size=(1,), device = device).item()), # 212 = 224 - 12 
            int(torch.randint(0, 26, size=(1,), device = device).item()),]
        botright = [ int(torch.randint(198, 224, size=(1,), device = device).item()),
            int(torch.randint(198, 224, size=(1,), device = device).item()),]
        botleft = [ int(torch.randint(0, 26, size=(1,), device = device).item()),
            int(torch.randint(198, 224, size=(1,), device = device).item()),]
        
        x = TF.perspective(x, [[0, 0], [223, 0], [223, 223], [0, 223]], 
                           [topleft, topright, botright, botleft],
                           interpolation = TF.InterpolationMode.BILINEAR)

        angle = torch.randint(-15, 16, (1,)).item() # Rotate
        x = TF.rotate(x, angle, interpolation=TF.InterpolationMode.BILINEAR)
        
        i = torch.randint(0, 224 - 190 + 1, (1, ), device = device).item() # Crop
        j = torch.randint(0, 224 - 190 + 1, (1, ), device = device).item()
        x = x[..., i: i + 190, j: j + 190]
        
        x = TF.resize(x, [224, 224], interpolation = TF.InterpolationMode.BICUBIC) # Resize
        
        return x

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
        self.sampler = DistributedSampler(data, num_replicas=world_size, rank=rank, shuffle=shuffle)
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
        continue
    
    for step, (x, y, id) in enumerate(dv):
        x, y = x.to('cuda'), y.to('cuda')
        continue

    return dt, dv
