import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, DistributedSampler
from torch import distributed as dist
import torchvision
from torchvision.transforms import v2

class ScriptedAugmentation(nn.Module):
    
    def __init__(self):
        super(ScriptedAugmentation, self).__init__()
        self.transforms = torch.jit.script(v2.Compose([
        v2.Resize([224, 224], interpolation = v2.InterpolationMode.BICUBIC),
        v2.RandomCrop([200, 200]),
        v2.RandomHorizontalFlip(),
        v2.RandomRotation(15, interpolation = v2.InterpolationMode.BICUBIC),
    ]))
    
    def forward(self, x):
        return self.transforms(x)
    
class ScriptedPreProcess(nn.Module):

    def __init__(self):
        super(ScriptedPreProcess, self).__init__()
        self.transforms = torch.compile(v2.Compose([
        v2.Resize([224, 224], interpolation = v2.InterpolationMode.BICUBIC),
        v2.PILToTensor(),
        v2.ToDtype(torch.float32, scale = True),
        v2.Normalize((0.4914, 0.4822, 0.4465), 
                            (0.2023, 0.1994, 0.2010))
        ]))
    
    def forward(self, x):
        return self.transforms(x)
    
class ScriptedOneHot(nn.Module):

    def __init__(self, num_classes=10):
        super(ScriptedOneHot, self).__init__()
        self.num_classes = num_classes

    @torch.jit.script_method
    def forward(self, x):
        return F.one_hot(x, num_classes=self.num_classes).float()

class DisributedLoader(DataLoader):

    def __init__(self, data, batch_size, num_workers, rank, world_size, shuffle = True, prefetch_factor = 2):
        self.rank = rank
        self.world_size = world_size
        self.prefetch_factor = prefetch_factor
        self.sampler = DistributedSampler(data, num_replicas=world_size, rank=rank, shuffle=shuffle)
        super().__init__(data, batch_size=batch_size, sampler=self.sampler, 
                         num_workers=num_workers, pin_memory=True, 
                         prefetch_factor=prefetch_factor, persistent_workers=True)
        
        self._prefetch_to_device()

    def _prefetch_to_device(self): # Prefetch the entire dataset to the GPU cause cifar 10 is tiny
        for batch in self:
            batch[0] = batch[0].to(self.rank, non_blocking=True)
            batch[1] = batch[1].to(self.rank, non_blocking=True)
    
    @torch.jit.ignore
    def __iter__(self):
        return super().__iter__()

def setup_distribute(rank, world_size):
    dist.init_process_group("nccl", rank = rank, world_size = world_size)

def cleanup_distribute():
    dist.destroy_process_group()

def get_cifar(rank, world_size, batch_size = 128):

    setup_distribute(rank, world_size)

    train_data = torchvision.datasets.CIFAR10("/u/tanaya_guest/tlab/datasets/CIFAR10/", train = True, download = False, 
                                              transform = nn.ModuleList([ScriptedAugmentation, ScriptedPreProcess]), target_transform = ScriptedOneHot)

    test_data = torchvision.datasets.CIFAR10("/u/tanaya_guest/tlab/datasets/CIFAR10/", train = False, download = False,
                                        transform = ScriptedPreProcess, target_transform = ScriptedOneHot)

    dt = DisributedLoader(
        train_data,
        batch_size,
        0,
        rank,
        world_size
    )

    dv = DisributedLoader(
        test_data,
        batch_size,
        0,
        rank,
        world_size
    )

    return dt, dv
