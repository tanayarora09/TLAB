import torch
import torch.nn.functional as F
import torchvision
from torchvision.transforms import v2 as transforms
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch import distributed as dist
import torch.multiprocessing as mp
import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

BICUBIC = torchvision.transforms.InterpolationMode.BICUBIC

batch_size = 128

train_transforms = transforms.Compose(
    [
        transforms.Resize([224, 224], interpolation = BICUBIC),
        transforms.RandomCrop([200, 200]),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15, interpolation = BICUBIC),
        transforms.Resize([224, 224]),
        transforms.PILToTensor(),
        transforms.ToDtype(torch.float32, scale = True),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                            (0.2023, 0.1994, 0.2010))
    ]
)

one_hot_transform = transforms.Compose([
    transforms.Lambda(lambda x: torch.tensor(x, dtype=torch.int64)), 
    transforms.Lambda(lambda x: F.one_hot(x, num_classes=10).float()) 
])

test_transforms = transforms.Compose(
    [
        transforms.Resize([224, 224], interpolation = BICUBIC),
        transforms.PILToTensor(),
        transforms.ToDtype(torch.float32, scale = True),
        transforms.Normalize((0.4914, 0.4822, 0.4465), 
                            (0.2023, 0.1994, 0.2010))
    ]
)


train_data = torchvision.datasets.CIFAR10("/u/tanaya_guest/tlab/datasets/CIFAR10/", train = True, download = False,
                                        transform = train_transforms, target_transform = one_hot_transform)

test_data = torchvision.datasets.CIFAR10("/u/tanaya_guest/tlab/datasets/CIFAR10/", train = False, download = False,
                                        transform = test_transforms, target_transform = one_hot_transform)

dt = torch.utils.data.DataLoader(train_data, batch_size = batch_size,  
                                num_workers = 4, pin_memory = True,
                                prefetch_factor = 4, pin_memory_device = "cuda")

dv = torch.utils.data.DataLoader(test_data, batch_size = batch_size,  
                                num_workers = 4, pin_memory = True,
                                prefetch_factor = 4, pin_memory_device = "cuda")

