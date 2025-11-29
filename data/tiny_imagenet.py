import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, DistributedSampler, Subset
import torchvision
from torchvision.transforms.v2 import functional as TF
import torchvision.transforms.v2 as T
from PIL import Image

import random
import os
import shutil
import subprocess
import tempfile
from filelock import FileLock

from typing import List, Tuple
import numpy as np

IS_ORCA = False
dataset_path = "/disk/bigstuff/imagenet" if not IS_ORCA else "/scratch/tarora_pdx-imagenet/tiny-imagenet"#"/tmp/imagenet

class NpyImageDataset(torch.utils.data.Dataset):
    def __init__(self, npy_path, transform=None):
        self.samples = np.load(npy_path, allow_pickle=True)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

class ScriptedToTensor(nn.Module):

    def forward(self, x: Image.Image) -> torch.Tensor:
        x = TF.pil_to_tensor(x)
        x = TF.to_dtype(x, dtype=torch.float32, scale=True)
        return x

class DataAugmentation(nn.Module):
    
    def __init__(self):
        super(DataAugmentation, self).__init__()

    def forward(self, x: torch.Tensor):
        
        device = x.device
        batch_size = x.size(0)

        flip_mask = torch.rand(batch_size, device=device) > 0.5
        x[flip_mask] = TF.hflip(x[flip_mask])

        x = nn.functional.pad(x, (4, 4, 4, 4), mode="reflect")
        
        i = torch.randint(0, 72 - 64 + 1, (batch_size,), device=device)
        j = torch.randint(0, 72 - 64 + 1, (batch_size,), device=device)

        # Perform batched cropping
        x = torch.stack([img[:, i_: i_ + 64, j_: j_ + 64] for img, i_, j_ in zip(x, i, j)])

        return x
    

class CenterCrop(nn.Module):
    def __init__(self):
        super(CenterCrop, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = TF.normalize(x, (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262), inplace = False)
        return x
    
def get_loaders(rank, world_size, batch_size = 1024, train = True, validation = True, shuffle = True):
    
    if IS_ORCA: _use_scratch_orca()

    dt, dv = None, None
    per_process_batch_size = batch_size // world_size
    
    if train:
        train_transform = T.Compose([ScriptedToTensor()]) 


        train_data = NpyImageDataset("data/tiny-imagenet_train.npy", transform = train_transform)
        #torchvision.datasets.ImageFolder(dataset_path + "/train", transform = train_transform)

        dt = DataLoader(
            train_data, 
            batch_size = per_process_batch_size, 
            sampler = DistributedSampler(train_data, rank = rank, num_replicas = world_size, shuffle = shuffle),
            pin_memory = True, 
            num_workers = 8, 
            persistent_workers = True
        )

    if validation:
        eval_transform = T.Compose([ ScriptedToTensor()]) 
        
        test_data = torchvision.datasets.ImageFolder(dataset_path + "/val", transform = eval_transform)

        dv = DataLoader(
            test_data, 
            batch_size = per_process_batch_size, 
            sampler = DistributedSampler(test_data, rank = rank, num_replicas = world_size, shuffle = shuffle), 
            pin_memory = True, 
            num_workers = 8, 
            persistent_workers = True
        )
    
    return dt, dv


def get_partial_train_loader(rank, world_size, data_fraction_factor: float = None, batch_count: float = None, batch_size = 1024):
    
    if IS_ORCA: _use_scratch_orca()

    train_transform = T.Compose([ ScriptedToTensor()]) 

    train_data = NpyImageDataset("data/tiny-imagenet_train.npy", transform = train_transform)
    #torchvision.datasets.ImageFolder(dataset_path + "/train", transform = train_transform)

    size = len(train_data)
    if batch_count is None and data_fraction_factor is None: raise ValueError 
    
    if batch_count is None: 
        indices = torch.randperm(size)[:(size//data_fraction_factor)]
    else: 
        per_process_batch_size = batch_size // world_size
        target_samples = int(min(size, per_process_batch_size * batch_count * world_size))
        indices = torch.randperm(size)[:target_samples]
        
    train_data = Subset(train_data, indices)

    dt = DataLoader(
        train_data, 
        batch_size = batch_size//world_size, 
        sampler = DistributedSampler(train_data, rank = rank, num_replicas = world_size),
        pin_memory = True, 
        num_workers = 8, 
        persistent_workers = True, 
        drop_last = True
    )

    return dt

def get_sp_loaders(batch_size = 256, train = True, validation = True, shuffle = True):
    
    if IS_ORCA: _use_scratch_orca()

    dt, dv = None, None
    
    if train:
        train_transform = T.Compose([ ScriptedToTensor()]) 


        train_data = NpyImageDataset("data/tiny-imagenet_train.npy", transform = train_transform)
        #torchvision.datasets.ImageFolder(dataset_path + "/train", transform = train_transform)

        dt = DataLoader(
            train_data, 
            batch_size = batch_size, 
            pin_memory = True, 
            num_workers = 8, 
            persistent_workers = True,
            shuffle = shuffle
        )

    if validation:
        eval_transform = T.Compose([ ScriptedToTensor()]) 
        
        test_data = torchvision.datasets.ImageFolder(dataset_path + "/val", transform = eval_transform)

        dt = DataLoader(
            test_data, 
            batch_size = batch_size, 
            pin_memory = True, 
            num_workers = 8, 
            persistent_workers = True,
            shuffle = shuffle
        )
    
    return dt, dv

def custom_fetch_data(dataloader, amount, samples=10, classes=1000, sampler_offset=None):
    
    if samples == 0: return None
    
    return next(dataloader)

    if sampler_offset is not None:
        dataloader.sampler.set_epoch(sampler_offset)
    
    results = [] 
    data_pool = [[] for _ in range(classes)]
    label_pool = [[] for _ in range(classes)]
    
    finish_loop = False
    
    for inputs, targets, *_ in dataloader: 
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
    return
    source_dir = "/scratch/tarora_pdx-imagenet/imagenet"
    target_dir = "/tmp/imagenet"
    lock_path = "/tmp/imagenet.lock"
    num_workers = 8  # Tune based on disk speed and CPU count

    def is_non_empty_dir(path):
        return os.path.isdir(path) and len(os.listdir(path)) > 0

    with FileLock(lock_path):
        if not is_non_empty_dir(target_dir):
            print(f"[{os.getpid()}] Copying Imagenet dataset in parallel...")
            os.makedirs(target_dir, exist_ok=True)

            with tempfile.TemporaryDirectory() as tmpdir:
                all_files = os.path.join(tmpdir, "all_files.txt")
                subprocess.run(["find", source_dir, "-type", "f"], stdout=open(all_files, "w"), check=True)

                # Split file list into roughly equal chunks
                subprocess.run(["split", "-n", f"l/{num_workers}", all_files, os.path.join(tmpdir, "chunk_")], check=True)

                # Launch parallel rsync workers
                procs = []
                for f in os.listdir(tmpdir):
                    if f.startswith("chunk_"):
                        p = subprocess.Popen([
                            "rsync", "-a",
                            f"--files-from={os.path.join(tmpdir, f)}",
                            source_dir + "/", target_dir + "/"
                        ])
                        procs.append(p)

                # Wait for all to complete
                for p in procs:
                    p.wait()

            print(f"[{os.getpid()}] Copy complete.")
        else:
            print(f"[{os.getpid()}] Dataset already available.")