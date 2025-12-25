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

from .hparams import DATASET_HPARAMS
from .base import make_distributed_loader, make_singleprocess_loader, make_subset, BaseModule

IS_ORCA = False
dataset_path = "/disk/bigstuff/imagenet" if not IS_ORCA else "/scratch/tarora_pdx-imagenet/tiny-imagenet"#"/tmp/imagenet

DATA_HPARAMS = DATASET_HPARAMS["tiny-imagenet"]


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
        super().__init__()
        self.size = (64, 64)
        self.scale = (0.6, 1.0)
        self.ratio = (0.8, 1.25)       
        self.randaugment_n = 2
        self.randaugment_m = 9  
        self.cj_b, self.cj_c, self.cj_s, self.cj_h = (0.4, 0.4, 0.4, 0.1)

    def get_params_batch(self, x: torch.Tensor):
        B = x.shape[0]
        device = x.device
        
        H = torch.full((B,), fill_value = 64, dtype = torch.int32, device = device)
        area = H * H

        log_ratio = torch.log(torch.tensor(self.ratio, device=device))

        final_h = torch.full((B,), -1, device=device, dtype=torch.int32)
        final_w = torch.full((B,), -1, device=device, dtype=torch.int32)
        final_i = torch.full((B,), -1, device=device, dtype=torch.int32)
        final_j = torch.full((B,), -1, device=device, dtype=torch.int32)
        success = torch.zeros(B, device=device, dtype=torch.bool)

        for _ in range(10):
            target_area = area * torch.empty(B, device=device).uniform_(self.scale[0], self.scale[1])
            aspect_ratio = torch.exp(torch.empty(B, device=device).uniform_(log_ratio[0], log_ratio[1]))

            crop_w = torch.round(torch.sqrt(target_area * aspect_ratio))
            crop_h = torch.round(torch.sqrt(target_area / aspect_ratio))
            
            valid = (crop_w > 0) & (crop_w <= H) & (crop_h > 0) & (crop_h <= H) & (~success)

            if valid.any():
                max_i = (H - crop_h + 1).clamp(min=1)
                max_j = (H - crop_w + 1).clamp(min=1)

                rand_i = torch.floor((torch.rand(B, device=device) * max_i.float()))
                rand_j = torch.floor(torch.rand(B, device=device) * max_j.float())

                final_h = torch.where(valid, crop_h, final_h)
                final_w = torch.where(valid, crop_w, final_w)
                final_i = torch.where(valid, rand_i, final_i)
                final_j = torch.where(valid, rand_j, final_j)

                success |= valid

            if success.all(): break

        failed = ~success
        if failed.any():
            in_ratio = H.to(torch.float32) / H.to(torch.float32)
            min_ratio = torch.tensor(self.ratio[0], device=device, dtype=torch.float32)
            max_ratio = torch.tensor(self.ratio[1], device=device, dtype=torch.float32)

            fb_w = H.clone()
            fb_h = H.clone()

            wide_mask = in_ratio > max_ratio
            fb_w[wide_mask] = torch.round(H[wide_mask].to(torch.float32) * max_ratio).to(torch.int32)
            
            tall_mask = in_ratio < min_ratio
            fb_h[tall_mask] = torch.round(H[tall_mask].to(torch.float32) / min_ratio).to(torch.int32)

            fb_i = ((H - fb_h) // 2)
            fb_j = ((H - fb_w) // 2)

            final_h = torch.where(failed, fb_h, final_h)
            final_w = torch.where(failed, fb_w, final_w)
            final_i = torch.where(failed, fb_i, final_i)
            final_j = torch.where(failed, fb_j, final_j)

        return final_i, final_j, final_h, final_w

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        
        B, C, H_max, W_max = x.shape
        device = x.device
        x_out = torch.empty_like(x)

        flip_mask = torch.rand(B, device=device) > 0.5
        x = torch.where(flip_mask[:, None, None, None], TF.hflip(x), x)

        i, j, h, w = self.get_params_batch(x)

        crops = [TF.resized_crop(x[b], int(i[b]), int(j[b]), int(h[b]), int(w[b]), self.size) for b in range(b)]
        x = torch.stack(crops, dim = 0)

        for b in range(B):
            img = x[b]
            b_factor = 1.0 + (torch.empty(1, device=device).uniform_(-self.cj_b, self.cj_b).item())
            img = TF.adjust_brightness(img, b_factor)
            c_factor = 1.0 + (torch.empty(1, device=device).uniform_(-self.cj_c, self.cj_c).item())
            img = TF.adjust_contrast(img, c_factor)
            s_factor = 1.0 + (torch.empty(1, device=device).uniform_(-self.cj_s, self.cj_s).item())
            img = TF.adjust_saturation(img, s_factor)
            h_factor = torch.empty(1, device=device).uniform_(-self.cj_h, self.cj_h).item()
            img = TF.adjust_hue(img, h_factor)
            x[b] = img

        return x


class Normalize(nn.Module):
    def __init__(self):
        super(Normalize, self).__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = TF.normalize(x, (0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262), inplace = False)
        return x
    
DEFAULT_DATA_MODULE = BaseModule(
    DATA_HPARAMS,
    train_transforms=(DataAugmentation,),
    eval_transforms= tuple(),
    final_transforms=(Normalize,),
)


def get_loaders(rank, world_size, batch_size = DATA_HPARAMS.default_batch_size, train = True, validation = True, shuffle = True):
    
    if IS_ORCA: _use_scratch_orca()

    dt, dv = None, None
    
    if train:
        train_transform = T.Compose([ScriptedToTensor()]) 


        train_data = NpyImageDataset("data/tiny-imagenet_train.npy", transform = train_transform)
        #torchvision.datasets.ImageFolder(dataset_path + "/train", transform = train_transform)

        dt = make_distributed_loader(train_data, rank, world_size, batch_size, shuffle = shuffle)

    if validation:
        eval_transform = T.Compose([ ScriptedToTensor()]) 
        
        test_data = torchvision.datasets.ImageFolder(dataset_path + "/val", transform = eval_transform)

        dv = make_distributed_loader(test_data, rank, world_size, batch_size, shuffle = shuffle)

    return dt, dv


def get_partial_train_loader(rank, world_size, data_fraction_factor: float = None, batch_count: float = None, batch_size = DATA_HPARAMS.default_batch_size):
    
    if IS_ORCA: _use_scratch_orca()

    train_transform = T.Compose([ ScriptedToTensor()]) 

    train_data = NpyImageDataset("data/tiny-imagenet_train.npy", transform = train_transform)
    #torchvision.datasets.ImageFolder(dataset_path + "/train", transform = train_transform)

    size = len(train_data)
    if batch_count is None and data_fraction_factor is None: raise ValueError 
    
    if batch_count is None: 
        target_samples = size//data_fraction_factor
    else: 
        target_samples = int(min(size, (batch_size // world_size) * batch_count * world_size))
         
    train_data = make_subset(train_data, target_size=target_samples)

    dt = make_distributed_loader(train_data, rank, world_size, batch_size, drop_last = True)

    return dt

def get_sp_loaders(batch_size = 256, train = True, validation = True, shuffle = True):
    
    if IS_ORCA: _use_scratch_orca()

    dt, dv = None, None
    
    if train:
        train_transform = T.Compose([ ScriptedToTensor()]) 


        train_data = NpyImageDataset("data/tiny-imagenet_train.npy", transform = train_transform)
        #torchvision.datasets.ImageFolder(dataset_path + "/train", transform = train_transform)

        dt = make_singleprocess_loader(
            train_data, 
            batch_size = batch_size,
            shuffle = shuffle
        )

    if validation:
        eval_transform = T.Compose([ ScriptedToTensor()]) 
        
        test_data = torchvision.datasets.ImageFolder(dataset_path + "/val", transform = eval_transform)

        dv = make_singleprocess_loader(
            test_data, 
            batch_size = batch_size, 
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