# Hyperparameter Access Guide

## Overview

This document describes the consolidated hyperparameter access system in the TLAB codebase. All hyperparameters are now centrally managed through the `DataHandle` class, eliminating the need to manually pass batch sizes, transforms, and other dataset-specific parameters throughout the codebase.

## Key Changes

### 1. Centralized DataHandle

Instead of importing from individual data modules and manually creating transforms, use the `DataHandle` class:

```python
from data.index import get_data_object

# Create a handle for your dataset
handle = get_data_object("cifar10")  # or "cifar100", "imagenet", "tiny-imagenet"

# Load transforms once (they get scripted and cached)
handle.load_transforms(device='cuda')
```

### 2. Accessing Hyperparameters

The `DataHandle` provides easy access to all dataset hyperparameters:

```python
# Access hyperparameters directly
batch_size = handle.batch_size  # Default batch size for the dataset
class_size = handle.class_size  # Number of classes
train_size = handle.train_size  # Training set size
val_size = handle.val_size      # Validation set size
input_shape = handle.input_shape  # (C, H, W)

# Calculate cardinality (number of batches)
cardinality = handle.cardinality()  # Uses default batch size
cardinality = handle.cardinality(batch_size=1024)  # Custom batch size
```

### 3. Getting Data Loaders

Get data loaders without specifying batch size (uses default):

```python
# Uses default batch size from hparams
train_loader, val_loader = handle.get_loaders(rank, world_size)

# Or specify a custom batch size
train_loader, val_loader = handle.get_loaders(rank, world_size, batch_size=1024)
```

### 4. Accessing Transforms

Transforms are automatically handled by the `DataHandle`:

```python
# After calling handle.load_transforms(), transforms are accessible:
# - handle.tt: Train transforms (e.g., data augmentation)
# - handle.et: Eval transforms (e.g., center crop)
# - handle.ft: Final transforms (e.g., normalization)

# These are used automatically in the trainer's fit() and evaluate() methods
```

### 5. Trainer Build Method

The trainer `build()` method now accepts a `DataHandle` instead of individual transform tuples:

**Old way (deprecated):**
```python
dataAug = torch.jit.script(DataAugmentation().to('cuda'))
normalize = torch.jit.script(Normalize().to('cuda'))
center_crop = torch.jit.script(CenterCrop().to('cuda'))

T.build(
    optimizer=torch.optim.SGD,
    optimizer_kwargs={'lr': 0.1, 'momentum': 0.9, 'weight_decay': 1e-3},
    loss=torch.nn.CrossEntropyLoss(reduction="sum").to('cuda'),
    collective_transforms=(normalize,),
    train_transforms=(dataAug,),
    eval_transforms=(center_crop,),
    final_collective_transforms=tuple(),
    scale_loss=True,
    gradient_clipnorm=2.0
)
```

**New way:**
```python
handle = get_data_object("cifar10")
handle.load_transforms(device='cuda')

T.build(
    optimizer=torch.optim.SGD,
    optimizer_kwargs={'lr': 0.1, 'momentum': 0.9, 'weight_decay': 1e-3},
    handle=handle,
    loss=torch.nn.CrossEntropyLoss(reduction="sum").to('cuda'),
    scale_loss=True,
    gradient_clipnorm=2.0
)
```

## Complete Example

Here's a complete example of the new pattern:

```python
from data.index import get_data_object
from training.VGG import VGG_CNN
from models.vgg import vgg
import torch
from torch.nn.parallel import DistributedDataParallel as DDP

def train_model(rank, world_size, name):
    # 1. Create DataHandle
    handle = get_data_object("cifar10")
    handle.load_transforms(device='cuda')
    
    # 2. Get loaders (batch size from hparams)
    train_loader, val_loader = handle.get_loaders(rank, world_size)
    
    # 3. Get cardinality (number of batches)
    cardinality = handle.cardinality()
    
    # 4. Create model
    model = vgg(depth=16, rank=rank, world_size=world_size, custom_init=True).cuda()
    if world_size > 1:
        model = DDP(model, device_ids=[rank], output_device=rank)
    
    # 5. Create trainer
    trainer = VGG_CNN(model=model, rank=rank, world_size=world_size)
    
    # 6. Build trainer with handle
    trainer.build(
        optimizer=torch.optim.SGD,
        optimizer_kwargs={'lr': 0.1, 'momentum': 0.9, 'weight_decay': 1e-3},
        handle=handle,
        loss=torch.nn.CrossEntropyLoss(reduction="sum").to('cuda'),
        scale_loss=True,
        gradient_clipnorm=2.0
    )
    
    # 7. Train
    logs = trainer.fit(
        train_loader, val_loader,
        epochs=160,
        train_cardinality=cardinality,
        name=name
    )
    
    return logs
```

## Benefits

1. **Reduced Code Duplication**: No need to repeat transform instantiation in every training file
2. **Centralized Configuration**: All dataset-specific parameters in one place
3. **Type Safety**: Clear interface with typed properties
4. **Maintainability**: Changes to hyperparameters only need to be made in `data/hparams.py`
5. **Consistency**: Same batch sizes and transforms used across all training scripts

## Migration Guide

To migrate old training code:

1. Replace `from data.cifar10 import *` with `from data.index import get_data_object`
2. Create a `DataHandle`: `handle = get_data_object("cifar10")`
3. Call `handle.load_transforms(device='cuda')`
4. Replace `get_loaders(rank, world_size, batch_size=512)` with `handle.get_loaders(rank, world_size)`
5. Replace hardcoded `CARDINALITY = 98` with `cardinality = handle.cardinality()`
6. Update trainer `build()` calls to pass `handle=handle` instead of individual transform tuples
7. Remove manual transform instantiation code

## Dataset Hyperparameters

Current datasets and their default batch sizes:

- **CIFAR-10**: 512
- **CIFAR-100**: 512
- **ImageNet**: 1024
- **Tiny ImageNet**: 512

These can be found and modified in `data/hparams.py`.
