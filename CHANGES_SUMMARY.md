# Hyperparameter Consolidation - Summary of Changes

## Problem Statement
The TLAB repository had scattered hyperparameters across multiple classes, making it difficult to:
- Access batch_size and other hyperparameters efficiently
- Maintain consistent configurations across training files
- Update dataset-specific parameters without touching multiple files
- Understand where hyperparameters are defined

Additionally, there was a critical bug where the `build()` method signature in `base.py` was updated to accept `DataHandle` but training files were still calling it with the old signature (individual transform tuples), causing runtime errors.

## Solution Overview

Consolidated all hyperparameters into a centralized, easily accessible system through an enhanced `DataHandle` class that provides:

1. **Direct property access** to hyperparameters (batch_size, class_size, etc.)
2. **Automatic transform management** (no manual instantiation needed)
3. **Smart loader creation** (default batch size from hparams, overridable)
4. **Cardinality calculation** (number of batches in training set)

## Key Changes

### 1. Enhanced DataHandle Class (`data/index.py`)
Added properties and methods for easy hyperparameter access:
```python
@property
def batch_size(self) -> int
def cardinality(self, batch_size: int = None) -> int
def get_loaders(self, rank, world_size, batch_size=None, **kwargs)
# ... and more
```

### 2. Fixed Build Method Signature Error
**Before (broken):**
```python
T.build(
    optimizer=...,
    optimizer_kwargs=...,
    loss=...,
    collective_transforms=(resize, normalize),  # Wrong!
    train_transforms=(dataAug,),                # Wrong!
    eval_transforms=(center_crop,),             # Wrong!
    final_collective_transforms=tuple(),        # Wrong!
    scale_loss=True,
    gradient_clipnorm=2.0
)
```

**After (working):**
```python
handle = get_data_object("cifar10")
handle.load_transforms(device='cuda')

T.build(
    optimizer=...,
    optimizer_kwargs=...,
    handle=handle,  # Correct!
    loss=...,
    scale_loss=True,
    gradient_clipnorm=2.0
)
```

### 3. Eliminated Hardcoded Batch Sizes
**Before:**
```python
dt, dv = get_loaders(rank, world_size, batch_size=512)  # Hardcoded!
CARDINALITY = 98  # Hardcoded!
```

**After:**
```python
handle = get_data_object("cifar10")
dt, dv = handle.get_loaders(rank, world_size)  # Uses default from hparams
cardinality = handle.cardinality()  # Calculated automatically
```

### 4. Simplified Transform Management
**Before:**
```python
from data.cifar10 import DataAugmentation, Normalize, CenterCrop, Resize

dataAug = torch.jit.script(DataAugmentation().to('cuda'))
resize = torch.jit.script(Resize().to('cuda'))
normalize = torch.jit.script(Normalize().to('cuda'))
center_crop = torch.jit.script(CenterCrop().to('cuda'))
```

**After:**
```python
from data.index import get_data_object

handle = get_data_object("cifar10")
handle.load_transforms(device='cuda')
# Transforms are now accessible as handle.tt, handle.et, handle.ft
```

## Files Modified

### Core Infrastructure
- **data/index.py**: Enhanced `DataHandle` with hyperparameter properties
- **.gitignore**: Added to prevent tracking build artifacts

### Training Files (Fixed build signatures and eliminated hardcoded values)
1. **training/naive_training.py**
2. **training/reference.py**
3. **training/imp_training.py**
4. **training/imp_backdrop.py**
5. **training/sanity_checks.py**
6. **training/concrete_training.py**
7. **training/saliency_training.py**
8. **training/interpolate_errors.py**

### Documentation
- **HPARAMS_USAGE.md**: Comprehensive guide with examples and migration instructions

## Statistics

- **11 files changed**
- **471 insertions(+), 162 deletions(-)** (net +309 lines, mostly documentation)
- **8 training files** updated to use new pattern
- **100%** of hardcoded batch_size values replaced
- **100%** of build() calls fixed

## Benefits

1. **Bug Fixes**: Resolved critical build() signature mismatch causing runtime errors
2. **Maintainability**: Hyperparameters now in one location (`data/hparams.py`)
3. **Consistency**: All training files use same access pattern
4. **Simplicity**: Less boilerplate code in training files
5. **Type Safety**: Clear interface with typed properties
6. **Flexibility**: Easy to override defaults when needed
7. **Documentation**: Clear examples and migration guide

## Backward Compatibility

The changes maintain backward compatibility where possible:
- Old data module imports still work
- Can still specify custom batch sizes when needed
- Existing model and optimizer code unchanged

## Testing

- Syntax validation passed for all modified files
- No import errors detected
- Build method signatures verified
- Documentation examples tested

## Next Steps

Users can now:
1. Use `get_data_object()` to create handles
2. Access hyperparameters via properties
3. Pass handles to trainer build methods
4. Refer to HPARAMS_USAGE.md for examples

## Migration Example

See HPARAMS_USAGE.md for complete migration guide with before/after examples.
