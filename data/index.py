from typing import Tuple, TYPE_CHECKING
import importlib

from .base import DatasetHparams, BaseModule
from .hparams import DATASET_HPARAMS, DATASET_MODULES

if TYPE_CHECKING:  # pragma: no cover
    import torch.nn as nn

DATASET_REGISTRY = {
    name: {"module": DATASET_MODULES[name], "hparams": hparams}
    for name, hparams in DATASET_HPARAMS.items()
}


def get_data_module(name: str):
    _ensure_known_dataset(name)
    return importlib.import_module(DATASET_REGISTRY[name]["module"])


def get_dataset_hparams(name: str) -> DatasetHparams:
    _ensure_known_dataset(name)
    return DATASET_HPARAMS[name]


def list_datasets() -> Tuple[str, ...]:
    return tuple(DATASET_HPARAMS.keys())


def _ensure_known_dataset(name: str) -> None:
    if name not in DATASET_HPARAMS:
        raise ValueError(f"Unknown dataset '{name}'. Available: {list_datasets()}")


def _get_default_module(name: str) -> BaseModule:
    module = get_data_module(name)
    if not hasattr(module, "DEFAULT_DATA_MODULE"):
        raise AttributeError(f"Dataset module '{name}' missing DEFAULT_DATA_MODULE.")
    return module.DEFAULT_DATA_MODULE


class DataHandle:
    """
    Centralized data access handle providing easy access to:
    - Dataset hyperparameters (hparams) - built from args with dataset fallbacks
    - Data loaders
    - Transforms
    - Module utilities
    """
    
    def __init__(self, name: str, args=None):
        self.name = name
        self.module = get_data_module(name)
        self.dm = _get_default_module(name)
        
        # Get base dataset hparams for immutable properties
        base_hparams = get_dataset_hparams(name)
        
        # Build hparams from args with dataset fallbacks
        from dataclasses import replace
        
        # Determine batch_size: args.batchsize > dataset default
        if args is not None and hasattr(args, 'batchsize') and args.batchsize is not None:
            batch_size = args.batchsize
        else:
            batch_size = base_hparams.default_batch_size
        
        # Create hparams with potentially overridden batch_size
        self.hparams = replace(base_hparams, default_batch_size=batch_size)
            
        self.pt = None
        self.tt = None
        self.et = None
        self.ft = None
        # Automatically load and script transforms
        self.load_transforms()

    def load_transforms(self, device: str = "cuda"):
        """Load and script transforms for the specified device."""
        self.pt, self.tt, self.et, self.ft = self.dm.script_transforms(device=device)

    def eval_transforms(self):
        """Get combined transforms for evaluation (pre + eval + final)."""
        return (*self.pt, *self.et, *self.ft)
    
    def train_transforms(self):
        """Get combined transforms for training (pre + train + final)."""
        return (*self.pt, *self.tt, *self.ft)

    def ptef_transforms(self):
        """Get pre, train, eval, and final transforms (unscripted)."""
        return get_dataset_transforms(self.name)

    def get_loaders(self, rank: int, world_size: int, batch_size: int = None, **kwargs):
        """
        Get train and validation loaders.
        If batch_size is not provided, uses the one from hparams (built from args or dataset default).
        """
        if batch_size is None:
            batch_size = self.hparams.default_batch_size
        return self.module.get_loaders(rank, world_size, batch_size=batch_size, **kwargs)

    def get_partial_train_loader(self, rank: int, world_size: int, batch_size: int = None, **kwargs):
        """
        Get partial train loader.
        If batch_size is not provided, uses the one from hparams (built from args or dataset default).
        """
        if batch_size is None:
            batch_size = self.hparams.default_batch_size
        return self.module.get_partial_train_loader(rank, world_size, batch_size=batch_size, **kwargs)

    def get_sp_loaders(self, batch_size: int = None, **kwargs):
        """
        Get single-process loaders.
        If batch_size is not provided, uses the one from hparams (built from args or dataset default).
        """
        if batch_size is None:
            batch_size = self.hparams.default_batch_size
        return self.module.get_sp_loaders(batch_size=batch_size, **kwargs)

    def custom_fetch_data(self, *args, **kwargs):
        """Fetch custom data from the module."""
        return self.module.custom_fetch_data(*args, **kwargs)
    
    @property
    def batch_size(self) -> int:
        """Get the batch size for this dataset (from args or default)."""
        return self.hparams.default_batch_size
    
    @property
    def class_size(self) -> int:
        """Get the number of classes in this dataset."""
        return self.hparams.class_size
    
    @property
    def train_size(self) -> int:
        """Get the training set size."""
        return self.hparams.train_size
    
    @property
    def val_size(self) -> int:
        """Get the validation set size."""
        return self.hparams.val_size
    
    @property
    def input_shape(self):
        """Get the input shape (C, H, W)."""
        return self.hparams.input_shape
    
    def cardinality(self, batch_size: int = None) -> int:
        """
        Calculate the number of batches in training set.
        If batch_size is not provided, uses the one from hparams (built from args or dataset default).
        """
        if batch_size is None:
            batch_size = self.hparams.default_batch_size
        return self.hparams.cardinality(batch_size)


def get_dataset_transforms(name: str) -> Tuple[Tuple["nn.Module", ...], Tuple["nn.Module", ...], Tuple["nn.Module", ...], Tuple["nn.Module", ...]]:
    dm = _get_default_module(name)
    return (
        dm.create_pre_transforms(),
        dm.create_train_transforms(),
        dm.create_eval_transforms(),
        dm.create_final_transforms(),
    )


def get_data_module_instance(name: str) -> BaseModule:
    return _get_default_module(name)


def get_data_object(name: str, args=None) -> DataHandle:
    _ensure_known_dataset(name)
    return DataHandle(name, args=args)


__all__ = [
    "DATASET_REGISTRY",
    "get_data_module",
    "get_dataset_hparams",
    "list_datasets",
    "get_dataset_transforms",
    "get_data_module_instance",
    "get_data_object",
    "DataHandle",
]