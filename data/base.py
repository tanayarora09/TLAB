from dataclasses import dataclass
import math
from typing import Callable, Optional, Sequence, Tuple, TYPE_CHECKING

try:  # pragma: no cover - allow import without torch for metadata-only usage
    import torch
except ImportError:  # pragma: no cover
    torch = None

if TYPE_CHECKING:  # pragma: no cover
    import torch.nn as nn


@dataclass(frozen=True)
class DatasetHparams:

    name: str
    class_size: int
    train_size: int
    val_size: int
    input_shape: Tuple[int, int, int]
    default_batch_size: int

    def cardinality(self, batch_size: Optional[int] = None) -> int:
        bsz = batch_size or self.default_batch_size
        return math.ceil(self.train_size / bsz)


def _per_process_batch(batch_size: int, world_size: int) -> int:
    return batch_size // world_size


def _require_torch():
    if torch is None:
        raise ImportError("torch is required for data loading utilities.")
    from torch.utils.data import DataLoader, DistributedSampler, Subset
    return DataLoader, DistributedSampler, Subset


class BaseModule:

    def __init__(
        self,
        hparams: DatasetHparams,
        *,
        pre_transforms: Sequence[Callable[[], "nn.Module"]] = (),
        train_transforms: Sequence[Callable[[], "nn.Module"]] = (),
        eval_transforms: Sequence[Callable[[], "nn.Module"]] = (),
        final_transforms: Sequence[Callable[[], "nn.Module"]] = (),
    ):
        self.hparams = hparams
        self._pre_factories = tuple(pre_transforms)
        self._train_factories = tuple(train_transforms)
        self._eval_factories = tuple(eval_transforms)
        self._final_factories = tuple(final_transforms)
        self._script_cache = {}

    def create_pre_transforms(self):
        return tuple(factory() for factory in self._pre_factories)

    def create_train_transforms(self):
        return tuple(factory() for factory in self._train_factories)

    def create_eval_transforms(self):
        return tuple(factory() for factory in self._eval_factories)

    def create_final_transforms(self):
        return tuple(factory() for factory in self._final_factories)

    def script_transforms(self, device: str = "cuda"):
        if torch is None:
            raise ImportError("torch is required to script transforms.")
        if device in self._script_cache:
            return self._script_cache[device]

        def _compile(seq):
            compiled = []
            for t in seq:
                try:
                    compiled.append(torch.jit.script(t.to(device)))
                except Exception:
                    compiled.append(t.to(device))
            return tuple(compiled)

        scripted = (
            _compile(self.create_pre_transforms()),
            _compile(self.create_train_transforms()),
            _compile(self.create_eval_transforms()),
            _compile(self.create_final_transforms()),
        )
        self._script_cache[device] = scripted
        return scripted


def make_distributed_loader(
    dataset,
    rank: int,
    world_size: int,
    batch_size: int,
    *,
    shuffle: bool = True,
    collate_fn=None,
    drop_last: bool = False,
    num_workers: int = 8,
):
    DataLoader, DistributedSampler, _ = _require_torch()
    return DataLoader(
        dataset,
        batch_size=_per_process_batch(batch_size, world_size),
        sampler=DistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=shuffle),
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True,
        collate_fn=collate_fn,
        drop_last=drop_last,
    )


def make_singleprocess_loader(
    dataset,
    *,
    batch_size: int,
    shuffle: bool = True,
    collate_fn=None,
    num_workers: int = 8,
):
    DataLoader, _, _ = _require_torch()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        pin_memory=True,
        num_workers=num_workers,
        persistent_workers=True,
        shuffle=shuffle,
        collate_fn=collate_fn,
    )


def make_subset(
    dataset,
    *,
    target_size: int,
):
    _, _, Subset = _require_torch()
    indices = torch.randperm(len(dataset))[:target_size]
    return Subset(dataset, indices)