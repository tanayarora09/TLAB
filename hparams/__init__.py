from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass(frozen=True)
class DatasetHparams:
    num_classes: int
    train_size: int
    total_epochs: int
    base_batch_size: int
    learning_rate: float
    warmup_epochs: int
    lr_milestones: List[int]


@dataclass(frozen=True)
class ModelHparams:
    depth: int
    start_epoch: int
    prune_rate: float


@dataclass(frozen=True)
class TrainingHparams:
    total_epochs: int
    batch_size: int
    learning_rate: float
    momentum: float
    weight_decay: float
    warmup_epochs: int
    lr_milestones: List[int]
    start_epoch: int
    micro_batch_size: Optional[int] = None

    def cardinality(self, train_size: int) -> int:
        return math.ceil(train_size / self.batch_size)


@dataclass(frozen=True)
class SearchHparams:
    prune_rate: float


@dataclass(frozen=True)
class ExperimentHparams:
    model: ModelHparams
    data: DatasetHparams
    train: TrainingHparams
    search: SearchHparams

    @property
    def cardinality(self) -> int:
        return self.train.cardinality(self.data.train_size)


_MODEL_CONFIGS: Dict[str, ModelHparams] = {
    "resnet20": ModelHparams(depth=20, start_epoch=9, prune_rate=0.8),
    "vgg16": ModelHparams(depth=16, start_epoch=15, prune_rate=0.8),
    "resnet50": ModelHparams(depth=50, start_epoch=18, prune_rate=0.1),
}

_DATASET_CONFIGS: Dict[str, DatasetHparams] = {
    "cifar10": DatasetHparams(
        num_classes=10,
        train_size=50000,
        total_epochs=160,
        base_batch_size=512,
        learning_rate=1e-1,
        warmup_epochs=0,
        lr_milestones=[80, 120],
    ),
    "cifar100": DatasetHparams(
        num_classes=100,
        train_size=50000,
        total_epochs=160,
        base_batch_size=512,
        learning_rate=1e-1,
        warmup_epochs=0,
        lr_milestones=[80, 120],
    ),
    "imagenet": DatasetHparams(
        num_classes=1000,
        train_size=1281167,
        total_epochs=90,
        base_batch_size=1024,
        learning_rate=4e-1,
        warmup_epochs=5,
        lr_milestones=[30, 60, 80],
    ),
    "tiny-imagenet": DatasetHparams(
        num_classes=200,
        train_size=100000,
        total_epochs=200,
        base_batch_size=512,
        learning_rate=4e-1,
        warmup_epochs=0,
        lr_milestones=[60, 120, 160],
    ),
}

_MICRO_BATCH_SIZES = {"cifar10": 256, "cifar100": 256, "imagenet": 64}
_TINY_IMAGENET_PRUNE = 0.31622776601
_MOMENTUM = 0.9


def _batch_size(dataset: str, model: str, pipeline: str) -> int:
    base = _DATASET_CONFIGS[dataset].base_batch_size
    if pipeline == "concrete" and model == "resnet50":
        return max(base, 1024)
    return base


def _learning_rate(dataset: str, model: str, pipeline: str) -> float:
    if pipeline == "saliency" and model == "resnet50":
        return 4e-1
    return _DATASET_CONFIGS[dataset].learning_rate


def _total_epochs(dataset: str, model: str, pipeline: str) -> int:
    if pipeline == "saliency":
        return {"resnet20": 160, "vgg16": 160, "resnet50": 90}[model]
    return _DATASET_CONFIGS[dataset].total_epochs


def _warmup_epochs(dataset: str, pipeline: str) -> int:
    if pipeline == "concrete" and dataset == "tiny-imagenet":
        return 10
    return _DATASET_CONFIGS[dataset].warmup_epochs


def _weight_decay(dataset: str, model: str, pipeline: str) -> float:
    if pipeline == "imp":
        return 1e-4 if dataset == "imagenet" else 1e-3
    if pipeline == "concrete":
        if dataset == "tiny-imagenet":
            return 5e-4
        return 1e-4 if model == "resnet50" else 1e-3
    if pipeline == "saliency":
        return 1e-4 if model == "resnet50" else 1e-3
    raise ValueError(f"Unsupported pipeline: {pipeline}")


def _prune_rate(dataset: str, model: str, pipeline: str) -> float:
    if dataset == "tiny-imagenet":
        return _TINY_IMAGENET_PRUNE
    if pipeline == "saliency" and model == "resnet50":
        return _TINY_IMAGENET_PRUNE
    return _MODEL_CONFIGS[model].prune_rate


def build_experiment_hparams(model: str, dataset: str, *, time: str = "rewind", pipeline: str) -> ExperimentHparams:
    if model not in _MODEL_CONFIGS:
        raise ValueError(f"Unsupported model: {model}")
    if dataset not in _DATASET_CONFIGS:
        raise ValueError(f"Unsupported dataset: {dataset}")

    model_cfg = _MODEL_CONFIGS[model]
    dataset_cfg = _DATASET_CONFIGS[dataset]

    start_epoch = 0 if time == "init" else model_cfg.start_epoch

    train = TrainingHparams(
        total_epochs=_total_epochs(dataset, model, pipeline),
        batch_size=_batch_size(dataset, model, pipeline),
        learning_rate=_learning_rate(dataset, model, pipeline),
        momentum=_MOMENTUM,
        weight_decay=_weight_decay(dataset, model, pipeline),
        warmup_epochs=_warmup_epochs(dataset, pipeline),
        lr_milestones=dataset_cfg.lr_milestones,
        start_epoch=start_epoch,
        micro_batch_size=_MICRO_BATCH_SIZES.get(dataset),
    )

    search = SearchHparams(prune_rate=_prune_rate(dataset, model, pipeline))

    return ExperimentHparams(
        model=model_cfg,
        data=dataset_cfg,
        train=train,
        search=search,
    )


__all__ = [
    "DatasetHparams",
    "ModelHparams",
    "TrainingHparams",
    "SearchHparams",
    "ExperimentHparams",
    "build_experiment_hparams",
]
