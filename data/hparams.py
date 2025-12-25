from .base import DatasetHparams

DATASET_MODULES = {
    "cifar10": "data.cifar10",
    "cifar100": "data.cifar100",
    "imagenet": "data.imagenet",
    "tiny-imagenet": "data.tiny_imagenet",
}

DATASET_HPARAMS = {
    "cifar10": DatasetHparams(
        name="cifar10",
        class_size=10,
        train_size=50000,
        val_size=10000,
        input_shape=(3, 32, 32),
        default_batch_size=512,
    ),
    "cifar100": DatasetHparams(
        name="cifar100",
        class_size=100,
        train_size=50000,
        val_size=10000,
        input_shape=(3, 32, 32),
        default_batch_size=512,
    ),
    "imagenet": DatasetHparams(
        name="imagenet",
        class_size=1000,
        train_size=1281167,
        val_size=50000,
        input_shape=(3, 224, 224),
        default_batch_size=1024,
    ),
    "tiny-imagenet": DatasetHparams(
        name="tiny-imagenet",
        class_size=200,
        train_size=100000,
        val_size=10000,
        input_shape=(3, 64, 64),
        default_batch_size=512,
    ),
}

__all__ = ["DATASET_HPARAMS", "DATASET_MODULES"]