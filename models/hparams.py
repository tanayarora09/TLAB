from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class ModelHparams:
    """Hyperparameters for model configuration."""
    model_name: str  # e.g., "vgg16", "resnet20", "resnet50"
    depth: int
    outfeatures: int  # number of output classes
    inchannels: int = 3
    custom_init: bool = True
    bn_track: bool = False
    dropout: Optional[float] = None
    conv_bias: bool = False  # VGG specific


# Model-specific configurations
MODEL_CONFIGS = {
    "vgg11": {"depth": 11, "valid_depths": (11, 13, 16, 19)},
    "vgg13": {"depth": 13, "valid_depths": (11, 13, 16, 19)},
    "vgg16": {"depth": 16, "valid_depths": (11, 13, 16, 19)},
    "vgg19": {"depth": 19, "valid_depths": (11, 13, 16, 19)},
    "resnet20": {"depth": 20, "valid_depths": tuple(2 + 6 * n for n in range(1, 10)), "type": "cifar"},
    "resnet32": {"depth": 32, "valid_depths": tuple(2 + 6 * n for n in range(1, 10)), "type": "cifar"},
    "resnet44": {"depth": 44, "valid_depths": tuple(2 + 6 * n for n in range(1, 10)), "type": "cifar"},
    "resnet56": {"depth": 56, "valid_depths": tuple(2 + 6 * n for n in range(1, 10)), "type": "cifar"},
    "resnet18": {"depth": 18, "valid_depths": (18, 34, 50, 101, 152), "type": "imagenet"},
    "resnet34": {"depth": 34, "valid_depths": (18, 34, 50, 101, 152), "type": "imagenet"},
    "resnet50": {"depth": 50, "valid_depths": (18, 34, 50, 101, 152), "type": "imagenet"},
    "resnet101": {"depth": 101, "valid_depths": (18, 34, 50, 101, 152), "type": "imagenet"},
    "resnet152": {"depth": 152, "valid_depths": (18, 34, 50, 101, 152), "type": "imagenet"},
}

# Dataset-specific output features (number of classes)
DATASET_CLASSES = {
    "cifar10": 10,
    "cifar100": 100,
    "tiny-imagenet": 200,
    "imagenet": 1000,
}

# Dataset-specific model adjustments
DATASET_MODEL_ADJUSTMENTS = {
    "tiny-imagenet": {
        "dropout": 0.5  # Tiny-ImageNet uses dropout
    }
}


def get_model_hparams(args) -> ModelHparams:
    """
    Build ModelHparams from args with dataset/model-specific fallbacks.
    
    Args flow:
    - args.model (required) → model_name, depth
    - args.dataset (required) → outfeatures (num classes), dataset-specific adjustments
    - args.custom_init (optional) → custom_init (default: True)
    - args.bn_track (optional) → bn_track (default: False)
    - args.dropout (optional) → dropout (default: dataset-specific or None)
    - args.inchannels (optional) → inchannels (default: 3)
    - args.conv_bias (optional) → conv_bias (default: False, VGG only)
    """
    
    # Get model config
    if args.model not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model '{args.model}'. Available: {list(MODEL_CONFIGS.keys())}")
    
    model_config = MODEL_CONFIGS[args.model]
    
    # Get output features from dataset
    if args.dataset not in DATASET_CLASSES:
        raise ValueError(f"Unknown dataset '{args.dataset}'. Available: {list(DATASET_CLASSES.keys())}")
    
    outfeatures = DATASET_CLASSES[args.dataset]
    
    # Get dataset-specific adjustments
    dataset_adjustments = DATASET_MODEL_ADJUSTMENTS.get(args.dataset, {})
    
    # Build hparams with args > dataset adjustments > defaults
    dropout = getattr(args, 'dropout', dataset_adjustments.get('dropout', None))
    custom_init = getattr(args, 'custom_init', True)
    bn_track = getattr(args, 'bn_track', False)
    inchannels = getattr(args, 'inchannels', 3)
    conv_bias = getattr(args, 'conv_bias', False)
    
    return ModelHparams(
        model_name=args.model,
        depth=model_config["depth"],
        outfeatures=outfeatures,
        inchannels=inchannels,
        custom_init=custom_init,
        bn_track=bn_track,
        dropout=dropout,
        conv_bias=conv_bias,
    )
