from dataclasses import dataclass
from typing import Optional
import re

@dataclass(frozen=True)
class ModelHparams:
    """Hyperparameters for model configuration."""
    family: str  # e.g., "vgg", "resnet"
    depth: int
    outfeatures: int  # number of output classes
    inchannels: int = 3
    custom_init: bool = True
    bn_track: bool = False
    dropout: Optional[float] = None
    conv_bias: bool = False  # VGG specific


# Dataset-specific output features (number of classes)
DATASET_CLASSES = {
    "cifar10": 10,
    "cifar100": 100,
    "tiny-imagenet": 200,
    "imagenet": 1000,
}

# Dataset-specific model adjustments
DATASET_MODEL_ADJUSTMENTS = {
    
}

#AVAILABLE_MODELS = {"vgg", "resnet", "scnn"}

@dataclass(frozen=True)
class ParsedModel:
    family: str   # "resnet", "vgg", ...
    depth: int

_MODEL_RE = re.compile(r"^([a-zA-Z_]+)-?(\d+)$")

def parse_model_name(name: str):
    m = _MODEL_RE.match(name.lower())
    if not m:
        raise ValueError(
            f"Invalid model name '{name}'. Expected format like 'resnet50', 'vgg16'."
        )
    return m

def return_model_name(name: str):
    m = parse_model_name(name)
    return "".join(m.groups())

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
    parsed_m = parse_model_name(args.model)
    parsed = ParsedModel(family=parsed_m.group(1), depth=int(parsed_m.group(2)))

    """if parsed.family not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model '{args.model}'. Available families: {list(AVAILABLE_MODELS)}")"""
    
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
        family=parsed.family,
        depth=parsed.depth,
        outfeatures=outfeatures,
        inchannels=inchannels,
        custom_init=custom_init,
        bn_track=bn_track,
        dropout=dropout,
        conv_bias=conv_bias,
    )