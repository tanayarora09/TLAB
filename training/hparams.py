from dataclasses import dataclass
from typing import Optional, List
from models.hparams import return_model_name

@dataclass(frozen=True)
class TrainingHparams:
    """Hyperparameters for training configuration."""
    learning_rate: float
    momentum: float
    weight_decay: float
    epochs: int
    warmup_epochs: int
    reduce_epochs: List[int]  # Epochs at which to reduce LR
    gradient_clipnorm: float
    rewind_epoch: int
    scale_loss: bool  # AMP/mixed precision

# Dataset-specific training defaults
DATASET_TRAINING_DEFAULTS = {
    "cifar10": {
        "learning_rate": 0.1,
        "epochs": 160,
        "warmup_epochs": 0,
        "reduce_epochs": [80, 120],
        "weight_decay": 1e-3,
    },
    "cifar100": {
        "learning_rate": 0.1,
        "epochs": 160,
        "warmup_epochs": 0,
        "reduce_epochs": [80, 120],
        "weight_decay": 1e-3,
    },
    "tiny-imagenet": {
        "learning_rate": 0.4,
        "epochs": 200,
        "warmup_epochs": 10,
        "reduce_epochs": [60, 120, 160],
        "weight_decay": 1e-4,
    },
    "imagenet": {
        "learning_rate": 0.4,
        "epochs": 90,
        "warmup_epochs": 5,
        "reduce_epochs": [30, 60, 80],
        "weight_decay": 1e-4,
    },
}

# Model-specific training adjustments
MODEL_TRAINING_ADJUSTMENTS = {
    "resnet50": {
        "weight_decay": 1e-4,  # ResNet50 uses different weight decay
        "rewind_epoch": 18,
    },
    "vgg16": {
        "rewind_epoch": 15,
    },
    "resnet20": {
        "rewind_epoch": 9,
    },
}


def get_training_hparams(args) -> TrainingHparams:
    """
    Build TrainingHparams from args with dataset/model-specific fallbacks.
    
    Args flow:
    - args.dataset (required) → base training params
    - args.model (optional) → model-specific adjustments
    - args.learning_rate, args.epochs, etc. (optional) → override defaults
    """
    
    # Get dataset defaults
    if args.dataset not in DATASET_TRAINING_DEFAULTS:
        raise ValueError(f"Unknown dataset '{args.dataset}'. Available: {list(DATASET_TRAINING_DEFAULTS.keys())}")
    
    defaults = DATASET_TRAINING_DEFAULTS[args.dataset].copy()
    
    # Apply model-specific adjustments
    if hasattr(args, 'model') and return_model_name(args.model) in MODEL_TRAINING_ADJUSTMENTS:
        defaults.update(MODEL_TRAINING_ADJUSTMENTS[return_model_name(args.model)])
    
    # Override with args if provided
    learning_rate = getattr(args, 'learning_rate', defaults["learning_rate"])
    epochs = getattr(args, 'epochs', defaults["epochs"])
    warmup_epochs = getattr(args, 'warmup_epochs', defaults["warmup_epochs"])
    reduce_epochs = getattr(args, 'reduce_epochs', defaults["reduce_epochs"])
    weight_decay = getattr(args, 'weight_decay', defaults["weight_decay"])
    
    # Common defaults with args override
    momentum = getattr(args, 'momentum', 0.9)
    rewind_epoch = getattr(args, 'rewind_epoch', defaults.get('rewind_epoch', 0))
    if getattr(args, "when_to_prune", None) == "init": rewind_epoch = 0

    gradient_clipnorm = getattr(args, 'gradient_clipnorm', 2.0)
    scale_loss = getattr(args, 'scale_loss', True)
    
    return TrainingHparams(
        learning_rate=learning_rate,
        momentum=momentum,
        weight_decay=weight_decay,
        epochs=epochs,
        warmup_epochs=warmup_epochs,
        reduce_epochs=reduce_epochs,
        rewind_epoch=rewind_epoch,
        gradient_clipnorm=gradient_clipnorm,
        scale_loss=scale_loss,
    )