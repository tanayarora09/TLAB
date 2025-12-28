"""
Training index providing centralized trainer creation with args-driven hparams.
"""

import torch
from .base import BaseCNNTrainer, BaseIMP
from .hparams import get_training_hparams, TrainingHparams


def get_trainer(args, model, trainer_type="cnn"):
    """
    Create a trainer from args with automatic configuration.
    
    Args flow:
    - args.dataset (required) → training hparams (lr, epochs, etc.)
    - args.model (optional) → model-specific training adjustments
    - args.rank, args.world_size (required)
    - trainer_type: "cnn" or "imp"
    
    Args:
        args: Argument object with training config
        model: The model to train
        trainer_type: Type of trainer ("cnn" or "imp")
    
    Returns:
        Trainer instance (VGG_CNN, ResNet_CNN, etc.)
    """
    
    # Determine trainer class based on model name and type
    model_name = args.model if hasattr(args, 'model') else "resnet20"
    
    trainer_class = None
    trainer_class = BaseIMP if trainer_type == "imp" else BaseCNNTrainer

    return trainer_class(model=model, rank=args.rank, world_size=args.world_size)


def _get_optimizer(optimizer_name: str):
    oname = optimizer_name.lower()
    for attr in dir(torch.optim):
        if attr.lower() == oname:
            return attr
    raise ValueError(f"Unknown optimizer '{optimizer_name}' in torch.optim.")    

def build_trainer(trainer, args, data_handle, loss=None):
    """
    Build/configure a trainer with optimizer and loss from args.
    
    Args flow:
    - args.dataset, args.model → training hparams (lr, momentum, weight_decay, etc.)
    - args.optimizer (optional) → optimizer class (default: SGD)
    - args.loss (optional) → loss function (default: CrossEntropyLoss)
    
    Args:
        trainer: Trainer instance to build
        args: Argument object with training config
        data_handle: DataHandle for transforms
        loss: Optional loss function override
    
    Returns:
        Configured trainer
    """
    
    # Get training hparams
    train_hparams = get_training_hparams(args)
    
    # Get optimizer class
    optimizer_class = getattr(args, 'optimizer', torch.optim.SGD)
    
    if isinstance(optimizer_class, str):
        optimizer_class = _get_optimizer(optimizer_class)
        
    
    # Build optimizer kwargs
    optimizer_kwargs = {
        'lr': train_hparams.learning_rate,
        'momentum': train_hparams.momentum,
        'weight_decay': train_hparams.weight_decay,
    }
    
    # Get loss function
    if loss is None:
        loss = torch.nn.CrossEntropyLoss(reduction="sum").to('cuda')
    
    # Build trainer
    trainer.build(
        optimizer=optimizer_class,
        optimizer_kwargs=optimizer_kwargs,
        handle=data_handle,
        loss=loss,
        scale_loss=train_hparams.scale_loss,
        gradient_clipnorm=train_hparams.gradient_clipnorm,
    )
    
    return trainer


__all__ = [
    "get_trainer",
    "build_trainer",
    "get_training_hparams",
    "TrainingHparams",
    "BaseCNNTrainer",
    "BaseIMP",
    "VGG_CNN",
    "VGG_IMP",
    "ResNet_CNN",
    "ResNet_IMP",
    "ResNet50_CNN",
    "ResNet50_IMP",
]