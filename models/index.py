from torch.nn.parallel import DistributedDataParallel as DDP

from .base import BaseModel
from .resnet import resnet_imagenet, resnet_cifar
from .vgg import vgg 
from .LotteryLayers import Lottery 
from .hparams import get_model_hparams, ModelHparams, MODEL_CONFIGS

"""
Model index providing centralized model creation with args-driven hparams.
"""


def get_model(args, use_ddp: bool = True):
    """
    Create a model from args with automatic DDP wrapping if requested.
    
    Args flow:
    - args.model (required) → determines model architecture
    - args.dataset (required) → determines output features
    - args.rank, args.world_size (required for DDP)
    - args.custom_init, args.bn_track, args.dropout, etc. (optional)
    
    Args:
        args: Argument object with model/dataset/training config
        use_ddp: Whether to wrap model in DDP (default: True if world_size > 1)
    
    Returns:
        Model instance (optionally wrapped in DDP)
    """
    
    # Build model hparams from args
    hparams = get_model_hparams(args)
    
    # Create model based on type
    model = None
    if hparams.model_name.startswith("vgg"):
        model = vgg(
            rank=args.rank,
            world_size=args.world_size,
            depth=hparams.depth,
            outfeatures=hparams.outfeatures,
            inchannels=hparams.inchannels,
            custom_init=hparams.custom_init,
            conv_bias=hparams.conv_bias,
            bn_track=hparams.bn_track,
        )
    elif hparams.model_name.startswith("resnet"):
        # Determine if cifar or imagenet variant
        model_config = MODEL_CONFIGS[hparams.model_name]
        resnet_fn = resnet_cifar if model_config.get("type") == "cifar" else resnet_imagenet
        
        model = resnet_fn(
            rank=args.rank,
            world_size=args.world_size,
            depth=hparams.depth,
            outfeatures=hparams.outfeatures,
            inchannels=hparams.inchannels,
            custom_init=hparams.custom_init,
            bn_track=hparams.bn_track,
            dropout=hparams.dropout,
        )
    else:
        raise ValueError(f"Unknown model type: {hparams.model_name}")
    
    # Move to CUDA
    model = model.cuda()
    
    # Wrap in DDP if requested and world_size > 1
    if use_ddp and args.world_size > 1:
        model = DDP(
            model,
            device_ids=[args.rank],
            output_device=args.rank,
            gradient_as_bucket_view=True
        )
    
    return model


def get_model_non_ddp(args):
    """Create a model without DDP wrapping."""
    return get_model(args, use_ddp=False)


__all__ = [
    "BaseModel",
    "Lottery",
    "vgg",
    "resnet_imagenet",
    "resnet_cifar",
    "get_model",
    "get_model_non_ddp",
    "get_model_hparams",
    "ModelHparams",
]