from torch.nn.parallel import DistributedDataParallel as DDP

from .base import MaskedModel
from .resnet import resnet
from .vgg import vgg 
from .small_cnn import small_cnn
from .LotteryLayers import Lottery 
from .hparams import get_model_hparams, ModelHparams

"""
Model index providing centralized model creation with args-driven hparams.
"""

FAMILY_TO_CLASS = {"vgg": vgg, "resnet": resnet, "scnn": small_cnn}


def get_model(args, state = None, ticket = None, use_ddp: bool = True):
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

    inp = {"rank": args.rank, "world_size": args.world_size, "depth": hparams.depth,
           "outfeatures": hparams.outfeatures, "inchannels": hparams.inchannels, 
           "custom_init": hparams.custom_init, "conv_bias": hparams.conv_bias, 
           "bn_track": hparams.bn_track, "is_imagenet": "imagenet" in args.dataset}
    
    if hparams.family in FAMILY_TO_CLASS.keys(): 
        model = FAMILY_TO_CLASS[hparams.family](**inp)

    else:
        raise ValueError(f"Unknown model type: {hparams.model_name}")
    
    # Move to CUDA
    model = model.cuda()
    
    if ticket is not None:
        model.set_ticket(ticket)

    # Wrap in DDP if requested and world_size > 1
    if use_ddp and args.world_size > 1:
        model = DDP(
            model,
            device_ids=[args.rank],
            output_device=args.rank,
            gradient_as_bucket_view=True
        )

    if state is not None:
        model.load_state_dict(state)
    
    return model


__all__ = [
    "MaskedModel",
    "Lottery",
    "get_model",
    "get_model_hparams",
    "ModelHparams",
]