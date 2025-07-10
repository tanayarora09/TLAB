import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from collections import defaultdict
import time

from training.base import CNN_DGTS, BaseIMP, BaseCNNTrainer

from utils.serialization_utils import read_tensor, save_tensor

import math
import os
import sys
import gc

class ResNet_CNN(BaseCNNTrainer):
    def post_epoch_hook(self, epoch, EPOCHS):
        if (epoch + 1) == 79 or (epoch + 1) == 119: # Epochs 80, 120
            self.reduce_learning_rate(10)
        return 
    
class ResNet_IMP(BaseIMP):
    def post_epoch_hook(self, epoch, EPOCHS):
        if epoch == 78 or epoch == 118: # Epochs 80, 120
            self.reduce_learning_rate(10)
        return 
