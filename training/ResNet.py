import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from collections import defaultdict
import time

from training.base import BaseIMP, BaseCNNTrainer

from utils.serialization_utils import read_tensor, save_tensor

import math
import os
import sys
import gc

class ResNet50_CNN(BaseCNNTrainer):
    
    warmup_epochs = 5

    def post_epoch_hook(self, epoch, EPOCHS):
        if (epoch + 1) == 29 or (epoch + 1) == 59 or (epoch + 1) == 79: # Epochs 80, 120
            self.scale_learning_rate(0.1)
        return 
    
class ResNet50_IMP(BaseIMP):

    warmup_epochs = 5

    def post_epoch_hook(self, epoch, EPOCHS):
        if (epoch + 1) == 29 or (epoch + 1) == 59 or (epoch + 1) == 79: # Epochs 80, 120
            self.scale_learning_rate(0.1)
        return 


class ResNet_CNN(BaseCNNTrainer):
    def post_epoch_hook(self, epoch, EPOCHS):
        if (epoch + 1) == 79 or (epoch + 1) == 119: # Epochs 80, 120
            self.scale_learning_rate(0.1)
        return 
    
class ResNet_IMP(BaseIMP):
    def post_epoch_hook(self, epoch, EPOCHS):
        if epoch == 78 or epoch == 118: # Epochs 80, 120
            self.scale_learning_rate(0.1)
        return 
