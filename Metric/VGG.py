import torch
from torch import nn
from torch.nn import functional as F

from TicketModels import TicketCNN
from LotteryLayers import LotteryConv2D, LotteryDense

class VGG19(TicketCNN):

    def __init__(self, args, input_channels = 3):
        super(VGG19, self).__init__()

        # Block 1
        self.block1_conv1 = LotteryConv2D(input_channels, 64, (3, 3), (1, 1))
        self.block1_relu1 = nn.ReLU()
        self.block1_conv2 = LotteryConv2D(64, 64, (3, 3), (1, 1))
        self.block1_norm = nn.BatchNorm2d(64)
        self.block1_relu2 = nn.ReLU()
        self.block1_pool = nn.MaxPool2d((2, 2), (2, 2))

        # Block 2
        self.block2_conv1 = LotteryConv2D(64, 128, (3, 3), (1, 1))
        self.block2_relu1 = nn.ReLU()
        self.block2_conv2 = LotteryConv2D(128, 128, (3, 3), (1, 1))
        self.block2_norm = nn.BatchNorm2d(128)
        self.block2_relu2 = nn.ReLU()
        self.block2_pool = nn.MaxPool2d((2, 2), (2, 2))


        # Block 3
        self.block3_conv1 = LotteryConv2D(128, 256, (3, 3), (1, 1))
        self.block3_relu1 = nn.ReLU()