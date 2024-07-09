import torch
from torch import nn

from LotteryLayers import LotteryConv2D, LotteryDense

class VGG19(nn.Module):

    def __init__(self, input_channels = 3):
        super(VGG19, self).__init__()

        # Block 1
        self.block1_conv1 = LotteryConv2D(input_channels, 64, (3, 3), (1, 1), padding="same")
        self.block1_norm1 = nn.BatchNorm2d(64)
        self.block1_relu1 = nn.ReLU()
        self.block1_conv2 = LotteryConv2D(64, 64, (3, 3), (1, 1), padding="same")
        self.block1_norm2 = nn.BatchNorm2d(64)
        self.block1_relu2 = nn.ReLU()
        self.block1_pool = nn.MaxPool2d((2, 2), (2, 2))
        
        # Block 2
        self.block2_conv1 = LotteryConv2D(64, 128, (3, 3), (1, 1), padding="same")
        self.block2_norm1 = nn.BatchNorm2d(128)
        self.block2_relu1 = nn.ReLU()
        self.block2_conv2 = LotteryConv2D(128, 128, (3, 3), (1, 1), padding="same")
        self.block2_norm2 = nn.BatchNorm2d(128)
        self.block2_relu2 = nn.ReLU()
        self.block2_pool = nn.MaxPool2d((2, 2), (2, 2))

        # Block 3
        self.block3_conv1 = LotteryConv2D(128, 256, (3, 3), (1, 1), padding="same")
        self.block3_norm1 = nn.BatchNorm2d(256)
        self.block3_relu1 = nn.ReLU()
        self.block3_conv2 = LotteryConv2D(256, 256, (3, 3), (1, 1), padding="same")
        self.block3_norm2 = nn.BatchNorm2d(256)
        self.block3_relu2 = nn.ReLU()
        self.block3_conv3 = LotteryConv2D(256, 256, (3, 3), (1, 1), padding="same")
        self.block3_norm3 = nn.BatchNorm2d(256)
        self.block3_relu3 = nn.ReLU()
        self.block3_conv4 = LotteryConv2D(256, 256, (3, 3), (1, 1), padding="same")
        self.block3_norm4 = nn.BatchNorm2d(256)
        self.block3_relu4 = nn.ReLU()
        self.block3_pool = nn.MaxPool2d((2, 2), (2, 2))

        # Block 4
        self.block4_conv1 = LotteryConv2D(256, 512, (3, 3), (1, 1), padding="same")
        self.block4_norm1 = nn.BatchNorm2d(512)
        self.block4_relu1 = nn.ReLU()
        self.block4_conv2 = LotteryConv2D(512, 512, (3, 3), (1, 1), padding="same")
        self.block4_norm2 = nn.BatchNorm2d(512)
        self.block4_relu2 = nn.ReLU()
        self.block4_conv3 = LotteryConv2D(512, 512, (3, 3), (1, 1), padding="same")
        self.block4_norm3 = nn.BatchNorm2d(512)
        self.block4_relu3 = nn.ReLU()
        self.block4_conv4 = LotteryConv2D(512, 512, (3, 3), (1, 1), padding="same")
        self.block4_norm4 = nn.BatchNorm2d(512)
        self.block4_relu4 = nn.ReLU()
        self.block4_pool = nn.MaxPool2d((2, 2), (2, 2))

        # Block 5
        self.block5_conv1 = LotteryConv2D(512, 512, (3, 3), (1, 1), padding="same")
        self.block5_norm1 = nn.BatchNorm2d(512)
        self.block5_relu1 = nn.ReLU()
        self.block5_conv2 = LotteryConv2D(512, 512, (3, 3), (1, 1), padding="same")
        self.block5_norm2 = nn.BatchNorm2d(512)
        self.block5_relu2 = nn.ReLU()
        self.block5_conv3 = LotteryConv2D(512, 512, (3, 3), (1, 1), padding="same")
        self.block5_norm3 = nn.BatchNorm2d(512)
        self.block5_relu3 = nn.ReLU()
        self.block5_conv4 = LotteryConv2D(512, 512, (3, 3), (1, 1), padding="same")
        self.block5_norm4 = nn.BatchNorm2d(512)
        self.block5_relu4 = nn.ReLU()
        self.block5_pool = nn.MaxPool2d((2, 2), (2, 2))

        # Output
        self.GAP_FC = nn.AdaptiveAvgPool2d((1, 1))
        self.Dropout = nn.Dropout(0.5)
        self.NORM_OUT = nn.BatchNorm1d(512)
        self.FC_OUT = nn.Linear(512, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Block 1
        x = self.block1_conv1(x)
        x = self.block1_norm1(x)
        x = self.block1_relu1(x)
        x = self.block1_conv2(x)
        x = self.block1_norm2(x)
        x = self.block1_relu2(x)
        x = self.block1_pool(x)

        # Block 2
        x = self.block2_conv1(x)
        x = self.block2_norm1(x)
        x = self.block2_relu1(x)
        x = self.block2_conv2(x)
        x = self.block2_norm2(x)
        x = self.block2_relu2(x)
        x = self.block2_pool(x)

        # Block 3
        x = self.block3_conv1(x)
        x = self.block3_norm1(x)
        x = self.block3_relu1(x)
        x = self.block3_conv2(x)
        x = self.block3_norm2(x)
        x = self.block3_relu2(x)
        x = self.block3_conv3(x)
        x = self.block3_norm3(x)
        x = self.block3_relu3(x)
        x = self.block3_conv4(x)
        x = self.block3_norm4(x)
        x = self.block3_relu4(x)
        x = self.block3_pool(x)

        # Block 4
        x = self.block4_conv1(x)
        x = self.block4_norm1(x)
        x = self.block4_relu1(x)
        x = self.block4_conv2(x)
        x = self.block4_norm3(x)
        x = self.block4_relu2(x)
        x = self.block4_conv3(x)
        x = self.block4_norm3(x)
        x = self.block4_relu3(x)
        x = self.block4_conv4(x)
        x = self.block4_norm4(x)
        x = self.block4_relu4(x)
        x = self.block4_pool(x)

        # Block 5
        x = self.block5_conv1(x)
        x = self.block5_norm1(x)
        x = self.block5_relu1(x)
        x = self.block5_conv2(x)
        x = self.block5_norm2(x)
        x = self.block5_relu2(x)
        x = self.block5_conv3(x)
        x = self.block5_norm3(x)
        x = self.block5_relu3(x)
        x = self.block5_conv4(x)
        x = self.block5_norm4(x)
        x = self.block5_relu4(x)
        x = self.block5_pool(x)

        #Output
        x = self.GAP_FC(x)
        x.squeeze_()
        x = self.Dropout(x)
        x = self.NORM_OUT(x)
        x = self.FC_OUT(x)

        return x