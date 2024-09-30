import torch
from torch import nn

from typing import Callable

class HReLU(nn.Module): ## "Hooked ReLU"; adds compatibility for forward hooks on DDP + torch.compile

    def __init__(self, *args, **kwargs):
        super(HReLU, self).__init__(*args, **kwargs)
        self.__HOOKS_ = list()
        self.isEnabled = False

    def forward(self, x):
        o = nn.functional.relu(x)
        if self.isEnabled:
            for H in self.__HOOKS_: H(self, x, o)
        return o
    
    def enable_fw_hooks(self):
        self.isEnabled = True

    def disable_fw_hooks(self):
        self.isEnabled = False

    def init_fw_hook(self, hook: Callable):
        self.__HOOKS_.append(hook)

    def remove_fw_hooks(self):
        self.__HOOKS_.clear()


class FakeHReLU(nn.Module): ## Does Not Transform Output; Captures Output

    def __init__(self, *args, **kwargs):
        super(FakeHReLU, self).__init__(*args, **kwargs)
        self.__HOOKS_ = list()
        self.isEnabled = False

    def forward(self, x):
        o = nn.functional.relu(x)
        if self.isEnabled:
            for H in self.__HOOKS_: H(self, x, o)
        return x
    
    def enable_fw_hooks(self):
        self.isEnabled = True

    def disable_fw_hooks(self):
        self.isEnabled = False

    def init_fw_hook(self, hook: Callable):
        self.__HOOKS_.append(hook)

    def remove_fw_hooks(self):
        self.__HOOKS_.clear()
