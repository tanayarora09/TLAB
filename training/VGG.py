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

class VGG_CNN(BaseCNNTrainer):
    def post_epoch_hook(self, epoch, EPOCHS):
        if (epoch + 1) == 79 or (epoch + 1) == 119: # Epochs 80, 120
            self.reduce_learning_rate(10)
        return 
    
    def post_step_hook(self, x, y, _, step, **kwargs):
        """if step % 370 == 0 and step != 0:
                print(x.view(-1)[x.numel()//4:3 * x.numel()//4].norm(2))"""
        return

class VGG_POC(BaseIMP):

    def __init__(self, model: torch.nn.parallel.DistributedDataParallel, rank: int):
        super(VGG_POC, self).__init__(model, rank)
        self.IsMetricRoot = rank == 1

    def build(self, *args, tickets_dict: dict = None, **kwargs):
        super().build(*args, **kwargs)
        self.ACTS = tickets_dict != None
        self.TICKETS = tickets_dict # {IMP_ITERATION - 1: (TICKET, SPARSITY_D)}
        self.activation_log = defaultdict(defaultdict) # {IMP_ITERATION: {Iteration:{Norm: (IMP KL, RAND KL), Base: (IMP KL, RAND KL)}}}

        #self.constant = 0

        self._hooks = list()

        self.__CURR_IMP_ITER = 0
        self.act_w = list()

    ### ----------------------------------------------------------- METRICS ---------------------------------------------------------------------------------------


    def Kullback_Leibler(self, input: torch.Tensor, target: torch.Tensor, eps: float = 1e-10): # + 1e-10 for numerical stability
        """
        Input and Target Not In Log Space, Non-negative, Sum to 1
        """
        return F.kl_div(input, target, reduction = "batchmean")
    
    def Hellinger(self, input: torch.Tensor, target: torch.Tensor):
        """
        Input and Target Non-negative, Sum to 1.
        """
        return torch.norm((input.sqrt() - target.sqrt()), p = 2 ).div(
            torch.sqrt(torch.as_tensor(2.0, dtype = target.dtype, device = target.device)))


    ### ----------------------------------------------------------- ACTIVATION CAPTURING ---------------------------------------------------------------------------------------

    
    def collect_activations_and_test(self, x: torch.Tensor, iteration) -> None:
        with torch.no_grad():
            
            if self.ACTS: 
                
                with torch.random.fork_rng(devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"], enabled = True):
                    
                    self.m.eval()
                    
                    full_activations_base = torch.cat(self.act_w)
                    full_activations_base += 1e-10
                    full_activations_base.div_(full_activations_base.sum())

                    for act in self.act_w:
                        act += 1e-10
                        act.div_(act.sum())

                    full_activations_norm = torch.cat(self.act_w)
                    full_activations_norm += 1e-10
                    full_activations_norm.div_(full_activations_norm.sum())
                    self.clear_act_captures()
                    
                    original_ticket = self.mm.export_ticket_cpu()

                    # IMP_TICKET
                    self.mm.set_ticket(self.TICKETS[self.__CURR_IMP_ITER][0])
                    with torch.no_grad(): 
                        self.mm(x)

                    curr_activations_base = torch.cat(self.act_w)
                    curr_activations_base += 1e-10
                    curr_activations_base.div_(curr_activations_base.sum())
                    curr_activations_base.log_()

                    for act in self.act_w:
                        act += 1e-10
                        act.div_(act.sum())

                    curr_activations_norm = torch.cat(self.act_w)
                    curr_activations_norm += 1e-10
                    curr_activations_norm.div_(curr_activations_norm.sum()) # Regularize to prob distribution
                    curr_activations_norm.log_()
                    self.clear_act_captures()

                    base_ikl = self.Kullback_Leibler(curr_activations_base, full_activations_base)
                    norm_ikl = self.Kullback_Leibler(curr_activations_norm, full_activations_norm)
                    
                    del curr_activations_norm
                    del curr_activations_base

                    #RAND_TICKET 
                    self.mm.set_ticket(self.TICKETS[self.__CURR_IMP_ITER][1])

                    with torch.no_grad(): 
                        self.mm(x)

                    curr_activations_base = torch.cat(self.act_w)
                    curr_activations_base += 1e-10
                    curr_activations_base.div_(curr_activations_base.sum())
                    curr_activations_base.log_()

                    for act in self.act_w:
                        act += 1e-10
                        act.div_(act.sum())

                    curr_activations_norm = torch.cat(self.act_w)
                    curr_activations_norm += 1e-10
                    curr_activations_norm.div_(curr_activations_norm.sum()) # Regularize to prob distribution
                    curr_activations_norm.log_()
                    self.clear_act_captures()

                    base_rkl = self.Kullback_Leibler(curr_activations_base, full_activations_base)
                    norm_rkl = self.Kullback_Leibler(curr_activations_norm, full_activations_norm)

                    self.activation_log[self.__CURR_IMP_ITER][iteration] = {"Base": (base_ikl.item(), base_rkl.item()),
                                                                       "Norm": (norm_ikl.item(), norm_rkl.item())}

                    #Reset
                    self.mm.set_ticket(original_ticket)

                    self.m.train()

        return

    def disable_act_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
        return 

    def init_act_hooks(self):
        if len(self._hooks) != 0: return
        for n, block in self.mm.named_children():
            for name, layer in block.named_children():
                if name.endswith("relu"):
                    self._hooks.append(layer.register_forward_hook(self._activation_hook))
                elif name.endswith("fc"):
                    self._hooks.append(layer.register_forward_hook(self._fake_activation_hook))
        return

    def _activation_hook(self, module, input, output: torch.Tensor) -> None:
        self.act_w.append(output.detach().to(torch.float64).mean(dim = 0).view(-1))#.cpu().to(torch.float64).mean(dim = 0).view(-1))
        return
    
    def _fake_activation_hook(self, module, input, output: torch.Tensor) -> None:
        self.act_w.append(F.relu(output.detach()).to(torch.float64).mean(dim = 0).view(-1))#.cpu()).to(torch.float64).mean(dim = 0).view(-1))
        return

    def clear_act_captures(self) -> None:
        self.act_w.clear()

    ### ----------------------------------------------------------- HOOKS ---------------------------------------------------------------------------------------

    def pre_IMP_hook(self, name: str):
        self.__CURR_IMP_ITER = 0
        self.act_w = list()
        #open(f'./tmp/swap/activations/{name}.h5', 'w').close()
        #os.remove(f"./tmp/swap/activations/{self.NAME}.h5")

    def post_prune_hook(self, iteration: int, epochs_per_run: int):
        self.__CURR_IMP_ITER = iteration

    def post_train_hook(self):
        self.disable_act_hooks()
        return
    
    def post_epoch_hook(self, epoch, EPOCHS):
        if epoch == 78 or epoch == 118: # Epochs 80, 120
            self.reduce_learning_rate(10)
        return 

    def pre_step_hook(self, step, steps_per_epoch):
        if step % 8 == 0 and self.ACTS:
            self.init_act_hooks()

    def post_step_hook(self, x, y, _, iteration, step, steps_per_epoch, **kwargs):
        if step % 8 == 0 and self.ACTS:
            with torch.autocast('cuda', dtype = torch.float16, enabled = self.AMP):
                self.collect_activations_and_test(x, iteration)
            self.disable_act_hooks()