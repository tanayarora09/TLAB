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


class POC(BaseIMP):


    def build(self, *args, tickets_dict: dict = None, **kwargs):
        super().build(*args, **kwargs)
        self.IsMetricRoot = self.RANK == 1
        self.ACTS = tickets_dict != None
        self.TICKETS = tickets_dict # {IMP_ITERATION - 1: (TICKET, SPARSITY_D)}
        self.activation_log = defaultdict(defaultdict) # {IMP_ITERATION: {Epoch:(IMP KL, RAND KL)}}
        
        self.Ikl_tr = torch.as_tensor(0.0, dtype = torch.float64, device = "cuda")
        self.Rkl_tr = torch.as_tensor(0.0, dtype = torch.float64, device = "cuda")

        self.eIkl = 0.0 #torch.as_tensor(0.0, dtype = torch.float64, device = "cuda")
        self.eRkl = 0.0 #torch.as_tensor(0.0, dtype = torch.float64, device = "cuda")

        #self.constant = 0

        self._hooks = list()

        self.__CURR_IMP_ITER = 0
        self.__CURREPOCH = None 
        self.act_w = list()

    ### ----------------------------------------------------------- METRICS ---------------------------------------------------------------------------------------

    def reset_metrics(self):
        with torch.no_grad():
            super().reset_metrics()
            self.eIkl = 0.0
            self.eRkl = 0.0

    def reset_running_metrics(self):
        with torch.no_grad():
            super().reset_running_metrics()
            self.Ikl_tr.fill_(0.0)
            self.Rkl_tr.fill_(0.0)

    def _collect_metrics(self):
        with torch.no_grad():
            super()._collect_metrics()
            dist.all_reduce(self.Ikl_tr, op = dist.ReduceOp.SUM)
            dist.all_reduce(self.Rkl_tr, op = dist.ReduceOp.SUM)

    
    def transfer_metrics(self):
        with torch.no_grad():
            self._collect_metrics()
            self.eloss += self.loss_tr
            self.eacc += self.acc_tr
            self.ecount += self.count_tr
            self.eIkl += self.Ikl_tr.item()
            self.eRkl += self.Rkl_tr.item()
            self.reset_running_metrics()

    def Kullback_Leibler(self, input: torch.Tensor, target: torch.Tensor, eps: float = 1e-9): # + 1e-9 for numerical stability
        """
        Input and Target Not In Log Space, Non-negative, Sum to 1
        """
        return F.kl_div((input + eps).log(), target + eps, reduction = "batchmean")
    
    def Hellinger(self, input: torch.Tensor, target: torch.Tensor):
        """
        Input and Target Non-negative, Sum to 1.
        """
        return torch.norm((input.sqrt() - target.sqrt()), p = 2 ).div(
            torch.sqrt(torch.as_tensor(2.0, dtype = target.dtype, device = target.device)))


    ### ----------------------------------------------------------- ACTIVATION CAPTURING ---------------------------------------------------------------------------------------

    
    def collect_activations_and_test(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            
            if self.ACTS: 
                
                with torch.random.fork_rng(devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"], enabled = True):
                    
                    self.m.eval()
                    
                    full_activations = torch.cat(self.act_w)
                    full_activations.div_(full_activations.sum())
                    original_ticket = self.mm.export_ticket_cpu()

                    # IMP_TICKET
                    self.clear_act_captures()
                    self.mm.set_ticket(self.TICKETS[self.__CURR_IMP_ITER][0])
                    with torch.no_grad(): 
                        with self.m.no_sync():
                            self.m(x)

                    curr_activations = torch.cat(self.act_w)
                    curr_activations.div_(curr_activations.sum()) # Regularize to prob distribution

                    self.Ikl_tr += self.Kullback_Leibler(curr_activations, full_activations)

                    #RAND_TICKET 
                    self.clear_act_captures()
                    self.mm.set_ticket(self.TICKETS[self.__CURR_IMP_ITER][1])
                    with torch.no_grad(): 
                        with self.m.no_sync():
                            self.m(x)

                    curr_activations = torch.cat(self.act_w)
                    curr_activations.div_(curr_activations.sum())

                    self.Rkl_tr += self.Kullback_Leibler(curr_activations, full_activations)

                    #Reset
                    self.clear_act_captures()
                    self.mm.set_ticket(original_ticket)

                    self.m.train()
            
        
            #if (self.__CURR_IMP_ITER == 0 and self.__CURREPOCH == 0 and self.constant < 26 and self.RANK == 0): 
                #self.print(f"|| STEP {self.constant} || ACCURACY: {self.acc_tr.item()}; LOSS: {self.loss_tr.item()}; EACC: {self.eacc.item()}; ELOSS: {self.eloss.item()}", "gray")

                #tmp = list()
                #for name, param in self.mm.named_parameters():
                    #tmp.append(param.grad.detach().view(-1))
                
                #self.print(f"Gradient Norm: {torch.cat(tmp).norm(2).item()}", "gray")

            #self.constant += 1

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
        self.act_w.append(output.detach().clone().to(torch.float64).mean(dim = 0).view(-1))#.cpu().to(torch.float64).mean(dim = 0).view(-1))
        return
    
    def _fake_activation_hook(self, module, input, output: torch.Tensor) -> None:
        self.act_w.append(F.relu(output.detach().clone()).to(torch.float64).mean(dim = 0).view(-1))#.cpu()).to(torch.float64).mean(dim = 0).view(-1))

    def clear_act_captures(self) -> None:
        self.act_w.clear()

    ### ----------------------------------------------------------- HOOKS ---------------------------------------------------------------------------------------

    def pre_IMP_hook(self, name: str):
        self.__CURR_IMP_ITER = 0
        self.__CURREPOCH = None 
        self.act_w = list()
        #open(f'./tmp/swap/activations/{name}.h5', 'w').close()
        #os.remove(f"./tmp/swap/activations/{self.NAME}.h5")

    def post_prune_hook(self, iteration: int, epochs_per_run: int):
        self.__CURR_IMP_ITER = iteration

    def pre_epoch_hook(self, epoch: int):
        self.__CURREPOCH = epoch
        if self.ACTS: self.init_act_hooks()

    def post_train_hook(self):
        self.disable_act_hooks()
        self.activation_log[self.__CURR_IMP_ITER][self.__CURREPOCH] = (self.eIkl / self.ecount.detach().item(), self.eRkl / self.ecount.detach().item())
        return

    def post_step_hook(self, x, y, _):
        with torch.autocast('cuda', dtype = torch.float16, enabled = self.AMP):
            self.collect_activations_and_test(x)