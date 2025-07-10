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
    
class VGG_IMP(BaseIMP):
    def post_epoch_hook(self, epoch, EPOCHS):
        if (epoch + 1) == 79 or (epoch + 1) == 119: # Epochs 80, 120
            self.reduce_learning_rate(10)
        return 

class VGG_POC(BaseIMP):

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
                    eps = 1e-11 * full_activations_base.masked_fill(full_activations_base == 0, float('inf')).min()
                    full_activations_base += eps
                    full_activations_base.div_(full_activations_base.sum())

                    for act in self.act_w:
                        act += eps
                        act.div_(act.sum())

                    full_activations_norm = torch.cat(self.act_w)
                    full_activations_norm += eps
                    full_activations_norm.div_(full_activations_norm.sum())
                    self.clear_act_captures()
                    
                    original_ticket = self.mm.export_ticket_cpu()

                    # IMP_TICKET
                    self.mm.set_ticket(self.TICKETS[self.__CURR_IMP_ITER][0])
                    with torch.no_grad(): 
                        self.mm(x)

                    curr_activations_base = torch.cat(self.act_w)
                    curr_activations_base += 1e-11
                    curr_activations_base.div_(curr_activations_base.sum())
                    curr_activations_base.log_()

                    for act in self.act_w:
                        act += 1e-11
                        act.div_(act.sum())

                    curr_activations_norm = torch.cat(self.act_w)
                    curr_activations_norm += 1e-11
                    curr_activations_norm.div_(curr_activations_norm.sum()) # Regularize to prob distribution
                    curr_activations_norm.log_()
                    self.clear_act_captures()

                    base_ikl = self.Kullback_Leibler(curr_activations_base, full_activations_base)
                    norm_ikl = self.Kullback_Leibler(curr_activations_norm, full_activations_norm)
                    
                    del curr_activations_norm
                    del curr_activations_base

                    #RAND_TICKET 
                    rand_ticket = self.TICKETS[self.__CURR_IMP_ITER][1]
                    self.mm.set_ticket(rand_ticket)

                    with torch.no_grad(): 
                        self.mm(x)

                    curr_activations_base = torch.cat(self.act_w)
                    curr_activations_base += 1e-11
                    curr_activations_base.div_(curr_activations_base.sum())
                    curr_activations_base.log_()

                    for act in self.act_w:
                        act += 1e-11
                        act.div_(act.sum())

                    curr_activations_norm = torch.cat(self.act_w)
                    curr_activations_norm += 1e-11
                    curr_activations_norm.div_(curr_activations_norm.sum()) # Regularize to prob distribution
                    curr_activations_norm.log_()
                    self.clear_act_captures()

                    base_rkl = self.Kullback_Leibler(curr_activations_base, full_activations_base)
                    norm_rkl = self.Kullback_Leibler(curr_activations_norm, full_activations_norm)

                    del curr_activations_norm
                    del curr_activations_base

                    #MG_TICKET 
                    self.mm.reset_ticket()
                    self.mm.prune_by_mg(rand_ticket.sum()/rand_ticket.numel(), 
                                        iteration = 1, distributed = False)

                    with torch.no_grad(): 
                        self.mm(x)

                    curr_activations_base = torch.cat(self.act_w)
                    curr_activations_base += 1e-11
                    curr_activations_base.div_(curr_activations_base.sum())
                    curr_activations_base.log_()

                    for act in self.act_w:
                        act += 1e-11
                        act.div_(act.sum())

                    curr_activations_norm = torch.cat(self.act_w)
                    curr_activations_norm += 1e-11
                    curr_activations_norm.div_(curr_activations_norm.sum()) # Regularize to prob distribution
                    curr_activations_norm.log_()
                    self.clear_act_captures()

                    base_mkl = self.Kullback_Leibler(curr_activations_base, full_activations_base)
                    norm_mkl = self.Kullback_Leibler(curr_activations_norm, full_activations_norm)

                    #EXPORT

                    self.activation_log[self.__CURR_IMP_ITER][iteration] = {"Base": (base_ikl.item(), base_rkl.item(), base_mkl.item()),
                                                                       "Norm": (norm_ikl.item(), norm_rkl.item(), norm_mkl.item())}


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
        if step % 2 == 0 and self.ACTS:
            self.init_act_hooks()

    def post_step_hook(self, x, y, iteration, step, steps_per_epoch, **kwargs):
        if step % 2 == 0 and self.ACTS:
            with torch.autocast('cuda', dtype = torch.float16, enabled = self.AMP):
                self.collect_activations_and_test(x, iteration)
            self.disable_act_hooks()

class VGG_POC_FULL(BaseIMP):

    def build(self, *args, tickets_dict: dict = None, **kwargs):
        super().build(*args, **kwargs)
        self.ACTS = tickets_dict != None
        self.TICKETS = tickets_dict # \\ {sparsity: (TICKET)}
        self._mgtickets = dict()
        self._negtickets = dict()
        self.activation_log = defaultdict(defaultdict) # {IMP_ITERATION: {Iteration:{TICKET: [linearglobal, softglobal, linearlayer, softlayer]}}}

        #self.constant = 0

        self._norm = torch.as_tensor(0.0, dtype = torch.float64, device = 'cuda')
        self._norm.requires_grad_(False)


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

    
    """def collect_activations_and_test(self, x: torch.Tensor, iteration) -> None:
        with torch.no_grad():
            
            if self.ACTS: 
                
                with torch.random.fork_rng(devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"], enabled = True):
                    
                    self.m.eval()
                    
                    full_acts = list() #LinearGlobal, SoftmaxGlobal, LinearLayer, SoftmaxLayer

                    #print("MAKING FULL ACTS")

                    full_activations = torch.cat(self.act_w)

                    eps = full_activations.masked_fill(full_activations == 0, float('inf')).min()

                    full_activations += eps
                    full_activations.div_(full_activations.sum())
                    full_acts.append((full_activations + eps).log())
                    full_acts.append((full_activations + eps).log_softmax(0))

                    for act in self.act_w:
                        act += eps
                        act.div_(act.sum())

                    full_activations = torch.cat(self.act_w)
                    full_activations += eps
                    full_activations.div_(full_activations.sum())
                    full_acts.append((full_activations + eps).log())
                    full_acts.append((full_activations + eps).log_softmax(0))
                    self.clear_act_captures()

                    #print(f"MADE FULL ACTS: {len(full_acts)}")

                    original_ticket = self.mm.export_ticket_cpu()

                    for spd in self.TICKETS.keys():

                        #print(f"RUNNING ON SPARSITY {spd}")

                        # IMP

                        ikl = list()

                        self.mm.set_ticket(self.TICKETS[spd][0])
                        with torch.no_grad(): self.mm(x)

                        curr_activations = torch.cat(self.act_w)
                        curr_activations += eps
                        curr_activations.div_(curr_activations.sum())
                        ikl.append(F.kl_div((curr_activations + eps).log(), full_acts[0], reduction = 'batchmean', log_target = True))
                        ikl.append(F.kl_div((curr_activations + eps).log_softmax(0), full_acts[1], reduction = 'batchmean', log_target = True))

                        #print("ADDED")

                        for act in self.act_w:
                            act += eps
                            act.div_(act.sum())

                        curr_activations = torch.cat(self.act_w)
                        curr_activations += eps
                        curr_activations.div_(curr_activations.sum()) # Regularize to prob distribution
                        ikl.append(F.kl_div((curr_activations + eps).log(), full_acts[2], reduction = 'batchmean', log_target = True))
                        ikl.append(F.kl_div((curr_activations + eps).log_softmax(0), full_acts[3], reduction = 'batchmean', log_target = True))
                        self.clear_act_captures()

                        
                        # Layerwise

                        lkl = list()

                        self.mm.set_ticket(self.TICKETS[spd][1])
                        with torch.no_grad(): self.mm(x)

                        curr_activations = torch.cat(self.act_w)
                        curr_activations += eps
                        curr_activations.div_(curr_activations.sum())
                        lkl.append(F.kl_div((curr_activations + eps).log(), full_acts[0], reduction = 'batchmean', log_target = True))
                        lkl.append(F.kl_div((curr_activations + eps).log_softmax(0), full_acts[1], reduction = 'batchmean', log_target = True))

                        for act in self.act_w:
                            act += eps
                            act.div_(act.sum())

                        curr_activations = torch.cat(self.act_w)
                        curr_activations += eps
                        curr_activations.div_(curr_activations.sum()) # Regularize to prob distribution
                        lkl.append(F.kl_div((curr_activations + eps).log(), full_acts[2], reduction = 'batchmean', log_target = True))
                        lkl.append(F.kl_div((curr_activations + eps).log_softmax(0), full_acts[3], reduction = 'batchmean', log_target = True))
                        self.clear_act_captures()                    


                        # RandomSame

                        rdkl = list()

                        self.mm.set_ticket(self.TICKETS[spd][2])
                        with torch.no_grad(): self.mm(x)

                        curr_activations = torch.cat(self.act_w)
                        curr_activations += eps
                        curr_activations.div_(curr_activations.sum())
                        rdkl.append(F.kl_div((curr_activations + eps).log(), full_acts[0], reduction = 'batchmean', log_target = True))
                        rdkl.append(F.kl_div((curr_activations + eps).log_softmax(0), full_acts[1], reduction = 'batchmean', log_target = True))

                        for act in self.act_w:
                            act += eps
                            act.div_(act.sum())

                        curr_activations = torch.cat(self.act_w)
                        curr_activations += eps
                        curr_activations.div_(curr_activations.sum()) # Regularize to prob distribution
                        rdkl.append(F.kl_div((curr_activations + eps).log(), full_acts[2], reduction = 'batchmean', log_target = True))
                        rdkl.append(F.kl_div((curr_activations + eps).log_softmax(0), full_acts[3], reduction = 'batchmean', log_target = True))
                        self.clear_act_captures()      

                        # Random

                        rkl = list()

                        self.mm.set_ticket(self.TICKETS[spd][3])
                        with torch.no_grad(): self.mm(x)

                        curr_activations = torch.cat(self.act_w)
                        curr_activations += eps
                        curr_activations.div_(curr_activations.sum())
                        rkl.append(F.kl_div((curr_activations + eps).log(), full_acts[0], reduction = 'batchmean', log_target = True))
                        rkl.append(F.kl_div((curr_activations + eps).log_softmax(0), full_acts[1], reduction = 'batchmean', log_target = True))

                        for act in self.act_w:
                            act += eps
                            act.div_(act.sum())

                        curr_activations = torch.cat(self.act_w)
                        curr_activations += eps
                        curr_activations.div_(curr_activations.sum()) # Regularize to prob distribution
                        rkl.append(F.kl_div((curr_activations + eps).log(), full_acts[2], reduction = 'batchmean', log_target = True))
                        rkl.append(F.kl_div((curr_activations + eps).log_softmax(0), full_acts[3], reduction = 'batchmean', log_target = True))
                        self.clear_act_captures()      

                        #MG_TICKET 
                        

                        mkl = list()

                        self.mm.set_ticket(self._mgtickets[spd])
                        with torch.no_grad(): self.mm(x)

                        curr_activations = torch.cat(self.act_w)
                        curr_activations += eps
                        curr_activations.div_(curr_activations.sum())
                        mkl.append(F.kl_div((curr_activations + eps).log(), full_acts[0], reduction = 'batchmean', log_target = True))
                        mkl.append(F.kl_div((curr_activations + eps).log_softmax(0), full_acts[1], reduction = 'batchmean', log_target = True))

                        for act in self.act_w:
                            act += eps
                            act.div_(act.sum())

                        curr_activations = torch.cat(self.act_w)
                        curr_activations += eps
                        curr_activations.div_(curr_activations.sum()) # Regularize to prob distribution
                        
                        mkl.append(F.kl_div((curr_activations + eps).log(), full_acts[2], reduction = 'batchmean', log_target = True))
                        mkl.append(F.kl_div((curr_activations + eps).log_softmax(0), full_acts[3], reduction = 'batchmean', log_target = True))
                        self.clear_act_captures()      

                        #EXPORT

                        self.activation_log[spd][iteration] = {'IMP': [kl.item() for kl in ikl], 'LAYER_RANDOM': [kl.item() for kl in lkl], 
                                                               'CLOSE_RANDOM': [kl.item() for kl in rdkl],
                                                               'TRUE_RANDOM': [kl.item() for kl in rkl], 
                                                               "REAL_TIME_MAGNITUDE": [kl.item() for kl in mkl]}

                        if not all([kl > 0 for key in self.activation_log[spd][iteration].keys() for kl in self.activation_log[spd][iteration][key]]):
                            print(self.activation_log[spd][iteration])
                            print(full_activations.sum(), full_activations.ge(0).all(), )
                        
                        #print(self.activation_log[spd][iteration])


                    #Reset
                    self.mm.set_ticket(original_ticket)

                    self.m.train()

        return"""

    def _grad_norm(self, x, y):
        """
        Assumes model ticket already set.
        """
        #with torch.autocast("cuda", dtype = torch.float16, enabled = True):
        loss = self.criterion(self.mm(x), y)
        grad_w = torch.autograd.grad(loss, self.mm.parameters(), allow_unused = True)
        grad_max_vals = [g.detach().abs().amax() for g in grad_w]
        max_val = torch.stack(grad_max_vals).amax().clamp(min=1e-30)
        grad_w = [torch.nan_to_num(grad.to(torch.float64)/max_val, nan = 0.0) for grad in grad_w]
        norms = list(torch._foreach_norm(grad_w, 2.0))
        total_norm = torch.norm(torch.stack(norms), 2.0) * max_val
        if torch.isinf(total_norm): raise ValueError("INF")
        if torch.isnan(total_norm): raise ValueError("NAN")
        return total_norm


    def _mse_loss(self, x, full_activations): # NOT MEANED --- As with Angular Distillation, Performs division by norm then computes squared error.
        self.mm(x)
        for act in self.act_w: 
            torch.nan_to_num_(act.div_(act.abs().amax(dim = 1, keepdim = True)), nan = 0.0)
            act.div_(act.norm(2, dim = 1, keepdim = True))
        difference = torch.cat(self.act_w, dim = 1) - full_activations
        self.clear_act_captures()
        return (difference * difference).sum()#F.mse_loss(curr_activations, full_activations, reduction = "sum")

    def collect_activations_and_test(self, x: torch.Tensor, y: torch.Tensor, iteration) -> None:
        with torch.no_grad():
            
            if self.ACTS: 
                
                with torch.random.fork_rng(devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"], enabled = True):
                    
                    self.init_act_hooks()
                    self.m.eval()
                    
                    self.mm(x)
                    for act in self.act_w: 
                        torch.nan_to_num_(act.div_(act.abs().amax(dim = 1, keepdim = True)), nan = 0.0)
                        act.div_(act.norm(2, dim = 1, keepdim = True))
                    full_activations = torch.cat(self.act_w, dim = 1)
                    self.clear_act_captures()

                    #full_acts = list() #LinearGlobal, SoftmaxGlobal, LinearLayer, SoftmaxLayer

                    #print("MAKING FULL ACTS")

                    #full_activations = torch.cat(self.act_w)

                    eps = 5e-12#full_activations.masked_fill(full_activations == 0, float('inf')).min()

                    """full_activations += eps
                    full_activations.div_(full_activations.sum())
                    full_acts.append((full_activations + eps).log())
                    full_acts.append((full_activations + eps).log_softmax(0))

                    for act in self.act_w:
                        act += eps
                        act.div_(act.sum())

                    full_activations = torch.cat(self.act_w)
                    full_activations += eps
                    full_activations.div_(full_activations.sum())
                    full_acts.append((full_activations + eps).log())
                    full_acts.append((full_activations + eps).log_softmax(0))
                    self.clear_act_captures()"""

                    #print(f"MADE FULL ACTS: {len(full_acts)}")

                    original_ticket = self.mm.export_ticket_cpu()

                    for spd in self.TICKETS.keys():

                        #print(f"RUNNING ON SPARSITY {spd}")

                        # IMP

                        ikl = list()

                        self.mm.set_ticket(self.TICKETS[spd][0])
                        
                        ikl.append(self._mse_loss(x, full_activations))
                        
                        # Layerwise

                        lkl = list()

                        self.mm.set_ticket(self.TICKETS[spd][1])

                        lkl.append(self._mse_loss(x, full_activations))

                        # RandomSame

                        rdkl = list()

                        self.mm.set_ticket(self.TICKETS[spd][2])
                        
                        rdkl.append(self._mse_loss(x, full_activations))


                        # Random

                        rkl = list()

                        self.mm.set_ticket(self.TICKETS[spd][3])
                        
                        rkl.append(self._mse_loss(x, full_activations))

                        #MG_TICKET 
                        

                        mkl = list()

                        self.mm.set_ticket(self._mgtickets[spd])
                        
                        mkl.append(self._mse_loss(x, full_activations))
                        
                        #NEG_TICKET 
                        

                        nkl = list()

                        self.mm.set_ticket(self._negtickets[spd])
                        
                        nkl.append(self._mse_loss(x, full_activations))
                        
                        
                        #EXPORT

                        self.activation_log[spd][iteration] = {'IMP': [kl.item() for kl in ikl], 'LAYER_RANDOM': [kl.item() for kl in lkl], 
                                                                'CLOSE_RANDOM': [kl.item() for kl in rdkl],
                                                                'TRUE_RANDOM': [kl.item() for kl in rkl], 
                                                                "REAL_TIME_MAGNITUDE": [kl.item() for kl in mkl], 
                                                                "REAL_TIME_NEGATIVES": [kl.item() for kl in nkl],}

                        #if not all([kl > 0 for key in self.activation_log[spd][iteration].keys() for kl in self.activation_log[spd][iteration][key]]):
                        #    print(self.activation_log[spd][iteration])
                        #    print(full_activations.sum(), full_activations.ge(0).all(), )
                        
                        #print(self.activation_log[spd][iteration])


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
        #return
        if len(self._hooks) != 0: return
        for n, block in self.mm.named_children():
            for name, layer in block.named_children():
                if name.endswith("relu"):
                    self._hooks.append(layer.register_forward_hook(self._activation_hook))
                #elif name.endswith("fc"):
                #    self._hooks.append(layer.register_forward_hook(self._fake_activation_hook))
        return

    def _activation_hook(self, module, input, output: torch.Tensor) -> None:
        self.act_w.append(output.detach().to(torch.float64).view(output.shape[0], -1))#.cpu().to(torch.float64).mean(dim = 0).view(-1))
        return
    
    def _fake_activation_hook(self, module, input, output: torch.Tensor) -> None:
        self.act_w.append(F.relu(output.detach()).to(torch.float64).view(output.shape[0], -1))#.cpu()).to(torch.float64).mean(dim = 0).view(-1))
        return

    def clear_act_captures(self) -> None:
        self.act_w.clear()

    ### ----------------------------------------------------------- HOOKS ---------------------------------------------------------------------------------------

    def pre_IMP_hook(self, name: str):
        #self.__CURR_IMP_ITER = 0
        self.act_w = list()
        #open(f'./tmp/swap/activations/{name}.h5', 'w').close()
        #os.remove(f"./tmp/swap/activations/{self.NAME}.h5")

    #def post_prune_hook(self, iteration: int, epochs_per_run: int):
    #    self.__CURR_IMP_ITER = iteration

    def post_train_hook(self):
        self.disable_act_hooks()
        return
    
    def pre_epoch_hook(self, *args):

        self.mm.reset_ticket()
        for sparsity in self.TICKETS.keys():

            self.mm.prune_by_mg(sparsity, iteration = 1)
            mg_ticket = self.mm.export_ticket_cpu()
            self._mgtickets[sparsity] = mg_ticket
            self.mm.reset_ticket()
            
            self.mm.prune_positives(sparsity)
            neg_ticket = self.mm.export_ticket_cpu()
            self._negtickets[sparsity] = neg_ticket
            self.mm.reset_ticket()

        return

    def post_epoch_hook(self, epoch, EPOCHS):
        if epoch == 78 or epoch == 118: # Epochs 80, 120
            self.reduce_learning_rate(10)
        return 

    #def pre_step_hook(self, step, steps_per_epoch):
    #    if step % 2 == 0 and self.ACTS:
    #        self.init_act_hooks()

    def post_step_hook(self, x, y, iteration, step, steps_per_epoch, **kwargs):
        #print(self.RANK, step)
        if step % 2 == 0 and self.ACTS:
            #with torch.autocast('cuda', dtype = torch.float16, enabled = self.AMP):
            self.init_act_hooks()
            self.collect_activations_and_test(x, y, iteration)
            self.disable_act_hooks()
