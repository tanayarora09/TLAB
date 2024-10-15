import torch
from torch import nn
import torch.distributed as dist

from collections import defaultdict
import time

from training.base import BaseIMP

from utils.serialization_utils import read_tensor, save_tensor

import math
import os

class VGG_POC(BaseIMP):

    def __init__(self, model: torch.nn.parallel.DistributedDataParallel, rank: int):
        super(VGG_POC, self).__init__(model, rank)
        self.IsMetricRoot = rank == 1

    def build(self, *args, tickets_dict: dict = None, **kwargs):
        super().build(*args, **kwargs)
        self.ACTS = tickets_dict != None
        self.TICKETS = tickets_dict # {IMP_ITERATION - 1: (TICKET, SPARSITY_D)}
        self.activation_log = defaultdict(defaultdict) # {IMP_ITERATION: {Epoch:(IMP KL, RAND KL)}}
        
        self.Ikl_tr = torch.as_tensor(0.0, dtype = torch.float64, device = "cuda")
        self.Rkl_tr = torch.as_tensor(0.0, dtype = torch.float64, device = "cuda")

        self.eIkl = torch.as_tensor(0.0, dtype = torch.float64, device = "cuda")
        self.eRkl = torch.as_tensor(0.0, dtype = torch.float64, device = "cuda")

    ### ----------------------------------------------------------- METRICS ---------------------------------------------------------------------------------------

    @torch.no_grad()
    def reset_metrics(self):
        super().reset_metrics()
        self.eIkl.fill_(0.0)
        self.eRkl.fill_(0.0)

    @torch.no_grad()
    def reset_running_metrics(self):
        super().reset_running_metrics()
        self.Ikl_tr.fill_(0.0)
        self.Rkl_tr.fill_(0.0)

    @torch.no_grad()
    def _collect_metrics(self):
        super()._collect_metrics()
        dist.all_reduce(self.Ikl_tr, op = dist.ReduceOp.SUM)
        dist.all_reduce(self.Rkl_tr, op = dist.ReduceOp.SUM)

    @torch.no_grad()
    def transfer_metrics(self):
        self._collect_metrics()
        self.eloss += self.loss_tr
        self.eacc += self.acc_tr
        self.ecount += self.count_tr
        self.eIkl += self.Ikl_tr
        self.eRkl += self.Rkl_tr
        self.reset_running_metrics()

    ### ----------------------------------------------------------- ACTIVATION CAPTURING ---------------------------------------------------------------------------------------

    @torch.no_grad()
    def collect_activations_and_test(self, x: torch.Tensor) -> None:

        if not self.ACTS: return
        
        print(f"Reached collection. {self.RANK}")

        with torch.random.fork_rng(devices = ["cuda:0", "cuda:1", "cuda:2", "cuda:3"], enabled = True):
        
            full_activations = torch.cat(self.act_w).to(torch.float64)
            full_activations.div_(full_activations.sum()) + 1e-11
            original_ticket = self.mm.export_ticket_cpu()

            print(self.mm.sparsity)

            # IMP_TICKET
            self.clear_act_captures()
            self.mm.set_ticket(self.TICKETS[self.__CURR_IMP_ITER][0])
            print(self.mm.sparsity)

            dist.barrier()

            with torch.no_grad(): self.m(x)

            print(f"[rank {self.RANK}] Completed forward pass.")

            print(self.act_w)

            curr_activations = torch.cat(self.act_w).to(torch.float64)

            print(f"[rank {self.RANK}]", curr_activations)
            
            curr_activations.div_(curr_activations.sum())

            print(f"[rank {self.RANK}] Normalized activations: ", curr_activations)

            curr_activations = torch.log(curr_activations) + 1e-11

            print(f"[rank {self.RANK}] Torch.log ran.")

            dist.barrier()

            self.Ikl_tr += torch.kl_div(curr_activations, full_activations) # + 1e-11 for numerical stability

            print(f"Logged.")

            #RAND_TICKET 
            self.clear_act_captures()

            print(f"[rank {self.RANK}] Cleared act_captures: ", self.act_w)

            self.mm.reset_ticket()
            self.mm.prune_random(self.TICKETS[self.__CURR_IMP_ITER][1], distributed = True, root = self.IsTicketRoot, rootn = 2)

            print(self.mm.sparsity)


            dist.barrier()
            with torch.no_grad(): self.m(x)
            curr_activations = torch.cat(self.act_w).to(torch.float64)
            curr_activations /= curr_activations.sum()
            self.Rkl_tr += torch.kl_div(torch.log(curr_activations) + 1e-11, full_activations)

            #Reset
            self.clear_act_captures()
            self.mm.set_ticket(original_ticket)

        return

    def disable_act_hooks(self):
        for n, block in self.mm.named_children():
            for name, layer in block.named_children():
                if name.endswith("relu"):
                    layer.disable_fw_hooks()
        return 

    def enable_act_hooks(self):
        for name, block in self.m.module.named_children():
            for name, layer in block.named_children():
                if name.endswith("relu"):
                    layer.enable_fw_hooks()
        return

    def init_act_hooks(self):
        for n, block in self.mm.named_children():
            for name, layer in block.named_children():
                if name.endswith("relu"):
                    layer.init_fw_hook(self._activation_hook)
        return

    @torch._dynamo.disable
    def _activation_hook(self, module, input, output) -> None:
        self.act_w.append(output.detach().to(torch.float64).view(-1).cpu())
        """
        out = [torch.empty_like(output, dtype = torch.float64, device = "cuda", requires_grad = False) for _ in range(dist.get_world_size())] if self.IsMetricRoot else None
        dist.gather(output.detach().to(torch.float64), out, dst = 1)
        if not self.IsMetricRoot: return
        out = torch.cat(out, dim = 0)
        out = torch.mean(out, dim = 0, keepdim = True) # Average Pooling Reduce
        self.act_w.append(out.view(-1).cpu())"""
        return

    def clear_act_captures(self) -> None:
        self.act_w.clear()

    ### ----------------------------------------------------------- HOOKS ---------------------------------------------------------------------------------------

    def pre_IMP_hook(self, name: str):
        self.__CURR_IMP_ITER = 0
        self.__CURREPOCH = None 
        self.act_w = list()
        self.init_act_hooks()
        self.disable_act_hooks()
        #open(f'./tmp/swap/activations/{name}.h5', 'w').close()
        #os.remove(f"./tmp/swap/activations/{self.NAME}.h5")

    def post_prune_hook(self, iteration: int, epochs_per_run: int):
        self.__CURR_IMP_ITER = iteration

    def pre_epoch_hook(self, epoch: int):
        self.__CURREPOCH = epoch
        if self.ACTS: self.enable_act_hooks()

    def post_train_hook(self):
        self.disable_act_hooks()
        self.activation_log[self.__CURR_IMP_ITER][self.__CURREPOCH] = (self.eIkl.div(self.ecount).detach().item(), self.eRkl.div(self.ecount).detach().item())
        return
    
    def post_epoch_hook(self, epoch):
        if epoch == 78 or epoch == 118: # Epochs 80, 120
            self.reduce_learning_rate(10)
        return 

    ### SAME AS BASE, BUT CALLS COLLECT_ACTIVATIONS_AND_TEST
    @torch.compile
    def train_step(self, x: torch.Tensor, y: torch.Tensor, accum: bool = True, accum_steps: int = 1, id: str = None): 

        with torch.autocast('cuda', dtype = torch.float16, enabled = self.AMP):

            output = self.m(x)
            loss = self.criterion(output, y)
            loss /= accum_steps

        if not accum:
            
            with self.m.no_sync():
                self.lossScaler.scale(loss).backward()

            return
        
        self.lossScaler.scale(loss).backward()

        with torch.autocast('cuda', dtype = torch.float16, enabled = self.AMP):
            self.collect_activations_and_test(x) # <-------------------------------------------------------------------------------------------------------------------

        dist.barrier()

        self.lossScaler.unscale_(self.optim)
        nn.utils.clip_grad_norm_(self.m.parameters(), max_norm = self.gClipNorm)

        self.lossScaler.step(self.optim)
        self.lossScaler.update()

        self.optim.zero_grad(set_to_none = True)

        with torch.no_grad():

            self.loss_tr += loss
            self.acc_tr += self.correct_k(output, y)
            self.count_tr += y.size(dim = 0)
        
        return