import torch
from torch import nn
import torch.distributed as dist

import numpy as np

import torchinfo

from typing import Callable, Tuple
from collections import defaultdict
import time
import math

import h5py
from models.LotteryLayers import Lottery

class BaseCNNTrainer:

    #------------------------------------------ MAIN INIT FUNCTIONS -------------------------------------- #

    def __init__(self, model: torch.nn.parallel.DistributedDataParallel, rank: int):
        """
        Model has to be DDP.
        Non-Distributed Environments Not Supported.
        Default cuda device should be set.
        """
        self.m = model
        self.RANK = rank
        self.IsRoot = rank == 0

    def build(self, optimizer: torch.optim.Optimizer, 
              collective_transforms: Tuple[nn.Module],
              train_transforms: Tuple[nn.Module],
              eval_transforms: Tuple[nn.Module],
              final_collective_transforms: Tuple[nn.Module],
              loss: Callable = nn.CrossEntropyLoss(reduction = "sum"), 
              scale_loss: bool = False, decay: float = 0.0,
              gradient_clipnorm: float = float('inf')):
        
        """
        Build Trainer.

        Loss must use reduction = sum.
        
        If transforms have already been implemented, ycou can pass [] to all of the transforms.
        
        I implement optional gradient clipping and loss scaling / mixed_precision.

        Final Collective Transforms mainly exists to provide compatibility for normalization.
        """

        self.criterion = loss
        self.optim = optimizer

        self.cT = collective_transforms
        self.tT = train_transforms
        self.eT = eval_transforms
        self.fcT = final_collective_transforms
        
        self.AMP = scale_loss
        self.reset_loss_scaler()

        self.gClipNorm = gradient_clipnorm

        self.LD = decay

        self.loss_tr = torch.as_tensor(0.0, dtype = torch.float64, device = 'cuda')
        self.acc_tr = torch.as_tensor(0, dtype = torch.int64, device = 'cuda')
        self.count_tr = torch.as_tensor(0, dtype = torch.int64, device = 'cuda')

        self.eloss = torch.as_tensor(0.0, dtype = torch.float64, device = 'cuda')
        self.eacc = torch.as_tensor(0, dtype = torch.int64, device = 'cuda')
        self.ecount = torch.as_tensor(0, dtype = torch.int64, device = 'cuda')

        self._COLORS = {
            "reset": f"\033[0m",
            "default": f"\033[0m",           # Logging Helper
            "red": f"\033[38;2;255;0;0m",
            "green": f"\033[38;2;0;255;0m",
            "blue": f"\033[38;2;0;0;255m",
            "cyan": f"\033[38;2;0;255;255m",
            "magenta": f"\033[38;2;255;0;255m",
            "yellow": f"\033[38;2;255;255;0m",
            "white": f"\033[38;2;255;255;255m",
            "gray": f"\033[38;2;128;128;128m",
            "orange": f"\033[38;2;255;165;0m",
            "purple": f"\033[38;2;128;0;128m",
            "brown": f"\033[38;2;165;42;42m",
            "pink": f"\033[38;2;255;192;203m"
        }

    #------------------------------------------ MAIN METRIC FUNCTIONS -------------------------------------- #

    @torch.no_grad()
    def correct_k(self, output: torch.Tensor, labels: torch.Tensor, topk: int = 1) -> torch.Tensor:
        """
        Returns number of correct prediction.
        Deprecates output tensor.
        """
        _, output = output.topk(topk, 1)
        output.t_()
        output.eq_(labels.view(1, -1).expand_as(output))
        return output[:topk].view(-1).to(torch.int64).sum(0)

    @torch.no_grad()
    def metric_results(self) -> dict[str, float]:
        """
        Return Loss and Accuracy. 
        Should be called from root process.
        """
        return {"loss": (self.eloss.div(self.ecount).detach().item()),
                "accuracy": (self.eacc.div(self.ecount).detach().item())}
    
    @torch.no_grad()
    def reset_metrics(self):
        self.eloss.fill_(0.0)
        self.eacc.fill_(0)
        self.ecount.fill_(0)
        self.reset_running_metrics()

    @torch.no_grad()
    def reset_running_metrics(self):
        """
        Reset Loss, Accuracy, and Sample Count.
        Should be called from all processes.
        """
        self.loss_tr.fill_(0.0)
        self.acc_tr.fill_(0)
        self.count_tr.fill_(0)
    
    @torch.no_grad()
    def transfer_metrics(self):
        """
        Move from running metrics.
        This
        """
        self._collect_metrics()
        self.eloss += self.loss_tr
        self.eacc += self.acc_tr
        self.ecount += self.count_tr
        self.reset_running_metrics()

    @torch.no_grad()
    def _collect_metrics(self):
        """
        Collects Loss, Accuracy, and Sample Count.
        Do not directly call. Use transfer_metrics instead.
        """
        dist.all_reduce(self.loss_tr, op = dist.ReduceOp.SUM)
        dist.all_reduce(self.acc_tr, op = dist.ReduceOp.SUM)
        dist.all_reduce(self.count_tr, op = dist.ReduceOp.SUM)

    #------------------------------------------ MAIN TRAIN FUNCTIONS -------------------------------------- #

    def fit(self, train_data: torch.utils.data.DataLoader, validation_data: torch.utils.data.DataLoader,
            epochs: int, train_cardinality: int, name: str, accumulation_steps: int = 1) -> dict:
        """
        Basic Training Run Implementation.
        Override by Copy and Pasting.

        train_data and validation_data are expected to be sharded and batched, and use DistributedSampler

        train_cardinality is the number of batches of the train set. It is used for logging.

        """

        logs = defaultdict(dict)
        self.reset_metrics()
        self.save_ckpt(name = name, prefix = "init")
        
        best_val_loss = float('inf')

        train_start = None
        val_start = None

        for epoch in range(epochs):

            self.print(f"\nStarting Epoch {epoch + 1}\n", 'red')
            train_start = time.time()
            self.m.train()

            accum = False

            self.reset_metrics()

            for step, (x, y) in enumerate(train_data):

                iter = int(epoch * train_cardinality + step + 1)
                accum = ((step + 1) % accumulation_steps == 0) or (step + 1 == train_cardinality)

                x, y = x.to('cuda'), y.to('cuda')

                for T in self.cT: x = T(x) # Transforms
                for T in self.tT: x = T(x)
                for T in self.fcT: x = T(x)

                self.train_step(x, y, accum, accumulation_steps)

                if accum:
                    
                    if (step + 1) % 24 == 0 or (step + 1 == train_cardinality): # Synchronize and Log.
                        
                        self.transfer_metrics()
                        
                        if self.IsRoot: logs[iter] = self.metric_results()
                        
                        if (step + 1) % 48 == 0 and self.IsRoot:
                            
                            self.print(f"----  Status at {math.ceil((step + 1) / 48):.0f}/8: ----     Accuracy: {logs[iter]['accuracy']:.4f}   --  Loss: {logs[iter]['loss']:.5f} --", 'white')

                if iter == 500:
                    self.save_ckpt(name = name, prefix = "rewind")

            self.print(f"Training stage took {(time.time() - train_start):.1f} seconds.", 'yellow')

            val_start = time.time()

            self.evaluate(validation_data)

            if self.IsRoot:

                logs[(epoch + 1) * train_cardinality].update({('val_' + k): v for k, v in self.metric_results().items()})

                self.print(f"\n||| EPOCH {epoch + 1} |||\n", 'orange')

                for k, v in logs[(epoch + 1) * train_cardinality].items():
                    self.print(f"\t{k}: {v:.6f}", 'cyan')

                if logs[(epoch + 1) * train_cardinality]['val_loss'] < best_val_loss:
                    best_val_loss = logs[(epoch + 1) * train_cardinality]['val_loss']
                    self.print(f"\n -- UPDATING BEST WEIGHTS TO {epoch + 1} -- \n", "magenta")
                    self.save_ckpt(name = name, prefix = "best")

                self.print(f"Validation stage took {(time.time() - val_start):.1f} seconds. \nTotal for Epoch: {(time.time() - train_start):.1f} seconds.", 'yellow')

        
        return logs
 

    @torch.compile
    def train_step(self, x: torch.Tensor, y: torch.Tensor, accum: bool = True, accum_steps: int = 1, id: str = None):
        
        with torch.autocast('cuda', dtype = torch.float16, enabled = self.AMP):

            output = self.m(x)
            loss = self.criterion(output, y) #+ self.LD * self.calculate_custom_regularization()
            loss /= accum_steps

        if not accum:
            
            with self.m.no_sync():
                self.lossScaler.scale(loss).backward()

            return
        
        self.lossScaler.scale(loss).backward()

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

    #------------------------------------------ MAIN EVALUATE FUNCTIONS -------------------------------------- #

    @torch.compile
    @torch.no_grad()
    def evaluate(self, test_data: torch.utils.data.DataLoader) -> None:
        """
        Evaluate model on dataloader.
        To retrieve results, call metric_results()
        """

        self.m.eval()
        self.reset_metrics()

        for x, y in test_data:

            x, y = x.to('cuda'), y.to('cuda')
            
            for T in self.cT: x = T(x) # Transforms
            for T in self.eT: x = T(x)
            for T in self.fcT: x = T(x)

            self.test_step(x, y)
        
        self.transfer_metrics()
        self.print(f"Evaluated on {self.ecount.detach().item()} samples.")
        return

    @torch.compile
    @torch.no_grad
    def test_step(self, x: torch.Tensor, y: torch.Tensor) -> None:

        output = self.m(x)
        loss = self.criterion(output, y)  #+ self.LD * self.calculate_custom_regularization()

        self.loss_tr += loss
        self.acc_tr += self.correct_k(output, y)
        self.count_tr += y.size(dim = 0)

        return
        

    #------------------------------------------ SERIALIZATION FUNCTIONS -------------------------------------- #

    @torch.no_grad()
    def save_ckpt(self, name: str, prefix: str = None):
        """
        Saves Model, Optimizer, and LossScaler state_dicts.
        Can be accessed with same name and prefix from load_ckpt.
        """
        if not self.IsRoot: return
        fp = f"./WEIGHTS/{self.fromNamePrefix(name, prefix)}.pt"
        ckpt = {'model': self.m.state_dict(),
                'optim': self.optim.state_dict(),
                'scaler': self.lossScaler.state_dict()}
        torch.save(ckpt, fp)        

    @torch.no_grad()
    def load_ckpt(self, name: str, prefix: str = None):
        """
        Loads Model, Optimizer, and LossScaler state_dicts.
        Meant to be used with save_ckpt.
        """
        fp = f"./WEIGHTS/{self.fromNamePrefix(name, prefix)}.pt"
        ckpt = torch.load(fp, map_location = {'cuda:%d' % 0: 'cuda:%d' % self.RANK},
                          weights_only = True) # Assumes rank = 0 is Root.
        self.m.load_state_dict(ckpt['model'])
        self.optim.load_state_dict(ckpt['optim'])
        self.lossScaler.load_state_dict(ckpt['scaler'])

    #------------------------------------------ HELPER FUNCTIONS -------------------------------------- #

    ### Logging

    def print(self, input, color: str = 'default' , rank: int = 0) -> None:
        """
        Prints with color. Make sure input has a __repr__ attribute.
        """
        if not self.RANK == rank: return
        torch._print(self._COLORS[color] + str(input) + self._COLORS["reset"])
        return
    
    def fromNamePrefix(self, name: str, prefix: str):
        return f"{prefix}_{name}" if prefix else name

    def summary(self, batch_size):
        """
        Prints torchinfo summary of model.
        """
        if not self.IsRoot: return
        torchinfo.summary(self.m.module, (batch_size, 3, 224, 224))

    ### Training

    def reset_loss_scaler(self):
        """
        Reset Cuda Loss Scaler - Use with Mixed Precision.
        Should be reset every 
        """
        self.lossScaler = torch.amp.GradScaler(device = 'cuda', enabled = self.AMP)

    @torch.no_grad()
    def reduce_learning_rate(self, factor: int):
        for pg in self.optim.param_groups:
            pg['lr'] /= factor

    ### Metrics

    @torch.compile
    def calculate_custom_regularization(self):
        running_regularization_loss = torch.as_tensor(0.0, device = "cuda")
        for name, buffer in self.m.named_buffers():
            if not name.endswith("mask"): continue
            running_regularization_loss += torch.norm(buffer * self.m.get_parameter(name[:-5]), p = 2)
        for name, param in self.m.named_parameters():
            if name.endswith("bias") or "out" in name and "norm" not in name:
                running_regularization_loss += torch.norm(param, p = 2)
        return running_regularization_loss







class BaseIMP(BaseCNNTrainer):

    #------------------------------------------ LTH IMP FUNCTIONS -------------------------------------- #

    def TicketIMP(self, train_data: torch.utils.data.DataLoader, validation_data: torch.utils.data.DataLoader, 
                  epochs_per_run: int, train_cardinality: int, name: str, prune_rate: float, prune_iters: int):

        """
        Find Winning Ticket Through IMP with Rewinding. Calls Trainer.fit(); See description for argument requirements.

        prune_rate should be a float that indicates what percent of weights to prune per training run. 
        I.E. to prune 20% per iteration, prune_rate = 20.0

        prune_iters should be the number of iterations to run IMP for.
        
        Final sparsity = (1 - prune_rate/100) ** prune_iters
        """
        
        total_logs = defaultdict()
        
        sparsities = [None] * (prune_iters + 1)
        current_sparsity = 100.0
        sparsities[0] = current_sparsity
        
        self.pre_IMP_hook(name)

        self.print(f"RUNNING IMP ON {self.m.module.num_prunable} PRUNABLE WEIGHTS.", "purple")

        total_logs[0] = self.fit(train_data, validation_data, epochs_per_run, train_cardinality, name + f"_IMP_{(current_sparsity):.1f}", isfirst = True)
        
        for iteration in range(1, prune_iters + 1):
            
            self.prune_by_mg(prune_rate, iteration)
            
            current_sparsity = self.m.module.sparsity

            sparsities[iteration] = current_sparsity

            self.print(f"SPARSITY: {current_sparsity:.1f}", "purple")

            self.post_prune_hook(iteration, epochs_per_run)

            self.export_ticket(f"{name}_IMP_{(current_sparsity):.1f}")

            self.load_ckpt(name + f"_IMP_{(100.0):.1f}", prefix = "rewind") 

            self.fit(train_data, validation_data, epochs_per_run, train_cardinality, name + f"_IMP_{(current_sparsity):.1f}", isfirst = False)

        self.post_IMP_hook()

        return total_logs, sparsities


    def prune_by_mg(self, rate: float, iteration: float):

        all_weights = torch.cat([(self.m.get_parameter(name[:-5]) * buffer).reshape([-1]) for name, buffer in self.m.named_buffers() if name.endswith("mask")], dim = 0)

        threshold = np.quantile(all_weights.abs_().detach().cpu().numpy(), q = 1.0 - rate ** iteration, method = "higher")

        for name, buffer in self.m.named_buffers():

            if not name.endswith("mask"): continue

            buffer.copy_((self.m.get_parameter(name[:-5]).abs().ge(threshold)).to(torch.float32))

        return 

    @torch._dynamo.disable
    @torch.no_grad()
    def export_ticket(self, name):
        if not self.IsRoot: return
        with h5py.File(f"./TICKETS/{name}.h5", 'w') as f:
            for n, mask in self.m.named_buffers():   
                if not n.endswith("mask"): continue
                f.create_dataset(n, data = mask.detach().cpu().numpy())

    @torch._dynamo.disable         
    @torch.no_grad()
    def load_ticket(self, name):

        if self.IsRoot:
            with h5py.File(f"./TICKETS/{name}.h5", 'r') as f:
                data = {n: torch.as_tensor(f[n][:], device = "cuda") for n, mask in self.m.named_buffers() if n.endswith("mask")}
        else:
            data = {n: torch.empty_like(mask) for n, mask in self.m.named_buffers() if n.endswith("mask")}

        for n, tensor in data.items():

            dist.broadcast(tensor, src=0)

            for buffer_name, mask in self.m.named_buffers():
                if buffer_name == n:
                    mask.copy_(tensor)

            dist.barrier()        

    def pre_IMP_hook(self, name: str):
        return    

    def post_IMP_hook(self):
        return

    def post_prune_hook(self, iteration: int, num_epochs: int):
        return
