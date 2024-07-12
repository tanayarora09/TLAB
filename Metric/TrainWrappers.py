import torch
from torch import nn
import torch.distributed as dist

import torchinfo

from typing import List, Tuple, Callable
from collections import defaultdict
import time
import math

from contextlib import contextmanager

@contextmanager
def conditional_no_grad(condition):
    if condition:
        with torch.no_grad():
            yield
    else:
        yield

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

    def build(self, optimizer: torch.optim.optimizer, 
              collective_transforms: List[nn.Module],
              train_transforms: List[nn.Module],
              eval_transforms: List[nn.Module],
              final_collective_transforms: List[nn.Module],
              loss: Callable = nn.CrossEntropyLoss(reduction = "sum"), 
              scale_loss: bool = False, 
              gradient_clipnorm: float = float('inf')):
        
        """
        Build Trainer.

        Loss must use reduction = sum.
        
        If transforms have already been implemented, you can pass [] to all of the transforms.
        
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

        self.loss_tr = torch.as_tensor(0.0, dtype = torch.float64, device = 'cuda')
        self.acc_tr = torch.as_tensor(0, dtype = torch.int64, device = 'cuda')
        self.count_tr = torch.as_tensor(0, dtype = torch.int64, device = 'cuda')

        self._COLORS = {
            "reset": f"\033[0m",           # Logging Helper
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
        return output[:topk].view(-1).float().sum(0)

    @torch.no_grad()
    def metric_results(self) -> dict[str, float]:
        """
        Return Loss and Accuracy. 
        Should be called from root process.
        """
        return {"loss": (self.loss_tr.div(self.count_tr).detach().item()),
                "accuracy": (self.acc_tr.div(self.count_tr).detach().item())}
    
    @torch.no_grad()
    def reset_metrics(self):
        """
        Reset Loss, Accuracy, and Sample Count.
        Should be called from all processes.
        """
        self.loss_tr.fill_(0.0)
        self.acc_tr.fill_(0)
        self.count_tr.fill_(0)

    @torch.no_grad()
    def collect_metrics(self):
        """
        Collects Loss, Accuracy, and Sample Count.
        Should be called from all processes.

        Technically, it only has to be called from root process, 
        but calling from all is good practice.
        """
        dist.all_reduce(self.loss_tr, op = dist.ReduceOp.SUM)
        dist.all_reduce(self.acc_tr, op = dist.ReduceOp.SUM)
        dist.all_reduce(self.count_tr, op = dist.ReduceOp.SUM)

    #------------------------------------------ MAIN TRAIN FUNCTIONS -------------------------------------- #

    def fit(self, train_data: torch.utils.data.DataLoader, validation_data: torch.utils.data.DataLoader,
            epochs: int, train_cardinality: int, name: str, accumulation_steps: int = 1):
        """
        Basic Training Run Implementation.
        Override by Copy and Pasting.

        train_data and validation_data are expected to be sharded and batched, and use DistributedSampler

        train_cardinality is the number of batches of the train set. It is used for logging.

        """

        if self.IsRoot: logs = defaultdict(dict)
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
                        
                        self.collect_metrics()
                        
                        if self.IsRoot: logs[iter] = self.metric_results()
                        
                        if (step + 1) % 48 == 0:
                            
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
    def train_step(self, x: torch.Tensor, y: torch.Tensor, accum: bool = True, accum_steps: int = 1):
        
        with torch.autocast('cuda', dtype = torch.float16, enabled = self.AMP):

            output = self.m(x)
            loss = self.criterion(output, y)
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

        self.optim.zero_grad(True)

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

        for step, (x, y) in enumerate(test_data):

            x, y = x.to('cuda'), y.to('cuda')
            
            for T in self.cT: x = T(x) # Transforms
            for T in self.eT: x = T(x)
            for T in self.fcT: x = T(x)

            self.test_step(x, y)
        
        self.collect_metrics()
        return

    @torch.compile
    @torch.no_grad
    def test_step(self, x: torch.Tensor, y: torch.Tensor) -> None:

        output = self.m(x)
        loss = self.criterion(output, y)

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
        fp = f"./WEIGHTS/{self.fromNamePrefix(name, prefix)}.pth.tar"
        ckpt = {'model': self.m.module.state_dict(),
                'optim': self.optim.state_dict(),
                'scaler': self.lossScaler.state_dict()}
        torch.save(ckpt, fp)        

    @torch.no_grad()
    def load_ckpt(self, name: str, prefix: str = None):
        """
        Loads Model, Optimizer, and LossScaler state_dicts.
        Meant to be used with save_ckpt.
        """
        fp = f"./WEIGHTS/{self.fromNamePrefix(name, prefix)}.pth.tar"
        ckpt = torch.load(fp, map_location = {'cuda:%d' % 0: 'cuda:%d' % self.RANK}) # Assumes rank = 0 is Root.
        self.m.module.load_state_dict(ckpt['model'])
        self.optim.load_state_dict(ckpt['optim'])
        self.lossScaler.load_state_dict(ckpt['scaler'])

    #------------------------------------------ HELPER FUNCTIONS -------------------------------------- #

    ### Logging

    def print(self, input, color: str = 'default') -> None:
        """
        Prints with color. Make sure input has a __repr__ attribute.
        """
        if not self.IsRoot: return
        torch._print(self._COLORS[color] + str(input) + self._COLORS["reset"])
        return
    
    def fromNamePrefix(self, name: str, prefix: str):
        return f"{prefix}_{name}" if prefix else name

    def summary(self, batch_size):
        """
        Prints torchinfo summary of model.
        """
        torchinfo.summary(self.m.module, (batch_size, 3, 224, 224))

    ### Training

    def reset_loss_scaler(self):
        """
        Reset Cuda Loss Scaler - Use with Mixed Precision.
        Should be reset every 
        """
        self.lossScaler = torch.amp.GradScaler(device = 'cuda', enabled = self.use_amp)

    @torch.no_grad()
    def reduce_learning_rate(self, factor: int):
        for pg in self.optim.param_groups:
            pg['lr'] /= factor


    ### Metrics
