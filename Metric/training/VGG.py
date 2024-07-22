import torch
import torch.distributed as dist

from collections import defaultdict
import time

from training.base import BaseIMP

from utils.serialization_utils import read_tensor, save_tensor

import math

class VGG_IMP(BaseIMP):

    """
    Support for VGG Training Loop
    """
    
    def fit(self, train_data: torch.utils.data.DataLoader, validation_data: torch.utils.data.DataLoader,
        epochs: int, train_cardinality: int, name: str, accumulation_steps: int = 1, isfirst: bool = True) -> dict:
        """
        Basic Training Run Implementation.
        Override by Copy and Pasting.

        train_data and validation_data are expected to be sharded and batched, and use DistributedSampler

        train_cardinality is the number of batches of the train set. It is used for logging.

        """

        logs = defaultdict(dict)
        self.reset_metrics()
        if isfirst: self.save_ckpt(name = name, prefix = "init")
        
        best_val_loss = float('inf')

        train_start = None
        val_start = None

        for epoch in range(epochs):

            self.print(f"\nStarting Epoch {epoch + 1}\n", 'red')
            train_start = time.time()
            self.m.train()

            self.pre_epoch_hook(epoch)

            accum = False

            self.reset_metrics()

            if epoch == 79 or epoch == 119:
                self.reduce_learning_rate(10)

            for step, (x, y, id) in enumerate(train_data):

                iter = int(epoch * train_cardinality + step + 1)
                accum = ((step + 1) % accumulation_steps == 0) or (step + 1 == train_cardinality)

                self.pre_step_hook(step, train_cardinality)

                x, y = x.to('cuda'), y.to('cuda')

                for T in self.cT: x = T(x) # Transforms
                for T in self.tT: x = T(x)
                for T in self.fcT: x = T(x)

                self.train_step(x, y, accum, accumulation_steps, id = id[0])

                if accum:
                    
                    if (step + 1) % 24 == 0 or (step + 1 == train_cardinality): # Synchronize and Log.
                        
                        self.transfer_metrics()
                        
                        if self.IsRoot: logs[iter] = self.metric_results()
                        
                        if (step + 1) % 48 == 0 and self.IsRoot:
                            
                            self.print(f"----  Status at {math.ceil((step + 1) / 48):.0f}/8: ----     Accuracy: {logs[iter]['accuracy']:.4f}   --  Loss: {logs[iter]['loss']:.5f} --", 'white')

                if iter == 500 and isfirst:
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
    @torch.no_grad()
    def evaluate(self, test_data: torch.utils.data.DataLoader) -> None:
        """
        Evaluate model on dataloader.
        To retrieve results, call metric_results()
        """

        self.m.eval()
        self.reset_metrics()

        for x, y, _ in test_data:

            x, y = x.to('cuda'), y.to('cuda')
            
            for T in self.cT: x = T(x) # Transforms
            for T in self.eT: x = T(x)
            for T in self.fcT: x = T(x)

            self.test_step(x, y)
        
        self.transfer_metrics()
        self.print(f"Evaluated on {self.ecount.detach().item()} samples.")
        return
    
    def pre_epoch_hook(self, epoch: int):
        return
    
    def pre_step_hook(self, step: int, train_cardinality: int):
        return
    
    

"""
    ### Gathering Functions

    def capture_grads(self):
        out = defaultdict(torch.Tensor)
        for name, param in self.m.named_parameters():
            out[name] = param.grad.detach().cpu()
        return out
    
    def sync_grads(self):

        if self.ROOT: print("SYNCING GRADS / DISTANCES")
        
        if len(self._grads == 1):
            out = [None] * dist.get_world_size()
            dist.all_gather_object(out, self._grads)
            dist.barrier()
            self._grads = out
        
        else:
            out = [None] * dist.get_world_size()
            dist.all_gather_object(out, self.curr_L2)
            for curr in out: # 4 Loops
                for id in curr.keys(): # Num Batches Loops
                    for map in self._grads: # 4 Loops
                        if id in map:
                            map[id].append(curr[id])
                            continue
"""

class VGG_POC(VGG_IMP):

    def __init__(self, model: torch.nn.parallel.DistributedDataParallel, rank: int):
        super(VGG_POC, self).__init__(model, rank)
        self.IsGradRoot = rank == 1

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)
        self.grad_captures = defaultdict(list) ### {IMP_ITER: [{id: score} for epochs]}
        self.act_captures = defaultdict()
        self.__CURR_IMP_ITER = None
        self.__CURREPOCH = None

    @torch.compile
    def train_step(self, x: torch.Tensor, y: torch.Tensor, accum: bool = True, 
                   accum_steps: int = 1, id: str = None):
        
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
        torch.nn.utils.clip_grad_norm_(self.m.parameters(), max_norm = self.gClipNorm)

        with torch.no_grad():
            self.gradient_log(id)

        self.lossScaler.step(self.optim)
        self.lossScaler.update()

        self.optim.zero_grad(set_to_none = True)

        with torch.no_grad():

            self.loss_tr += loss
            self.acc_tr += self.correct_k(output, y)
            self.count_tr += y.size(dim = 0)
        
        return

    def gradient_log(self, id: str):
        
        if not self.IsGradRoot or not self.SAVE_GRADS: return

        grads = self._capture_grads()
        
        if self.__CURR_IMP_ITER > 0:

            full = read_tensor("gradients", f"{self.NAME}", f"full_{str(self.__CURREPOCH)}_{id}")
            curr = read_tensor("gradients", f"{self.NAME}", f"curr_{str(self.__CURREPOCH)}_{id}")
        
            self.grad_captures[self.__CURR_IMP_ITER][self.__CURREPOCH][id] = (torch.linalg.vector_norm(grads - full, ord = 2).item(), 
                                                                          torch.linalg.vector_norm(grads - curr, ord = 2).item(),)
            
        else:
            save_tensor(grads, "gradients", f"{self.NAME}", f"full_{str(self.__CURREPOCH)}_{id}")
        
        save_tensor(grads, "gradients", f"{self.NAME}", f"curr_{str(self.__CURREPOCH)}_{id}")

        return

    def _capture_grads(self):
        """
        Returns a vector of the masked gradients of all weights and biases of linear and convolutional layers.
        """
        
        grad_w = list()

        for name, param in self.m.named_parameters():
            
            if name.endswith("norm"): continue

            grad_w.append(param.grad.detach().cpu().flatten())   

        return torch.cat(grad_w)
    
    def pre_epoch_hook(self, epoch: int):
        self.__CURREPOCH = epoch
        return
    
    def pre_step_hook(self, step: int, train_cardinality: int):
        self.SAVE_GRADS = ((step + 1) % 25 == 0 or (step + 1 == train_cardinality)) and (self.__CURREPOCH % 4 == 3) # Save Gradients every 4th epoch, 16 times.
        return
    
    def post_prune_hook(self, iteration: int, num_epochs: int):
        self.__CURR_IMP_ITER = iteration
        self.grad_captures[iteration] = [defaultdict(tuple)] * num_epochs
        return

    def pre_IMP_hook(self, name: str):
        self.__CURR_IMP_ITER = 0
        self.NAME = name
        open(f'tensor_swap/gradients/{name}.h5', 'w').close()
        return

    def post_IMP_hook(self):
        self.__CURR_IMP_ITER = None
        self.__CURREPOCH = None

        """
        tmp = [None] * dist.get_world_size()
        dist.gather_object(self.grad_captures,
                           tmp if self.IsRoot else None,
                           dst = 0)
        self.grad_captures = tmp
        """

        return 