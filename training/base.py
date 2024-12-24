import torch
from torch import nn
import torch.nn.functional as F

import torch.distributed as dist

import torch.multiprocessing as mp 

import numpy as np

import torchinfo
from tqdm import tqdm

from typing import Callable, Tuple
from collections import defaultdict
import time
import math

import h5py
from models.LotteryLayers import Lottery
from models.base import BaseModel

class BaseCNNTrainer:

    #------------------------------------------ MAIN INIT FUNCTIONS -------------------------------------- #

    def __init__(self, model: torch.nn.parallel.DistributedDataParallel, 
                 rank: int, world_size: int):
        """
        Model has to be DDP.
        Non-Distributed Environments Not Supported.
        Default cuda device should be set.
        """
        self.m = model
        self.mm: BaseModel  = model.module
        self.RANK = rank
        self.IsRoot = rank == 0
        self.WORLD_SIZE = world_size

    def build(self, optimizer: torch.optim.Optimizer, 
              collective_transforms: Tuple[nn.Module],
              train_transforms: Tuple[nn.Module],
              eval_transforms: Tuple[nn.Module],
              final_collective_transforms: Tuple[nn.Module],
              loss: Callable = nn.CrossEntropyLoss(reduction = "sum"), 
              scale_loss: bool = False, decay: float = 0.0,
              gradient_clipnorm: float = float('inf'), ):
        
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
    
    def correct_k(self, output: torch.Tensor, labels: torch.Tensor, topk: int = 1) -> torch.Tensor:
        """
        Returns number of correct prediction.
        Deprecates output tensor.
        """
        with torch.no_grad():
            _, output = output.topk(topk, 1)
            output.t_()
            output.eq_(labels.view(1, -1).expand_as(output))
            return output[:topk].view(-1).to(torch.int64).sum(0)

    def metric_results(self) -> dict[str, float]:
        """
        Return Loss and Accuracy. 
        Should be called from root process.
        """
        with torch.no_grad():
            return {"loss": (self.eloss.div(self.ecount).detach().item()),
                "accuracy": (self.eacc.div(self.ecount).detach().item())}
    
    def reset_metrics(self):
        with torch.no_grad():
            self.eloss.fill_(0.0)
            self.eacc.fill_(0)
            self.ecount.fill_(0)
            self.reset_running_metrics()

    def reset_running_metrics(self):
        """
        Reset Loss, Accuracy, and Sample Count.
        Should be called from all processes.
        """
        with torch.no_grad():
            self.loss_tr.fill_(0.0)
            self.acc_tr.fill_(0)
            self.count_tr.fill_(0)
    
    def transfer_metrics(self):
        """
        Move from running metrics.
        This
        """
        with torch.no_grad():
            self._collect_metrics()
            self.eloss += self.loss_tr
            self.eacc += self.acc_tr
            self.ecount += self.count_tr
            self.reset_running_metrics()

    def _collect_metrics(self):
        """
        Collects Loss, Accuracy, and Sample Count.
        Do not directly call. Use transfer_metrics instead.
        """
        with torch.no_grad():
            dist.all_reduce(self.loss_tr, op = dist.ReduceOp.SUM)
            dist.all_reduce(self.acc_tr, op = dist.ReduceOp.SUM)
            dist.all_reduce(self.count_tr, op = dist.ReduceOp.SUM)

    #------------------------------------------ MAIN TRAIN FUNCTIONS -------------------------------------- #

    def fit(self, train_data: torch.utils.data.DataLoader, validation_data: torch.utils.data.DataLoader,
            epochs: int, train_cardinality: int, name: str, accumulation_steps: int = 1, save: bool = True, 
            verbose: bool = True, rewind_iter: int = 500) -> dict:
        """
        Basic Training Run Implementation.
        Override by Copy and Pasting.

        train_data and validation_data are expected to be sharded and batched, and use DistributedSampler

        train_cardinality is the number of batches of the train set. It is used for logging.

        """

        if self.IsRoot: 
            progress_bar = tqdm(total = epochs, unit = "epoch", colour = "green", bar_format="{l_bar}{bar:25}{r_bar}{bar:-25b}", leave = False)
            print("\n\n\n\n")


        logs = defaultdict(dict)
        self.reset_metrics()
        #if save: self.save_ckpt(name = name, prefix = "init")
        
        best_val_loss = float('inf')
        best_epoch = 1

        train_start = None
        val_start = None


        for epoch in range(epochs):

            #if verbose: self.print(f"\nStarting Epoch {epoch + 1}\n", 'red')
            train_start = time.time()
            self.m.train()

            self.pre_epoch_hook(epoch)

            accum = False

            self.reset_metrics()

            train_data.sampler.set_epoch(epoch)

            for step, (x, y, *_) in enumerate(train_data):

                iter = int(epoch * train_cardinality + step + 1)
                accum = ((step + 1) % accumulation_steps == 0) or (step + 1 == train_cardinality)

                if (iter - 1) == rewind_iter and save:
                    self.save_ckpt(name = name, prefix = "rewind")

                self.pre_step_hook(step, train_cardinality)

                dist.barrier(device_ids = [self.RANK])

                x, y = x.to('cuda'), y.to('cuda')

                for T in self.cT: x = T(x) # Transforms
                for T in self.tT: x = T(x)
                for T in self.fcT: x = T(x)

                self.train_step(x, y, accum, accumulation_steps)

                self.post_step_hook(x = x, y = y, _ = _, step = step, train_cardinality = train_cardinality, epoch = epoch)

                if accum:
                    
                    if (step + 1) % 48 == 0 or (step + 1 == train_cardinality): # Synchronize and Log.
                        
                        self.transfer_metrics()
                        
                        if self.IsRoot: logs[iter] = self.metric_results()
                        
                        if (step + 1) % 48 == 0 and self.IsRoot and verbose:
                            
                            self.print(f"----  Status at {math.ceil((step + 1) / 48):.0f}/8: ----     Accuracy: {logs[iter]['accuracy']:.4f}   --  Loss: {logs[iter]['loss']:.5f} --", 'white')

            self.post_train_hook()

            #if verbose: self.print(f"Training stage took {(time.time() - train_start):.1f} seconds.", 'yellow')

            val_start = time.time()

            validation_data.sampler.set_epoch(epoch)

            dist.barrier(device_ids = [self.RANK])

            self.evaluate(validation_data)

            if self.IsRoot:

                logs[(epoch + 1) * train_cardinality].update({('val_' + k): v for k, v in self.metric_results().items()})

                if logs[(epoch + 1) * train_cardinality]['val_loss'] < best_val_loss:
                    best_val_loss = logs[(epoch + 1) * train_cardinality]['val_loss']
                    #if verbose: self.print(f"\n -- UPDATING BEST WEIGHTS TO {epoch + 1} -- \n", "magenta")
                    best_epoch = epoch + 1
                    self.save_ckpt(name = name, prefix = "best")

                #if verbose: self.print(f"Validation stage took {(time.time() - val_start):.1f} seconds.", 'yellow')
                
                self.clear_lines(6)

                print(self.color_str("\nEpoch ", "white") + self.color_str(epoch + 1, "orange")+ self.color_str(f"/{epochs}: ", "white"))
                for k, v in logs[(epoch + 1) * train_cardinality].items():
                    self.print(f" {k}: {v:.9f}", 'cyan')

                progress_bar.set_postfix_str(self.color_str(f"Time Taken: {(time.time() - train_start):.2f}s", "green") + ", " + self.color_str(f"Best Epoch: {best_epoch}", "green")) 

                progress_bar.update(1)

                #self.print(f"Total for Epoch: {(time.time() - train_start):.1f} seconds.", 'yellow')

            self.post_epoch_hook(epoch)

        if self.IsRoot:

            progress_bar.close()



        return logs
 

    #@torch.compile
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
    

    #------------------------------------------ TRAINING HOOKS ----------------------------------------------- #

    def pre_epoch_hook(self, epoch: int) -> None:
        pass
    
    def post_epoch_hook(self, epoch) -> None:
        pass
    
    def pre_step_hook(self, step: int, steps_per_epoch: int) -> None:
        pass

    def post_step_hook(self, x, y, _, step, train_cardinality, epoch) -> None:
        pass

    def post_train_hook(self) -> None:
        pass

    #------------------------------------------ MAIN EVALUATE FUNCTIONS -------------------------------------- #

    #@torch.compile
    def evaluate(self, test_data: torch.utils.data.DataLoader) -> None:
        """
        Evaluate model on dataloader.
        To retrieve results, call metric_results()
        """
        with torch.no_grad():

            self.m.eval()
            self.reset_metrics()

            for x, y, *_ in test_data:

                x, y = x.to('cuda'), y.to('cuda')
                
                for T in self.cT: x = T(x) # Transforms
                for T in self.eT: x = T(x)
                for T in self.fcT: x = T(x)

                self.test_step(x, y)
            
            self.transfer_metrics()
        #self.print(f"Evaluated on {self.ecount.detach().item()} samples.")
        return

    #@torch.compile
    def test_step(self, x: torch.Tensor, y: torch.Tensor) -> None:
        
        with torch.no_grad():

            output = self.m(x)
            loss = self.criterion(output, y)

            self.loss_tr += loss
            self.acc_tr += self.correct_k(output, y)
            self.count_tr += y.size(dim = 0)

        return
        

    #------------------------------------------ SERIALIZATION FUNCTIONS -------------------------------------- #

    
    def save_ckpt(self, name: str, prefix: str = None):
        """
        Saves Model, Optimizer, and LossScaler state_dicts.
        Can be accessed with same name and prefix from load_ckpt.
        """
        with torch.no_grad():
            if not self.IsRoot: return
            fp = f"./logs/WEIGHTS/{self.fromNamePrefix(name, prefix)}.pt"
            ckpt = {'model': self.m.state_dict(),
                    'optim': self.optim.state_dict(),
                    'scaler': self.lossScaler.state_dict()}
            torch.save(ckpt, fp)        

    def load_ckpt(self, name: str, prefix: str = None):
        """
        Loads Model, Optimizer, and LossScaler state_dicts.
        Meant to be used with save_ckpt.
        """
        with torch.no_grad():
                
            fp = f"./logs/WEIGHTS/{self.fromNamePrefix(name, prefix)}.pt"

            dist.barrier(device_ids = [self.RANK])
            
            ckpt = torch.load(fp, map_location = {'cuda:%d' % 0: 'cuda:%d' % self.RANK},
                            weights_only = True) # Assumes rank = 0 is Root.
            self.m.load_state_dict(ckpt['model'])
            self.optim.load_state_dict(ckpt['optim'])
            self.lossScaler.load_state_dict(ckpt['scaler'])

    #------------------------------------------ HELPER FUNCTIONS -------------------------------------- #

    ### Logging

    def color_str(self, input: str, color: str):
        return self._COLORS[color] + str(input) + self._COLORS["reset"]

    def print(self, input, color: str = 'default' , rank: int = 0) -> None:
        """
        Prints with color. Make sure input has a __repr__ attribute.
        """
        if not self.RANK == rank: return
        print(self.color_str(input, color))
        return
    
    def fromNamePrefix(self, name: str, prefix: str):
        return f"{prefix}_{name}" if prefix else name

    def summary(self, batch_size):
        """
        Prints torchinfo summary of model.
        """
        if not self.IsRoot: return
        torchinfo.summary(self.mm, (batch_size, 3, 224, 224))

    ### Training

    def reset_loss_scaler(self):
        """
        Reset Cuda Loss Scaler - Use with Mixed Precision.
        Should be reset every 
        """
        self.lossScaler = torch.amp.GradScaler(device = 'cuda', enabled = self.AMP)

    def reduce_learning_rate(self, factor: int):
        with torch.no_grad():    
            for pg in self.optim.param_groups:
                pg['lr'] /= factor

    def clear_lines(self, n: int):
        """
        Logging Util
        """
        for _ in range(n):
            print("\033[F\033[K", end="")




class BaseIMP(BaseCNNTrainer):

    def __init__(self, model: torch.nn.parallel.DistributedDataParallel, rank: int):
        super(BaseIMP, self).__init__(model, rank)
        self.IsTicketRoot = rank == 2

    #------------------------------------------ LTH IMP FUNCTIONS -------------------------------------- #

    def TicketIMP(self, train_data: torch.utils.data.DataLoader, validation_data: torch.utils.data.DataLoader, 
                  epochs_per_run: int, train_cardinality: int, name: str, prune_rate: float, prune_iters: int,
                  rewind_iter: int = 500):

        """
        Find Winning Ticket Through IMP with Rewinding. Calls Trainer.fit(); See description for argument requirements.

        prune_rate should be a float that indicates what percent of weights to prune per training run. 
        I.E. to prune 20% per iteration, prune_rate = 0.2

        prune_iters should be the number of iterations to run IMP for.
        
        Final sparsity = prune_rate ** prune_iters
        """
        
        total_logs = defaultdict()
        
        sparsities_d = [None] * (prune_iters + 1)
        current_sparsity = 100.0
        sparsities_d[0] = current_sparsity / 100

        if self.IsRoot: h5py.File(f"./logs/TICKETS/{name}.h5", "w").close() # For logging tickets.

        self.pre_IMP_hook(name)

        self.print(f"\nRUNNING IMP ON {self.mm.num_prunable} PRUNABLE WEIGHTS.", "pink")

        self.print(f"\nSPARSITY: {current_sparsity:.2f}\n", "red")

        total_logs[0] = self.fit(train_data, validation_data, epochs_per_run, train_cardinality, name + f"_{(100.0):.2f}", save = True, verbose = False, rewind_iter = rewind_iter)
        
        for iteration in range(1, prune_iters + 1):
            
            self.mm.prune_by_mg(prune_rate, iteration, root = 0)
            
            current_sparsity = self.mm.sparsity

            sparsities_d[iteration] = current_sparsity.item() / 100

            self.print(f"\nSPARSITY: {current_sparsity:.2f}\n", "red")

            self.post_prune_hook(iteration, epochs_per_run)

            self.mm.export_ticket(name, entry_name = f"{(current_sparsity):.2f}")

            self.load_ckpt(name + f"_{(100.0):.2f}", prefix = "rewind") 

            total_logs[iteration] = self.fit(train_data, validation_data, epochs_per_run, train_cardinality, 
                                             name + f"_{(current_sparsity):.2f}", save = False, verbose = False)

        self.post_IMP_hook()

        return total_logs, sparsities_d
    
    #--------------------------------------------------------------- IMP HOOKS --------------------------------------------------------------#
    
    def pre_IMP_hook(self, name: str) -> None:
        pass    

    def post_IMP_hook(self) -> None:
        pass

    def post_prune_hook(self, iteration: int, num_epochs: int) -> None:
        pass



class CNN_DGTS(BaseCNNTrainer):

    """
    Genetic Search Over CNN
    Implements Distributed RedBlackBst
    """

    """
    Monitor: KL Percentage Differnce
    """

    class DGTS:

        def __init__(self, rank: int, world_size: int, lock, dynamic_list, 
                     model: BaseModel, act_w: list, max_size: int = 50):
            self.lock = lock
            self.population = dynamic_list
            del self.population[:]
            self.RANK = rank
            self.WORLD_SIZE = world_size
            self.MAX_SIZE = max_size
            self.mm = model
            self.act_w = act_w

        def add_sample(self, mask: torch.Tensor, fitness: float):

            mask.share_memory_()

            with self.lock:

                left, right = 0, len(self.population) - 1
                while left <= right:
                    mid = (left + right) // 2
                    if self.population[mid][1] > fitness:
                        left = mid + 1
                    else: 
                        right = mid - 1

                self.population.insert(left, (mask, fitness, ))


        def distribute_parents(self):
            """
            Samples Parents - Linear Rank based from SUS sample
            """
            
            fitnesses = [fitness for mask, fitness in self.population]

            total_fitness = sum(fitnesses)
            pointer_distance = total_fitness / (len(self.population) // self.WORLD_SIZE) 
            start_point = (torch.rand(1, ) * pointer_distance).item()
            pointers = [start_point + n * pointer_distance for n in range(len(self.population) // self.WORLD_SIZE)]

            for i in range(1, len(fitnesses)):
                fitnesses[i] += fitnesses[i - 1]

            sample = list()
            ptr = 0
            for i, cum_fit in enumerate(fitnesses):
                while ptr < (len(self.population) // self.WORLD_SIZE) and pointers[ptr] <= cum_fit:
                    sample.append(self.population[i])
                    ptr += 1


            probs = np.arange(len(sample), 0, step = -1, dtype = np.float32)
            probs /= probs.sum()
            selection = np.random.choice(len(sample), replace = False, p = probs, size = 2)

            return (sample[selection[0]], sample[selection[1]])
        
        def clean_population(self):
            if (self.RANK != 0): return
            while len(self.population) > self.MAX_SIZE:
                del self.population[-1]
    
        def search_step(self, full_acts: torch.Tensor, inp: torch.Tensor, mutation_rate = 0.1):
            
            dist.barrier(device_ids = [self.RANK])

            with torch.no_grad():

                parents = self.distribute_parents()
                fitness_sum = parents[0][1] + parents[1][1] 
                child = self.mm.merge_tickets(parents[0][0], parents[1][0], 
                                              (parents[0][1]/fitness_sum), 
                                              (parents[1][1]/fitness_sum))

                if (torch.rand(1, ).item() < mutation_rate): child = self.mm.mutate_ticket(child)

                self.mm.set_ticket(child)
                self.mm(inp)
                
                curr_activations = torch.cat(self.act_w)
                self.act_w.clear()
                curr_activations.div_(curr_activations.sum())

                fitness = F.kl_div((curr_activations + 1e-10).log(), full_acts + 1e-10, reduction = "batchmean").item()

                self.add_sample(child.cpu(), fitness)
                dist.barrier(device_ids = [self.RANK])
                self.clean_population()
                dist.barrier(device_ids = [self.RANK])

        def search(self, sparsity: float, max_iterations: int, full_acts: torch.Tensor, inp: torch.Tensor):

            with torch.no_grad():

                self.mm.eval()

                for _ in range(2):
                    self.mm.reset_ticket()
                    self.mm.prune_random(sparsity, distributed = False)
                    sample = self.mm.export_ticket_cpu().clone()

                    self.mm(inp)

                    curr_activations = torch.cat(self.act_w)
                    self.act_w.clear()
                    curr_activations.div_(curr_activations.sum())

                    fitness = F.kl_div((curr_activations + 1e-10).log(), full_acts + 1e-10, reduction = "batchmean").item()

                    self.add_sample(sample.cpu(), fitness)

                dist.barrier(device_ids = [self.RANK])

                for it in range(max_iterations):
                    self.search_step(full_acts, inp)

                output = self.population[0]

                dist.barrier(device_ids = [self.RANK])

                del self.population[:] 

            return output # ( TICKET, FITNESS )


    

    def __init__(self, *args, lock, dynamic_list, **kwargs):

        """
        self._capture_layers = list[module]
        self._fcapture_layers = list[(act_func, module)]
        """

        super(CNN_DGTS, self).__init__(*args, **kwargs)
        self.LOCK = lock
        self.SHARED_LIST = dynamic_list
        
        self._act_w = list()
        self._fitness_monitor = list() #[fitnesses]

        self._capture_layers = list() if not hasattr(self, "_capture_layers") else self._capture_layers
        self._fcapture_layers = list() if not hasattr(self, "_fcapture_layers") else self._fcapture_layers
        self._handles = list()



    def build(self, sparsity_rate, *args, **kwargs):
        super().build(*args, **kwargs)
        self.sparsity_rate = sparsity_rate

    def init_capture_hooks(self): 
        for layer in self._capture_layers:
            self._handles.append(layer.register_forward_hook(self._capture_hook))
        for layer, func in self._fcapture_layers:
            self._handles.append(layer.register_forward_hook(lambda *args, **kwargs: self._fake_capture_hook(func, *args, **kwargs)))
        return 
    
    def remove_handles(self):
        for handle in self._handles: handle.remove()
        self._handles.clear()

    def _capture_hook(self, module, input, output: torch.Tensor):
        self._act_w.append(output.detach().to(torch.float64).mean(dim = 0).view(-1))

    def _fake_capture_hook(self, func, module, input, output: torch.Tensor):
        self._act_w.append(func(output.detach().to(torch.float64)).mean(dim = 0).view(-1))

    def post_step_hook(self, x, y, _, step, train_cardinality, epoch):
        if (step + 2) % 128 == 7 and (step + 2 != 7):#(step + 2) == train_cardinality:
            #self.print("init hooks", color = "red") 
            self.init_capture_hooks()
        elif (step + 1) % 128 == 7 and (step + 1 != 7):#((step + 1) == train_cardinality):
            ticket, fitness = self.search(x, y)
            self.remove_handles()
            self._fitness_monitor.append(((step + 1 + train_cardinality * epoch), fitness))
            #self.print("added fitness", color = "red")

    def search(self, x, y):
        with torch.no_grad():

            self.m.eval()

            self.GTS = self.DGTS(self.RANK, self.WORLD_SIZE, 
                                self.LOCK, self.SHARED_LIST,
                                self.mm, act_w = self._act_w,
                                max_size = 50)
            
            activation_mask = None
            if len(self._act_w) != 0: 
                activation_mask = torch.cat(self._act_w)
                self._act_w.clear()

            return self.GTS.search(self.mm.sparsity * self.sparsity_rate, max_iterations = 48, full_acts = activation_mask, inp = x)