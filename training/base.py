import torch
from torch import nn
import torch.nn.functional as F

import torch.distributed as dist

import torch.multiprocessing as mp 

import numpy as np
import copy

import torchinfo
from tqdm import tqdm

from typing import Callable, Tuple
from collections import defaultdict
import time
import math

import h5py
from models.LotteryLayers import Lottery
from models.base import BaseModel

from data.cifar10 import custom_fetch_data

class BaseCNNTrainer:

    #------------------------------------------ MAIN INIT FUNCTIONS -------------------------------------- #

    def __init__(self, model, 
                 rank: int, world_size: int):
        """
        Model has to be DDP.
        Non-Distributed Environments Not Supported.
        Default cuda device should be set.
        """

        self.RANK = rank
        self.IsRoot = rank == 0
        self.WORLD_SIZE = world_size
        self.DISTRIBUTED = world_size > 1

        self.m = model
        if self.DISTRIBUTED: self.mm  = getattr(model, 'module')
        else: self.mm = model

    def build(self, optimizer,
              optimizer_kwargs: dict, 
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
        self.optimizer = optimizer
        self.optimizer_kwargs = optimizer_kwargs

        self.reset_optimizer()

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
        if not self.DISTRIBUTED: return
        with torch.no_grad():
            dist.all_reduce(self.loss_tr, op = dist.ReduceOp.SUM)
            dist.all_reduce(self.acc_tr, op = dist.ReduceOp.SUM)
            dist.all_reduce(self.count_tr, op = dist.ReduceOp.SUM)

    #------------------------------------------ MAIN TRAIN FUNCTIONS -------------------------------------- #

    def fit(self, train_data: torch.utils.data.DataLoader, validation_data: torch.utils.data.DataLoader,
            epochs: int, train_cardinality: int, name: str, accumulation_steps: int = 1, save: bool = True, 
            verbose: bool = False, rewind_iter: int = 500, sampler_offset: int = 0, validate: bool = True, 
            start = 0, save_init = True) -> dict:
        """
        Basic Training Run Implementation.
        Override by Copy and Pasting.

        train_data and validation_data are expected to be sharded and batched, and use DistributedSampler

        train_cardinality is the number of batches of the train set. It is used for logging.

        """

        if self.IsRoot: 
            progress_bar = tqdm(total = epochs, unit = "epoch", colour = "green", bar_format="{l_bar}{bar:25}{r_bar}{bar:-25b}", leave = False, smoothing = 0.8)
            progress_bar.update(start)
            print(f"\n\n\n\n")


        logs = defaultdict(dict)
        self.reset_metrics()
        if save_init: self.save_ckpt(name = name, prefix = "init")
        
        best_val_loss = float('inf')
        best_epoch = 1

        train_start = None
        val_start = None

        _break = False

        for epoch in range(start, epochs):

            #if verbose: self.print(f"\nStarting Epoch {epoch + 1}\n", 'red')
            train_start = time.time()
            self.m.train()

            data_list = self.pre_epoch_hook(train_data)

            accum = False

            self.reset_metrics()

            train_data.sampler.set_epoch(epoch + train_cardinality * sampler_offset)

            for step, (x, y, *_) in enumerate(train_data):
                if _break: continue
                iter = int(epoch * train_cardinality + step + 1)
                accum = ((step + 1) % accumulation_steps == 0) or (step + 1 == train_cardinality)

                if (iter - 1) == rewind_iter and save:
                    self.save_ckpt(name = name, prefix = "rewind")

                self.pre_step_hook(step, train_cardinality)

                #dist.barrier(device_ids = [self.RANK])

                x, y = x.to('cuda'), y.to('cuda')

                for T in self.cT: x = T(x) # Transforms
                for T in self.tT: x = T(x)
                for T in self.fcT: x = T(x)

                self.train_step(x, y, accum, accumulation_steps)

                if not self.post_step_hook(data_list = data_list, step = step, iteration = iter, name = name): 
                    #if self.IsRoot: progress_bar.close()
                    _break = True

                if accum:
                    
                    if (step + 1) % 13 == 0 or (step + 1 == train_cardinality): # Synchronize and Log.
                        
                        self.transfer_metrics()
                        
                        if self.IsRoot: logs[iter] = self.metric_results()
                        
                        if (step + 1) % 52 == 0 and self.IsRoot and verbose:
                            
                            self.print(f"----  Status at {math.ceil((step + 1) / 52):.0f}/8: ----     Accuracy: {logs[iter]['accuracy']:.4f}   --  Loss: {logs[iter]['loss']:.5f} --", 'white')
            
            if _break: break

            self.post_train_hook()


            #if verbose: self.print(f"Training stage took {(time.time() - train_start):.1f} seconds.", 'yellow')
            if validate:
                #val_start = time.time()

                validation_data.sampler.set_epoch(epoch + epochs * sampler_offset)

                if self.DISTRIBUTED: dist.barrier(device_ids = [self.RANK])

                self.evaluate(validation_data)

                if self.IsRoot:

                    logs[(epoch + 1) * train_cardinality].update({('val_' + k): v for k, v in self.metric_results().items()})

                    if logs[(epoch + 1) * train_cardinality]['val_loss'] < best_val_loss:
                        best_val_loss = logs[(epoch + 1) * train_cardinality]['val_loss']
                        #if verbose: self.print(f"\n -- UPDATING BEST WEIGHTS TO {epoch + 1} -- \n", "magenta")
                        best_epoch = epoch + 1
                        #self.save_ckpt(name = name, prefix = "best")

                    #if verbose: self.print(f"Validation stage took {(time.time() - val_start):.1f} seconds.", 'yellow')
                    
                    #self.print(f"Total for Epoch: {(time.time() - train_start):.1f} seconds.", 'yellow')

            if self.IsRoot:

                self.clear_lines(6 if validate else 4)

                print(self.color_str("\nEpoch ", "white") + self.color_str(epoch + 1, "orange")+ self.color_str(f"/{epochs}: ", "white"))
                for k, v in logs[(epoch + 1) * train_cardinality].items():
                    self.print(f" {k}: {v:.9f}", 'cyan')

                progress_bar.set_postfix_str(self.color_str(f"Time Taken: {(time.time() - train_start):.2f}s", "green") + ", " + self.color_str(f"Sparsity: {self.mm.sparsity.item()}", "green")) 

                progress_bar.update(1)

            self.post_epoch_hook(epoch, epochs)

        if self.IsRoot:

            progress_bar.close()

        if save: self.save_ckpt(name = name, prefix = "final")

        return logs
 

    #@torch.compile
    def train_step(self, x: torch.Tensor, y: torch.Tensor, accum: bool = True, accum_steps: int = 1, id: str = None):
        
        with torch.autocast('cuda', dtype = torch.float16, enabled = self.AMP):

            output = self.m(x)
            loss = self.criterion(output, y) #+ self.LD * self.calculate_custom_regularization()
            loss /= accum_steps

        if not accum:
            
            if self.DISTRIBUTED:
                with self.m.no_sync():
                    self.lossScaler.scale(loss).backward()
            else:
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

    def pre_epoch_hook(self, *args) -> None:
        pass
    
    def post_epoch_hook(self, *args, **kwargs) -> None:
        pass
    
    def pre_step_hook(self, *args, **kwargs) -> None:
        pass

    def post_step_hook(self, *args, **kwargs) -> bool:
        return True

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

    def _from_ddp_ckpt(self, state_dict):
        return dict((k[7:] if k.startswith("module.") else k, v) for k, v in state_dict.items())

    def _to_ddp_ckpt(self, state_dict):
        return dict((k if k.startswith("module.") else f"module.{k}", v) for k, v in state_dict.items())

    
    def save_ckpt(self, name: str, prefix: str = None):
        """
        Saves Model, Optimizer, and LossScaler state_dicts.
        Can be accessed with same name and prefix from load_ckpt.
        """
        with torch.no_grad():
            if not self.IsRoot: return
            fp = f"./logs/WEIGHTS/{self.fromNamePrefix(name, prefix)}.pt"
            ckpt = {'model': self.m.state_dict() if self.DISTRIBUTED else self._to_ddp_ckpt(self.m.state_dict()),
                    'optim': self.optim.state_dict(),
                    'scaler': self.lossScaler.state_dict()}
            torch.save(ckpt, fp)        

    def load_ckpt(self, name: str, prefix: str = None, zero_out = True, weights_only = True):
        """
        Loads Model, Optimizer, and LossScaler state_dicts.
        Meant to be used with save_ckpt.
        """
        with torch.no_grad():

            fp = f"./logs/WEIGHTS/{self.fromNamePrefix(name, prefix)}.pt"
            
            if self.DISTRIBUTED: 
                dist.barrier(device_ids = [self.RANK])
                ckpt = torch.load(fp, map_location = {'cuda:%d' % 0: 'cuda:%d' % self.RANK},
                            weights_only = True) # Assumes rank = 0 is Root.
            else:
                ckpt = torch.load(fp, weights_only = True)
                ckpt['model'] = self._from_ddp_ckpt(ckpt['model'])

            self.m.load_state_dict(ckpt['model'])
            #ckpt['optim']['param_groups'] =self.optim.state_dict()['param_groups']
            if weights_only:
                self.reset_optimizer()
                self.reset_loss_scaler()
            else:
                self.optim.load_state_dict(ckpt['optim'])
                self.lossScaler.load_state_dict(ckpt['scaler'])
            
            #if zero_out:
            #    self.prune_model(self.mm.export_ticket_cpu())

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

    def reset_optimizer(self):
        self.optim = self.optimizer(self.m.parameters(), **self.optimizer_kwargs)

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

    def prune_model(self, ticket: torch.Tensor):
    
        #old_sp = self.mm.sparsity_d.item()
        self.mm.set_ticket(ticket, zero_out = True)
        #new_sp = self.mm.sparsity_d.item()
        #self.adjust_weight_decay(factor = (1 - (1 - new_sp/old_sp)/10))

    def adjust_weight_decay(self, factor: float):
        """
        factor = 0.1 -> new_decay = 0.1 * old_decay
        """
        with torch.no_grad():
            for pg in self.optim.param_groups:
                wd = pg['weight_decay']
                pg['weight_decay'] = wd * (1 - factor / (1 + wd))


class BaseIMP(BaseCNNTrainer):

    #------------------------------------------ LTH IMP FUNCTIONS -------------------------------------- #

    def _get_results(self, train_data, validation_data):
        self.m.eval()
        self.evaluate(train_data)
        if self.RANK == 0:
            train_res = self.metric_results()
            print("Train Results: ", train_res)
        self.evaluate(validation_data)
        if self.RANK == 0:
            val_res =  self.metric_results()
            print("Validation Results: ", val_res)
            train_res.update({('val_' + k): v for k, v in val_res.items()})
        self.m.train()
        return train_res

    def TicketIMP(self, train_data: torch.utils.data.DataLoader, validation_data: torch.utils.data.DataLoader, 
                  epochs_per_run: int, train_cardinality: int, name: str, prune_rate: float, prune_iters: int,
                  rewind_iter: int = 500, validate = True):

        """
        Find Winning Ticket Through IMP with Rewinding. Calls Trainer.fit(); See description for argument requirements.

        prune_rate should be a float that indicates what percent of weights to prune per training run. 
        I.E. to prune 20% per iteration, prune_rate = 0.8

        prune_iters should be the number of iterations to run IMP for.
        
        Final sparsity = prune_rate ** prune_iters
        """
        
        total_logs = defaultdict()
        results = defaultdict()
        
        sparsities_d = [None] * (prune_iters + 1)
        current_sparsity = 100.0
        sparsities_d[0] = current_sparsity / 100

        sampler_offset = 0

        if self.IsRoot: h5py.File(f"./logs/TICKETS/{name}.h5", "w").close() # For logging tickets.

        self.pre_IMP_hook(name)

        self.print(f"\nRUNNING IMP ON {self.mm.num_prunable} PRUNABLE WEIGHTS.", "pink")

        self.print(f"\nSPARSITY: {current_sparsity:.2f}\n", "red")

        total_logs[0] = self.fit(train_data, validation_data, epochs_per_run, train_cardinality, name + f"_{(100.0):.2f}", save = True, verbose = False,
                                 rewind_iter = rewind_iter, sampler_offset = sampler_offset, validate = validate)
        
        if not validate: results[0] = self._get_results(train_data, validation_data)

        for iteration in range(1, prune_iters + 1):

            self.mm.prune_by_mg(prune_rate, iteration, root = 0)
            
            current_sparsity = self.mm.sparsity

            sparsities_d[iteration] = current_sparsity.item() / 100

            self.print(f"\nSPARSITY: {current_sparsity:.2f} | SEEDED: {torch.rand(1).item()}\n", "red")

            self.post_prune_hook(iteration, epochs_per_run)

            self.mm.export_ticket(name, entry_name = f"{(current_sparsity):.2f}")

            self.load_ckpt(name + f"_{(100.0):.2f}", prefix = "rewind", weights_only = True)

            #sampler_offset += 1

            total_logs[iteration] = self.fit(train_data, validation_data, epochs_per_run, train_cardinality, 
                                             name + f"_{(current_sparsity):.2f}", save = False, verbose = False,
                                             sampler_offset = sampler_offset, start = rewind_iter//train_cardinality,
                                             save_init = False, validate = validate)
            
            if not validate: results[iteration] = self._get_results(train_data, validation_data)

        self.post_IMP_hook()

        if not validate: total_logs = (total_logs, results)

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
                     model: BaseModel, act_w: list, max_size: int = 50,
                     possible_children: int = 5, reverse_scores: bool = False):
            self.lock = lock
            self.population = dynamic_list
            del self.population[:]
            self.RANK = rank
            self.WORLD_SIZE = world_size
            self.MAX_SIZE = max_size
            self.POSSIBLE_CHILDREN = possible_children
            self.mm = model
            self.act_w = act_w
            self.reverse = reverse_scores
            #self.__div_strat = True

        def add_sample(self, mask: torch.Tensor, fitness: float):

            mask = mask.cpu()
            mask.share_memory_()

            with self.lock:

                left, right = 0, len(self.population) - 1
                while left <= right:
                    mid = (left + right) // 2
                    if (self.population[mid][1] < fitness) != self.reverse:
                        left = mid + 1
                    else: 
                        right = mid - 1

                self.population.insert(left, (mask, fitness, ))


        def distribute_parents(self):
            """
            Samples Parents - Linear Rank based from SUS sample
            """

            size = (len(self.population) // self.WORLD_SIZE) 

            max_fitness = self.population[-1][1] + 1e-11
            fitnesses = [fitness for mask, fitness in self.population] if self.reverse else [max_fitness - fitness for mask, fitness in self.population]

            total_fitness = sum(fitnesses)
            pointer_distance = total_fitness / size
            start_point = (torch.rand(1, ) * pointer_distance).item()
            pointers = [start_point + n * pointer_distance for n in range(size)]

            for i in range(1, len(fitnesses)):
                fitnesses[i] += fitnesses[i - 1]

            sample = list()
            ptr = 0
            for i, cum_fit in enumerate(fitnesses):
                while ptr < size and pointers[ptr] <= cum_fit:
                    sample.append(self.population[i])
                    ptr += 1          

            probs = np.arange(len(sample), 0, step = -1, dtype = np.float32)
            #if self.__div_strat: probs = np.sqrt(probs)
            probs /= probs.sum()
            selection = np.random.choice(len(sample), replace = False, p = probs, size = 2)

            return (sample[selection[0]], sample[selection[1]])
        
        def clean_population(self):
            if (self.RANK != 0): return
            del self.population[self.MAX_SIZE:]

        def _calculate_sample(self, sample: torch.Tensor, full_acts: torch.Tensor, inp: torch.Tensor):
            
            #torch.cuda.empty_cache()

            self.mm.set_ticket(sample)
            self.mm(inp)

            curr_activations = torch.cat(self.act_w)#torch.cat([i * t for i, t in enumerate(self.act_w, start=1)]) # Weight later activations higher, linearly
            self.act_w.clear()
            curr_activations += 1e-10
            curr_activations.div_(curr_activations.sum())
            curr_activations.log_()

            return F.kl_div(curr_activations, full_acts, reduction = "batchmean").item()
    
        def search_step(self, full_acts: torch.Tensor, inp: torch.Tensor, mutation_rate = 0.15, temperature_rate = 0.15):

            with torch.no_grad():

                parents = self.distribute_parents()
                num_children = torch.randint(1, self.POSSIBLE_CHILDREN + 1, (1,), device = "cpu").item()
                fitness_sum = parents[0][1] + parents[1][1] 
                zero_weight, one_weight = (torch.as_tensor(parents[0][1])/fitness_sum), (torch.as_tensor(parents[1][1])/fitness_sum)
                children = [self.mm.merge_tickets(parents[0][0].cuda(), parents[1][0].cuda(), 
                                              zero_weight if self.reverse else one_weight, # the better sample should be weighted with the worse fitness, so that it is greater
                                              one_weight if self.reverse else zero_weight) 
                                              for _ in range(num_children)]

                for child in children:
                    if (torch.rand(1, ).item() < mutation_rate): child = self.mm.mutate_ticket(child, temperature_rate)
                    self.add_sample(child, self._calculate_sample(child, full_acts, inp))

        def search(self, sparsity_rate: float, max_iterations: int, 
                   full_acts: torch.Tensor, inp: torch.Tensor,
                   extra_starters: list[torch.Tensor]):

            with torch.no_grad():

                self.mm.eval()

                original_ticket = self.mm.export_ticket_cpu()

                if (self.RANK == 2):
                    for sample in extra_starters: 
                        self.add_sample(sample, self._calculate_sample(sample, full_acts, inp))

                cnt = len(extra_starters) if (self.RANK == 2) else 0
                while cnt < (self.MAX_SIZE // 8):  # 4 * (1/2) * 4, 1/2 full
                    self.mm.set_ticket(original_ticket)
                    self.mm.prune_random(sparsity_rate, distributed = False)
                    sample = self.mm.export_ticket_cpu()
                    self.add_sample(sample, self._calculate_sample(sample, full_acts, inp))
                    cnt += 1

                if self.DISTRIBUTED: dist.barrier(device_ids = [self.RANK])

                for it in range(max_iterations):
                    self.search_step(full_acts, inp,
                                     0.2, 0.1)
                    if (it % 2) == 0:
                        if self.DISTRIBUTED: dist.barrier(device_ids = [self.RANK])
                        self.clean_population()
                        if self.DISTRIBUTED: dist.barrier(device_ids = [self.RANK])

                output = self.population[0]

                if self.DISTRIBUTED: dist.barrier(device_ids = [self.RANK])

                self.mm.set_ticket(original_ticket)

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
        self.prunes = list() #[fitnesses]
        self.fitnesses = list()

        self._capture_layers = list() if not hasattr(self, "_capture_layers") else self._capture_layers
        self._fcapture_layers = list() if not hasattr(self, "_fcapture_layers") else self._fcapture_layers
        self._handles = list()
        
        #self._prune_status = 0
        #self._last_prune = None
        self._pruned = False



    def build(self, sparsity_rate, experiment_args: list, type_of_exp = 1, reverse_scores = False, *args, **kwargs):#type_of_exp: int, experiment_args: list, *args, **kwargs):
        super().build(*args, **kwargs)
        self.sparsity_rate = sparsity_rate
        self.search_iterations = int(experiment_args[0])
        self.search_size = int(experiment_args[1])
        self.stopping_plateau = int(experiment_args[2])
        #self.steps_waiting = int(experiment_args[3])
        self.possible_children = int(experiment_args[3])
        self.desired_sparsity = float(experiment_args[4])
        self.reverse_scores = reverse_scores

        #if self.IsRoot: 
            #print("ARGUMENTS")
            #print("-------------")
            #print(f"Sparsity_rate: {self.sparsity_rate}")
            #print(f"Search Iterations: {self.search_iterations}")
            #print(f"Search Size: {self.search_size}")
        #if (type_of_exp == 1): 
        #if self.IsRoot: print(f"Plateau Epochs: {self.stopping_plateau}")
        self.post_step_hook = self.post_step_hook_single_plateau_iterative
        self.pre_epoch_hook = self.pre_epoch_hook_iterative

        if type_of_exp == 2: 
            self.post_step_hook = self.post_step_hook_break
            self.pre_epoch_hook = self.pre_epoch_hook_blank
            self.prune_iteration = self.stopping_plateau

        if type_of_exp == 3:
            self.post_step_hook = self.post_step_hook_monitor
            self.pre_epoch_hook = self.pre_epoch_hook_monitor
        #elif (type_of_exp == 2):
        #if self.IsRoot: print(f"Prune Iteration: {self.prune_iteration}")
        #self.post_step_hook = self.post_step_hook_single_shot_no_plateau
        #if self.IsRoot:
        #    print(f"Final Sparsity (leq): {self.desired_sparsity}")
        #    print("-------------")

        self._best_fitness = float("inf")
        self._best_ticket = None
        self._plat_eps = 0
        #self._prune_next = False


    def init_capture_hooks(self): 
        for layer in self._capture_layers:
            self._handles.append(layer.register_forward_hook(self._capture_hook))
        for layer, func in self._fcapture_layers:
            self._handles.append(layer.register_forward_hook(lambda *args, **kwargs: self._fake_capture_hook(func, *args, **kwargs)))
        return 
    
    def remove_handles(self):
        for handle in self._handles: handle.remove()
        self._handles.clear()

    def pre_epoch_hook_iterative(self, dt, *args):
        if self._pruned: return None
        return custom_fetch_data(dt, 4)

    def pre_epoch_hook_blank(self, *args):
        return None
    
    def pre_epoch_hook_monitor(self, dt, *args):
        return custom_fetch_data(dt, 2)

    def _capture_hook(self, module, input, output: torch.Tensor):
        tmp = output.detach().to(torch.float64).mean(dim = 0).view(-1)
        #tmp += 1e-10
        #tmp.div_(tmp.sum())
        self._act_w.append(tmp)

    def _fake_capture_hook(self, func, module, input, output: torch.Tensor):
        tmp = func(output.detach().to(torch.float64)).mean(dim = 0).view(-1)
        #tmp += 1e-10
        #tmp.div_(tmp.sum())
        self._act_w.append(tmp)
    """
    def post_step_hook_single_plateau_continue(self, x, y, _, step, iteration):

        if (self._prune_status == 2): 
            pass

        elif (self._prune_status == 0):
            
            if (step + 2) == 390: self.init_capture_hooks()
            
            elif (step + 1) == 390:
                status, ticket, fitness = self.plateau_monitor(x, y)
                self.remove_handles()
                self.fitnesses.append((iteration, fitness, self.mm.sparsity.item() * 0.8))
                if status:
                    self.prune_model(ticket)
                    self._last_prune = iteration
                    self.prunes.append(f"Epoch {(iteration + 1) / 391:.1f}: Pruned to {self.mm.sparsity.item():.3f} with fitness: {fitness}.")
                    self._prune_status = 1
            
        elif (self._prune_status == 1): 
 
            check_prune = lambda it, st: ((it - self._last_prune) >= self.steps_waiting and (st < 390))
            if self._prune_next: 
                ticket, fitness = self.search(x, y)
                self.remove_handles()
                self._prune_next = False
                self.prune_model(ticket)
                self._last_prune = iteration
                self.fitnesses.append((iteration, fitness, self.mm.sparsity.item()))
                self.prunes.append(f"Epoch {(iteration + 1) / 391:.1f}: Pruned to {self.mm.sparsity.item():.3f} with fitness: {fitness}.")
                if (self.mm.sparsity_d.item() <= self.desired_sparsity): self._prune_status = 2
            elif check_prune(iteration + 1, step + 1): 
                self.init_capture_hooks()
                self._prune_next = True
            

        return True
    """

    def post_step_hook_break(self, iteration, *args, **kwargs):
        if self._pruned or iteration != self.prune_iteration: return True
        return False

    def post_step_hook_monitor(self, data_list, iteration, step, *args, **kwargs):
        if (step % 196 == 0):
            x, y = data_list[step//196]
            x, y = x.to('cuda'), y.to('cuda')

            for T in self.cT: x = T(x)
            for T in self.tT: x = T(x)
            for T in self.fcT: x = T(x)
            
            self.init_capture_hooks()
            self.mm(x)

            _, fitness = self.search(x, y)
            self.remove_handles()
            self.fitnesses.append((iteration, fitness))
        
        return True
        

    """
    def post_step_hook_single_shot_no_plateau(self, x, y, _, step, iteration):
        
        if self._pruned or (iteration != self.prune_iteration): return True
        
        elif iteration == self.prune_iteration: 

            while self.mm.sparsity_d >= self.desired_sparsity:
                    
                with torch.no_grad():
                    self._act_w.clear()
                    self.m(x)
                
                ticket, fitness = self.search(x, y)
                
                self.mm.set_ticket(ticket)

                self.fitnesses.append((self.mm.sparsity.item(), fitness))
                self.prunes.append(f"Epoch {(iteration + 1) / 391}: Pruned to {self.mm.sparsity:.3f} with fitness: {fitness}.")
                
                #idx += 1
                
                dist.barrier(device_ids = [self.RANK])

            self._pruned = True
            self.remove_handles()
            return False
            
        dist.barrier(device_ids = [self.RANK]) """
    
        
    """
        elif (iteration - self.prune_iteration) % 391 == 0:

            ticket, fitness = self.search(x, y)

            self.mm.set_ticket(ticket)

            sp = self.mm.sparsity_d.item()
            self.fitnesses.append((sp, fitness))
            self.prunes.append(f"Iteration {iteration}: Pruned to {self.mm.sparsity:.3f} with fitness: {fitness}.")

            dist.barrier(device_ids = [self.RANK])

            if (sp <= self.desired_sparsity):
                self._pruned = True
                
            self.remove_handles()
            return
            
        """
    """
        else:

            ticket, fitness = self.search(x, y)

            sp = self.mm.sparsity_d.item()
            self.fitnesses.append((sp, fitness))
            self.prunes.append(f"Iteration {iteration}: Pruned to {self.mm.sparsity:.3f} with fitness: {fitness}.")

            self.mm.set_ticket(ticket)

            dist.barrier(device_ids = [self.RANK])

            if sp <= self.desired_sparsity:

                self._pruned = True
                self.remove_handles()
        """
     
        

    """
    def post_step_hook(self, x, y, _, step, iteration):
        
        if self._pruned or (step < (388)): return 

        if (step == 388): self.init_capture_hooks()

        elif (step == 389): # 2nd to last, 128 batch size 

            status, ticket, fitness = self.plateau_monitor(x, y)
            
            self.remove_handles()

            self.fitnesses.append(((iteration+1)/391, fitness, self.mm.sparsity.item() * 0.8))

            if status:

                self.mm.set_ticket(ticket)
                self.prunes.append(f"Epoch {(iteration + 1) / 391:.1f}: Pruned to {self.mm.sparsity.item():.3f} with fitness: {fitness}.")

                if self.mm.sparsity_d <= self.desired_sparsity:
                    self.prunes.append(f"Pruning finished on epoch {(iteration + 1)/391:.1f}.")
                    self._pruned = True

        dist.barrier(device_ids = [self.RANK])
    """
    """
    def post_step_hook_single_plateau(self, x, y, _, step, iteration):

        if self._pruned or (step < (390 - 2)): return 

        if (step + 2 == 390): self.init_capture_hooks()

        elif (step + 1 == 390): 

            status, ticket, fitness = self.plateau_monitor(x, y)
            
            if not status: 
                
                self.remove_handles()
                self.fitnesses.append(((iteration+1)/391, fitness, self.mm.sparsity * 0.8))

            else: 
                idx = 1
                self.mm.set_ticket(ticket)
                self.fitnesses.append(((iteration+1)/391, fitness, self.mm.sparsity))
                self.prunes.append(f"Epoch {(iteration + 1) / 391}; {idx}: Pruned to {self.mm.sparsity:.3f} with fitness: {fitness}.")
                self.m.eval()
                
                while self.mm.sparsity_d > self.desired_sparsity:
                    
                    with torch.no_grad():
                        self._act_w.clear()
                        self.m(x)
                    
                    ticket, fitness = self.search(x, y)
                    
                    self.mm.set_ticket(ticket)

                    self.fitnesses.append(((iteration + 1)/391, fitness, self.mm.sparsity))
                    self.prunes.append(f"Epoch {(iteration + 1) / 391}; {idx}: Pruned to {self.mm.sparsity:.3f} with fitness: {fitness}.")

                    idx += 1
                    
                    dist.barrier(device_ids = [self.RANK])

                self._pruned = True
                self.remove_handles()
            
        dist.barrier(device_ids = [self.RANK])
    """
    
    def post_step_hook_single_plateau_iterative(self, data_list, step, iteration, name, *args, **kwargs):
        with torch.no_grad():

            if self._pruned: return True
            
            elif step % 129 == 3:
                x, y = data_list[step//129]
                x, y = x.to('cuda'), y.to('cuda')

                for T in self.cT: x = T(x)
                for T in self.tT: x = T(x)
                for T in self.fcT: x = T(x)
                
                self.init_capture_hooks()
                self.mm(x)

                status, ticket, fitness = self.plateau_monitor(x, y, name)
                self.remove_handles()
                self.fitnesses.append((iteration, fitness, self.mm.sparsity.item() * self.sparsity_rate))
                
                if not status: return True
                
                else:
                    self.prune_model(ticket)
                    self.prunes.append(f"Epoch {(iteration + 1) / 391}: Pruned to {self.mm.sparsity:.3f} with fitness: {self._best_fitness}.")
                    self._best_fitness = float("inf")
                    self._best_ticket = None
                    self._plat_eps = 0
                    if self.mm.sparsity_d.item() <= self.desired_sparsity:
                        self._pruned = True
                    
                    return False

            return True

    def search(self, x, y):
        with torch.no_grad():

            self.m.eval()

            self.GTS = self.DGTS(self.RANK, self.WORLD_SIZE, 
                                self.LOCK, self.SHARED_LIST,
                                self.mm, act_w = self._act_w,
                                max_size = self.search_size,
                                possible_children = self.possible_children,
                                reverse_scores = self.reverse_scores)
            
            activation_mask = None
            if len(self._act_w) != 0: 
                activation_mask = torch.cat(self._act_w)#torch.cat([i * t for i, t in enumerate(self._act_w, start=1)]) # Weight later activations higher, linearly
                self._act_w.clear()
                activation_mask += 1e-10
                activation_mask.div_(activation_mask.sum())

            return self.GTS.search(self.sparsity_rate, max_iterations = self.search_iterations, full_acts = activation_mask, inp = x,
                                   extra_starters = [self._best_ticket] if not (self._best_ticket == None) else [])
        

    def plateau_monitor(self, x: torch.Tensor, y: torch.Tensor, name: str):

        ticket, fitness = self.search(x, y)

        if (fitness <= self._best_fitness) == self.reverse_scores:
            self._plat_eps += 1
            if self._plat_eps >= self.stopping_plateau:
                return True, self._best_ticket, fitness
        
        else: 
            self._best_fitness = fitness
            self._best_ticket = ticket
            self._plat_eps = 0
            self.save_ckpt(name, "rewind")

        return False, ticket, fitness