from models.base import MaskedModel
from models.LotteryLayers import Lottery 

from data.cifar10 import custom_fetch_data, get_loaders

import gc
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from torch.nn import Module

import torch.distributed as dist
import torch.multiprocessing as mp

from torch.utils.data import DataLoader

import numpy as np 
import time
from typing import Callable, List, Tuple
from tqdm import tqdm 

__all__ = ["GradientNormSearch", "NormalizedMSESearch", "KldLogitSearch", "OldKldSearch", "LossSearch", "GradMatchSearch", "DeltaLossSearch"]
 

class BaseGeneticSearch:
    """
    Population Management, Parent Selection, Mutation, Search Loop, Etc.
    """
    def __init__(self, rank: int, world_size: int, model: MaskedModel,
                 input: torch.Tensor | DataLoader, 
                 sparsity_rate: float,
                 transforms: Tuple[Callable] = tuple(),
                 input_sampler_offset: int = None,
                 reverse_fitness_scores: bool = False):
        
        """
        Reverse = False: Lower Fitness is Chosen
        """

        self.RANK = rank
        self.IsRoot = rank == 0
        self.WORLD_SIZE = world_size

        self.mm = model
        self.spr = sparsity_rate

        self.best_log = list()

        self.inp = input
        self.running = isinstance(input, DataLoader)
        self.transforms = transforms

        if (input_sampler_offset is not None) and self.running: self.inp.sampler.set_epoch(input_sampler_offset)

        self.reverse = reverse_fitness_scores 

    def build(self, mp_lock, mp_list,
              search_iterations: int, search_size: int, 
              possible_children: int, 
              mutation_temperature: float,
              final_mutation_temperature: float,
              init_by_mutate = False,
              allow_random = True,
              elite_percentage: float = 0.05,
              fitness_scaling_factor: float = 1.0):
        
        self.lock = mp_lock
        self.population = mp_list

        self.SIZE = search_size
        self.ITERS = search_iterations
        self.MUTATE_TEMP = mutation_temperature
        self.MUTATE_ANNEAL_FACTOR = np.exp(np.log(final_mutation_temperature / mutation_temperature) / search_iterations)
        
        self.FITNESS_SCALING_EXP_FACTOR = fitness_scaling_factor
        self.POSSIBLE_CHILDREN = possible_children
        self.ELITE_PERCENTAGE = elite_percentage

        self.MUTATE_INIT = init_by_mutate
        self.ALLOW_RANDOM = allow_random

    def _prepare_for_fitness_calculation(self):
        pass

    def _cleanup_after_fitness_calculation(self):
        pass

    def _calculate_fitness(self, ticket: torch.Tensor) -> float:
        raise NotImplementedError("Implement Fitness Calculation")

    # --------------------------- DIAGNOSTIC CALCULATION WRAPPERS ------------------------------

    def calculate_fitness_magnitude(self):
        orig = self.mm.export_ticket_cpu()
        self.mm.reset_ticket()
        self.mm.prune_by_mg(self.spr, 1, root = 0)
        out = self.calculate_fitness_given(self.mm.export_ticket_cpu())
        self.mm.set_ticket(orig)
        return out
        

    def calculate_fitness_random(self):
        orig = self.mm.export_ticket_cpu()
        self.mm.reset_ticket()
        self.mm.prune_random(self.spr, distributed = True, root = 0)
        #self.mm.pls()
        out = self.calculate_fitness_given(self.mm.export_ticket_cpu())
        self.mm.set_ticket(orig)
        return out
   
    def calculate_fitness_given(self, ticket: torch.Tensor):
        """
        Calculates fitness for a provided ticket, outside of search.
        """
        orig = self.mm.export_ticket_cpu()
        self.mm.reset_ticket()
        assert hasattr(self, 'inp')
        self._prepare_for_fitness_calculation()
        self.mm.set_ticket(ticket)
        out = self._calculate_fitness(self.mm.export_ticket_cpu())
        self._cleanup_after_fitness_calculation()
        self.mm.set_ticket(orig)
        return out

    # ------------------------- Search Implementation ---------------------------

    def _init_random(self, original_ticket):
        self.mm.set_ticket(original_ticket)
        self.mm.prune_random(self.spr, distributed = False)
        sample = self.mm.export_ticket_cpu()
        self.add_sample(sample, self._calculate_fitness(sample))

    def _init_mutate(self, mg_ticket, mutate_rate):
        sample = self.mm.mutate_ticket(mg_ticket.cuda(), mutate_rate)
        self.add_sample(sample, self._calculate_fitness(sample))

    def search(self, init_given = None):

        self._prepare_for_fitness_calculation()

        original_mode = self.mm.training
        original_ticket = self.mm.export_ticket_cpu()
        self.mm.eval()

        cnt = 0  
        if not self.MUTATE_INIT:
            while cnt < (self.SIZE // (self.WORLD_SIZE)):
                self._init_random(original_ticket)
                cnt += 1
        else: 
            mg_ticket = init_given if init_given is not None else self.mm.prune_by_mg(self.spr, iteration=1).export_ticket_cpu()
            while cnt < int((self.SIZE // (self.WORLD_SIZE)) * 0.5): 
                self._init_mutate(mg_ticket, 0.4)
                cnt += 1
            del mg_ticket
            while cnt < (self.SIZE // (self.WORLD_SIZE)):
                self._init_random(original_ticket)
                cnt += 1
            self.mm.set_ticket(original_ticket)

        dist.barrier(device_ids = [self.RANK])
        if self.IsRoot: bar = tqdm(total = self.ITERS, unit = "iteration", colour="white", leave=False)

        for it in range(1, self.ITERS + 1):
            self.search_step(0.25, self.MUTATE_TEMP)
            if (it % 2) == 0:
                dist.barrier(device_ids = [self.RANK])
                self.clean_population()
                dist.barrier(device_ids=[self.RANK])
                if self.IsRoot and it > 0: 
                    best = self.population[0][1]
                    mean, std = self._get_mean_std_fitness()
                    self.best_log.append((it, best, mean, std))
                    bar.set_postfix({"Best": f"{best:.4f}", "Mean": f"{mean:.4f}", "Std": f"{std:.4f}", "Temp": f"{self.MUTATE_TEMP:.6f}"}) 
                    bar.update(2)

        self._cleanup_after_fitness_calculation()
        dist.barrier(device_ids = [self.RANK])
        output = self.population[0]
        dist.barrier(device_ids = [self.RANK])
        self.mm.set_ticket(original_ticket)
        self.mm.train(original_mode)

        if self.IsRoot: 
            del self.population[:] 
            bar.close()

        dist.barrier(device_ids = [self.RANK])
        return output

    def search_step(self, mutation_rate = 0.15, temperature_rate = 0.1):
        parents = self.distribute_parents()
        num_children = torch.randint(1, self.POSSIBLE_CHILDREN + 1, (1,), device="cpu").item()
        fitness_sum = parents[0][1] + parents[1][1] 
        zero_weight, one_weight = (torch.as_tensor(parents[0][1])/fitness_sum), (torch.as_tensor(parents[1][1])/fitness_sum)
        
        children = [self.mm.merge_tickets(parents[0][0].cuda(), parents[1][0].cuda(), 
                                        zero_weight if self.reverse else one_weight,
                                        one_weight if self.reverse else zero_weight) 
                                        for _ in range(num_children)]

        for child in children:
            if (torch.rand(1,).item() < mutation_rate):
                child = self.mm.mutate_ticket(child, temperature_rate)
            self.add_sample(child, self._calculate_fitness(child))

        self.MUTATE_TEMP *= self.MUTATE_ANNEAL_FACTOR    

    # -------------------------------------- Population Management -----------------------------

    def clean_population(self):

        if self.RANK != 0: return
        desired_size = int(self.SIZE)
        elite_size = int(self.SIZE * self.ELITE_PERCENTAGE)
        remaining_size = desired_size - elite_size

        with self.lock:
            if len(self.population) <= desired_size: return
            
            rest_candidates = list(range(elite_size, len(self.population)))
            sampled_indices = np.random.choice(rest_candidates, size=remaining_size, replace=False)
            
            kept_indices = set(range(elite_size)).union(set(sampled_indices))
            
            self.population[:] = [self.population[i] for i in sorted(list(kept_indices))]

            gc.collect()

    
    def _get_mean_std_fitness(self, root=0):
        if self.RANK != root: return
        fitnesses = np.asarray([fitness for mask, fitness in self.population])
        return fitnesses.mean(), fitnesses.std()

    def add_sample(self, mask: torch.Tensor, fitness: float):
        """
        Adds a new sample (ticket, fitness) to the population, maintaining sorted order.
        """
        mask = mask.cpu()
        mask.share_memory_()

        with self.lock:
            left, right = 0, len(self.population) - 1
            insert_pos = len(self.population)
            while left <= right:
                mid = (left + right) // 2
                if (self.population[mid][1] < fitness) != self.reverse:
                    left = mid + 1
                else:
                    insert_pos = mid
                    right = mid - 1
            self.population.insert(insert_pos, (mask, fitness,))

    # -------------------------------- PARENT SELECTION ---------------------------
    
    def distribute_parents(self):
        """
        Selects two parents from the population using a scaled, rank-based selection.
        """
        size = len(self.population) // self.WORLD_SIZE
        fitnesses = np.array([fitness for _, fitness in self.population], dtype=np.float64)

        min_fitness, max_fitness = fitnesses.min(), fitnesses.max()
        if max_fitness == min_fitness: # Avoid division by zero
             fitnesses = np.ones_like(fitnesses)
        else:
            fitnesses = (fitnesses - min_fitness) / (max_fitness - min_fitness)

        fitnesses = 1.0 - fitnesses  # Assumes lower score is better

        # Rank-based selection probabilities
        probs = np.arange(len(fitnesses), 0, step=-1, dtype=np.float32)
        probs /= probs.sum()
        
        # Select 2 distinct parents based on rank probabilities
        parent_indices = np.random.choice(len(self.population), size=2, replace=False, p=probs)
        
        return (self.population[parent_indices[0]], self.population[parent_indices[1]])


class KldLogitSearch(BaseGeneticSearch):

    def __init__(self, rank: int, world_size: int, model: MaskedModel,
                 input: torch.Tensor | DataLoader, sparsity_rate: float,
                 transforms: Tuple[Callable] = tuple(),
                 input_sampler_offset: int = None,
                 reverse_fitness_scores: bool = False,
                 reverse: bool = True):
        
        super().__init__(rank, world_size, model, input, sparsity_rate, transforms, 
                         input_sampler_offset, reverse_fitness_scores)
        
        self.full_activations = None
        self.reversekld = reverse

    def _prepare_for_fitness_calculation(self):
        self.make_full_acts()

    @torch.no_grad()
    def make_full_acts(self):

        if self.full_activations is not None: return

        if not self.running:
            logits = self.mm(self.inp)
            self.full_activations = logits.log_softmax(1)
            return
        
        self.full_activations = list()

        for x, *_ in self.inp:
            x = x.cuda()
            for T in self.transforms: x = T(x)
            
            logits = self.mm(x)
            activation_mask = (logits).log_softmax(1)

            self.full_activations.append(activation_mask)

    def _calculate_fitness(self, ticket):

        self.mm.eval()
        self.mm.set_ticket(ticket)

        with torch.no_grad():

            if not self.running:
        
                torch.cuda.empty_cache()

                logits = self.mm(self.inp)
                curr_activations = logits.log_softmax(1)

                if not self.reversekld: return F.kl_div(curr_activations, self.full_activations, reduction = "batchmean", log_target = True).item()
                else: return F.kl_div(self.full_activations, curr_activations, reduction = "batchmean", log_target = True).item()

            kl_tr = torch.as_tensor(0.0, dtype = torch.float64, device = 'cuda')
            cnt = torch.as_tensor(0, dtype = torch.int64, device = 'cuda')

            for idx, (x, *_) in enumerate(self.inp):
                
                torch.cuda.empty_cache()

                x = x.cuda()
                for T in self.transforms: x = T(x)

                logits = self.mm(x)
                curr_activations = logits.log_softmax(1)

                if not self.reversekld: kl_tr += F.kl_div(curr_activations, self.full_activations[idx], reduction = "batchmean", log_target = True)
                else: kl_tr += F.kl_div(self.full_activations[idx], curr_activations, reduction = "batchmean", log_target = True)
                cnt += 1
            
            return kl_tr.div_(cnt.float()).item()

class GradMatchSearch(BaseGeneticSearch):

    def __init__(self, rank: int, world_size: int, model: MaskedModel,
                 input: torch.Tensor | DataLoader, sparsity_rate: float,
                 transforms: Tuple[Callable] = tuple(),
                 input_sampler_offset: int = None,
                 reverse_fitness_scores: bool = False,):
        
        super().__init__(rank, world_size, model, input, sparsity_rate, transforms, 
                         input_sampler_offset, reverse_fitness_scores)
        
        self.full_grads = None

    def _prepare_for_fitness_calculation(self):
        self.make_full_grads()

    def make_full_grads(self):

        if self.full_grads is not None: return

        self.full_grads = list()

        for x, y, *_ in self.inp:
            x, y = x.cuda(), y.cuda()
            for T in self.transforms: x = T(x)
            loss = F.cross_entropy(self.mm(x), y)
            grad_mask =  [grad.detach() for grad in torch.autograd.grad(loss, [layer.weight for layer in self.mm.lottery_layers])]
            grad_mask = [grad.sub(grad.mean()).div(grad.std() + 1e-12).view(-1) for grad in grad_mask]

            self.full_grads.append(grad_mask)

    def _calculate_fitness(self, ticket):

        self.mm.eval()
        self.mm.set_ticket(ticket)

        kl_tr = torch.as_tensor(0.0, dtype = torch.float64, device = 'cuda')
        cnt = torch.as_tensor(0, dtype = torch.int64, device = 'cuda')

        for idx, (x, y, *_) in enumerate(self.inp):
            
            torch.cuda.empty_cache()
            
            x, y = x.cuda(), y.cuda()
            for T in self.transforms: x = T(x)

            loss = F.cross_entropy(self.mm(x), y)
            grad_mask = [grad.detach() for grad in torch.autograd.grad(loss, [layer.weight for layer in self.mm.lottery_layers])]
            
            with torch.no_grad():
            
                grad_mask = [grad.sub(grad.mean()).div(grad.std() + 1e-12).view(-1) for grad in grad_mask]

                step_loss =  torch.as_tensor(0.0, dtype = torch.float32, device = 'cuda')
                layer_cnt = 0
                
                for sparse, dense in zip(grad_mask, self.full_grads[idx]):
                    step_loss += F.mse_loss(sparse, dense, reduction = 'mean')
                    layer_cnt += 1
                
                kl_tr += step_loss / layer_cnt
                cnt += 1
            
        return kl_tr.div_(cnt.float()).item()


class LossSearch(BaseGeneticSearch):
        
    def _calculate_fitness(self, ticket):
        
        self.mm.eval()
        self.mm.set_ticket(ticket)

        loss_tr = torch.as_tensor(0.0, dtype=torch.float64, device='cuda')
        cnt = torch.as_tensor(0, dtype=torch.int64, device='cuda')

        for x, y, *_ in self.inp:
            torch.cuda.empty_cache()

            x, y = x.cuda(), y.cuda()
            for T in self.transforms: x = T(x)

            loss = F.cross_entropy(self.mm(x), y)
            
            loss_tr += loss
            cnt += 1
        
        if cnt == 0: return float('inf') # Return a high value if no batches were processed
        return loss_tr.div(cnt.float()).item()
    
class DeltaLossSearch(BaseGeneticSearch):
        
    @torch.no_grad()
    def _calculate_fitness(self, ticket):
        
        self.mm.eval()
        #self.mm.set_ticket(ticket)

        loss_tr = torch.as_tensor(0.0, dtype=torch.float64, device='cuda')
        cnt = torch.as_tensor(0, dtype=torch.int64, device='cuda')

        for x, y, *_ in self.inp:
            torch.cuda.empty_cache()

            x, y = x.cuda(), y.cuda()
            for T in self.transforms: x = T(x)

            self.mm.reset_ticket()

            lossD = F.cross_entropy(self.mm(x), y)

            self.mm.set_ticket(ticket)

            loss = F.cross_entropy(self.mm(x), y)
            
            loss_tr += (loss/lossD - 1).abs()
            cnt += 1
        
        if cnt == 0: return float('inf') # Return a high value if no batches were processed
        return loss_tr.div(cnt.float()).item()

class GradientNormSearch(BaseGeneticSearch):

    def __init__(self, rank: int, world_size: int, model: MaskedModel,
                 input: torch.Tensor | DataLoader, sparsity_rate: float,
                 transforms: Tuple[Callable] = tuple(),
                 input_sampler_offset: int = None,
                 reverse_fitness_scores: bool = False,):
        
        super().__init__(rank, world_size, model, input, sparsity_rate, transforms, 
                         input_sampler_offset, reverse_fitness_scores)
        
        self.reverse = not reverse_fitness_scores # since higher gradient norm is better

    def _calculate_fitness(self, ticket):
        
        self.mm.eval()
        self.mm.set_ticket(ticket)

        norm_tr = torch.as_tensor(0.0, dtype=torch.float64, device='cuda')
        cnt = torch.as_tensor(0, dtype=torch.int64, device='cuda')

        for x, y, *_ in self.inp:
            torch.cuda.empty_cache()

            x, y = x.cuda(), y.cuda()
            for T in self.transforms: x = T(x)

            loss = F.cross_entropy(self.mm(x), y)
            grad_w = torch.autograd.grad(loss, self.mm.parameters(), allow_unused=True)
            
            grad_max_vals = [g.detach().abs().amax() for g in grad_w if g is not None]
            if not grad_max_vals: continue # Skip if no gradients were computed

            max_val = torch.stack(grad_max_vals).amax().clamp(min=1e-30)
            grad_w = [torch.nan_to_num(grad.to(torch.float64) / max_val, nan=0.0) for grad in grad_w if grad is not None]
            
            if not grad_w: continue

            norms = list(torch._foreach_norm(grad_w, 2.0))
            total_norm = torch.norm(torch.stack(norms), 2.0) * max_val
            norm_tr += total_norm
            cnt += 1
        
        if cnt == 0: return float('-inf') # Return a high value if no batches were processed
        return norm_tr.div(cnt.float()).item()

class NormalizedMSESearch(BaseGeneticSearch):

    def __init__(self, rank: int, world_size: int, model: MaskedModel,
                 input: torch.Tensor | DataLoader, 
                 sparsity_rate: float,
                 transforms: Tuple[Callable] = tuple(),
                 capture_layers: List[Module] = [], 
                 fake_capture_layers: List[Tuple[Module, Callable]] = [],
                 input_sampler_offset: int = None,
                 reverse_fitness_scores: bool = False,):
        
        super().__init__(rank, world_size, model, input, sparsity_rate, transforms,
                         input_sampler_offset, reverse_fitness_scores)
        
        self._captures = capture_layers
        self._fcaptures = fake_capture_layers
        self._handles = list()
        self.act_w = list()
        self.full_activations = None

    def _prepare_for_fitness_calculation(self):
        self.init_hooks()
        self.make_full_acts()
    
    def _cleanup_after_fitness_calculation(self):
        self.remove_handles()


    # ------------------------- ACTIVATION HOOK MANAGEMENT ---------------------------


    @torch.no_grad()
    def make_full_acts(self):

        if self.full_activations is not None: return

        if not self.running:
            self.mm(self.inp)
            activation_mask = [act for act in self.act_w]
            self.clear_capture()
            self.full_activations = activation_mask
            return
        
        self.full_activations = list()

        for x, *_ in self.inp:
            x = x.cuda()
            for T in self.transforms: x = T(x)
            self.mm(x)

            activation_mask = [act for act in self.act_w]
            self.clear_capture()
            #activation_mask.div_(activation_mask.sum(dim = 1, keepdim = False))
            #activation_mask = (activation_mask).log_softmax(0)

            self.full_activations.append(activation_mask)

    def init_hooks(self):
        for layer in self._captures:
            self._handles.append(layer.register_forward_hook(self._hook))
        for layer, func in self._fcaptures:
            self._handles.append(layer.register_forward_hook(lambda *args, **kwargs: self._fhook(func, *args, **kwargs)))
        return 
    
    def remove_handles(self):
        for handle in self._handles: handle.remove()
        self._handles.clear()

    def _hook(self, module, input: torch.Tensor, output: torch.Tensor):
        self.act_w.append(output.detach().to(torch.float64).view(output.shape[0], -1))

    def _fhook(self, func, module, input: torch.Tensor, output: torch.Tensor):
        return self.act_w.append(func(output.detach().to(torch.float64)).view(output.shape[0], -1))

    def clear_capture(self):
        self.act_w.clear()

    # ----------------------------------------------- FITNESS CALCULATION  -----------------------------------------
        
    def _calculate_fitness(self, ticket: torch.Tensor):

        self.mm.eval()
        self.mm.set_ticket(ticket)

        with torch.no_grad():

            if not self.running:
        
                torch.cuda.empty_cache()

                self.mm(self.inp)

                loss = torch.as_tensor(0.0, device = 'cuda', dtype = torch.float32)

                for act_idx, act in enumerate(self.act_w):
                    std, mean = torch.std_mean(self.full_activations[act_idx], dim = 1 , keepdim = True)
                    loss += F.mse_loss(act.sub(act.mean(dim = 1, keepdim = True)).div(act.std(dim = 1, keepdim = True) + 1e-12), 
                               self.full_activations[act_idx].sub(mean).div(std + 1e-12), reduction = "mean")
                     
                self.clear_capture()
                
                return loss
            
            kl_tr = torch.as_tensor(0.0, dtype = torch.float64, device = 'cuda')
            cnt = torch.as_tensor(0, dtype = torch.int64, device = 'cuda')

            for idx, (x, *_) in enumerate(self.inp):
                
                torch.cuda.empty_cache()

                x = x.cuda()
                for T in self.transforms: x = T(x)

                self.mm(x)
                loss = torch.as_tensor(0.0, device = 'cuda', dtype = torch.float32)
                for act_idx, act in enumerate(self.act_w):
                    std, mean = torch.std_mean(self.full_activations[idx][act_idx], dim = 1 , keepdim = True)
                    loss += F.mse_loss(act.sub(act.mean(dim = 1, keepdim = True)).div(act.std(dim = 1, keepdim = True) + 1e-12), 
                               self.full_activations[idx][act_idx].sub(mean).div(std + 1e-12), reduction = "mean")

                self.clear_capture()

                kl_tr += loss
                cnt += 1
            
            return kl_tr.div_(cnt.float()).item()
        
class OldKldSearch(BaseGeneticSearch):

    def __init__(self, rank: int, world_size: int, model: MaskedModel,
                 input: torch.Tensor | DataLoader, 
                 sparsity_rate: float,
                 transforms: Tuple[Callable] = tuple(),
                 capture_layers: List[Module] = [], 
                 fake_capture_layers: List[Tuple[Module, Callable]] = [],
                 input_sampler_offset: int = None,
                 reverse_fitness_scores: bool = False,):
        
        super().__init__(rank, world_size, model, input, sparsity_rate, transforms,
                         input_sampler_offset, reverse_fitness_scores)
        
        self._captures = capture_layers
        self._fcaptures = fake_capture_layers
        self._handles = list()
        self.act_w = list()
        self.full_activations = None

    def _prepare_for_fitness_calculation(self):
        self.init_hooks()
        self.make_full_acts()
    
    def _cleanup_after_fitness_calculation(self):
        self.remove_handles()


    # ------------------------- ACTIVATION HOOK MANAGEMENT ---------------------------


    @torch.no_grad()
    def make_full_acts(self):

        if self.full_activations is not None: return

        if not self.running:
            self.mm(self.inp)
            activation_mask = [act for act in self.act_w]
            self.clear_capture()
            self.full_activations = activation_mask.cpu()
            return
        
        self.full_activations = list()

        for x, *_ in self.inp:
            x = x.cuda()
            for T in self.transforms: x = T(x)
            self.mm(x)

            activation_mask = torch.cat([act.log_softmax(1) for act in self.act_w], dim = 1)
            self.clear_capture()
            #activation_mask = activation_mask.div(activation_mask.sum(dim = 1, keepdim = True))
            #activation_mask = activation_mask.log_softmax(1)

            self.full_activations.append(activation_mask.cpu())

    def init_hooks(self):
        for layer in self._captures:
            self._handles.append(layer.register_forward_hook(self._hook))
        for layer, func in self._fcaptures:
            self._handles.append(layer.register_forward_hook(lambda *args, **kwargs: self._fhook(func, *args, **kwargs)))
        return 
    
    def remove_handles(self):
        for handle in self._handles: handle.remove()
        self._handles.clear()

    def _hook(self, module, input: torch.Tensor, output: torch.Tensor):
        self.act_w.append(output.detach().to(torch.float64).view(output.shape[0], -1) )#+ 1e-8)

    def _fhook(self, func, module, input: torch.Tensor, output: torch.Tensor):
        return self.act_w.append(func(output.detach().to(torch.float64)).view(output.shape[0], -1) )#+ 1e-8)

    def clear_capture(self):
        self.act_w.clear()

    # ----------------------------------------------- FITNESS CALCULATION  -----------------------------------------
        
    def _calculate_fitness(self, ticket: torch.Tensor):

        self.mm.eval()
        self.mm.set_ticket(ticket)

        with torch.no_grad():

            if not self.running:
        
                torch.cuda.empty_cache()

                self.mm(self.inp)

                curr_acts = torch.cat([act.log_softmax(1) for act in self.act_w], dim = 1)
                self.clear_capture()
                #curr_acts = curr_acts.div(curr_acts.sum(dim = 1, keepdim = True))
                #curr_acts = curr_acts.log_softmax(1)

                loss = F.kl_div(curr_acts, self.full_activations.cuda(), log_target = True, reduction = "batchmean")

                return loss
            
            kl_tr = torch.as_tensor(0.0, dtype = torch.float64, device = 'cuda')
            cnt = torch.as_tensor(0, dtype = torch.int64, device = 'cuda')

            for idx, (x, *_) in enumerate(self.inp):
                
                torch.cuda.empty_cache()

                x = x.cuda()
                for T in self.transforms: x = T(x)

                self.mm(x)

                curr_acts = torch.cat([act.log_softmax(1) for act in self.act_w], dim = 1)
                self.clear_capture()
                #curr_acts = curr_acts.div(curr_acts.sum(dim = 1, keepdim = True))
                #curr_acts = curr_acts.log_softmax(1)

                loss = F.kl_div(curr_acts, self.full_activations[idx].cuda(), reduction = "batchmean", log_target = True)

                kl_tr += loss
                cnt += 1
            
            return kl_tr.div_(cnt.float()).item()

