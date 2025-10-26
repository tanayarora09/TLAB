import gc
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch.optim.adam
from torch.utils.data import DataLoader
from typing import Callable, List, Tuple
from collections import defaultdict

import copy

import time
from tqdm import tqdm 

from models.base import BaseModel
from data.cifar10 import get_loaders, custom_fetch_data


__all__ = ["SNIPConcrete", "GraSPConcrete", "NormalizedMseFeatures", "KldLogit", "OldKld", "StepAlignmentConcrete", "LossChangeConcrete"]


class FrozenConcrete:
    def __init__(self, rank: int, world_size: int, model: BaseModel | DDP,
                 capture_layers: List[Module] = None,
                 fake_capture_layers: List[Tuple[Module, Callable]] = None,
                 concrete_temperature = 2./3.):
        
        self.RANK = rank
        self.IsRoot = rank == 0
        self.WORLD_SIZE = world_size
        self.DISTRIBUTED = world_size > 1
        self.m = model
        self.mm = model.module if isinstance(model, DDP) else model
        self.original_training_mode = self.m.training
        self.leafed_state_dict = None
        self._captures = capture_layers or []
        self._fcaptures = fake_capture_layers or []
        self._handles = []
        self.act_w = []

        self.concrete_temperature = concrete_temperature
        self._current_temperature = self.concrete_temperature

    def metric_results(self) -> dict[str, float]:
        """
        Return Loss and Accuracy. 
        Should be called from root process.
        """
        with torch.no_grad():
            return {"loss": (self.eloss.div(self.ecount).detach().item()),
                    "sparsity": (self.mm.sparsity.detach().item()), 
                    "true_sparsity": (100 * self.mm.get_true_active().item() / self.mm.num_prunable),}
                    
    
    def reset_metrics(self):
        with torch.no_grad():
            self.eloss.fill_(0.0)
            self.ecount.fill_(0)
            self.reset_running_metrics()

    def reset_running_metrics(self):
        """
        Reset Loss, Accuracy, and Sample Count.
        Should be called from all processes.
        """
        with torch.no_grad():
            self.loss_tr.fill_(0.0)
            self.count_tr.fill_(0)
    
    def transfer_metrics(self):
        """
        Move from running metrics.
        This
        """
        with torch.no_grad():
            self._collect_metrics()
            self.eloss += self.loss_tr
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
            dist.all_reduce(self.count_tr, op = dist.ReduceOp.SUM)

    def _set_concrete_temperature(self, temp: float  = None):
        if temp is None: temp = self.concrete_temperature
        self._current_temperature = temp
        self.mm.set_concrete_temperature(temp)

    def build(self, desired_sparsity: float, optimizer, 
              optimizer_kwargs: dict, 
              transforms: Tuple[Callable],
              gradbalance = False):

        """
        desired_sparsity = 0.2 --> 80% Sparse, 20% Dense
        """

        self.gradbalance = gradbalance

        self.loss_tr = torch.as_tensor(0.0, dtype = torch.float64, device = 'cuda')
        self.count_tr = torch.as_tensor(0, dtype = torch.int64, device = 'cuda')

        self.eloss = torch.as_tensor(0.0, dtype = torch.float64, device = 'cuda')
        self.ecount = torch.as_tensor(0, dtype = torch.int64, device = 'cuda')

        self.spr = desired_sparsity
        self._desired_active = self.spr * self.mm.num_prunable
        if gradbalance: self._desired_active *= 1.1
        self._inv_desired_active = 1. / self._desired_active
        self._sparsity_scaler_constant = 100. #100. / self.mm.num_prunable

        self._loss_scaler_constant = 1.

        self.transforms = transforms

        self.leafed_state_dict = self.m.state_dict()
        
        self.m.eval() 
        for param in self.mm.parameters():
            param.grad = None
            param.requires_grad_(False)
        
        self.target_logit = torch.log(torch.as_tensor(desired_sparsity/(1 - desired_sparsity), device = 'cuda')) # / self.concrete_temperature 
        self.mm.prepare_for_continuous_optimization(initial_alpha = self.target_logit)
        self._set_concrete_temperature()

        self.lagrange_multiplier = torch.as_tensor(0.0, device = 'cuda', dtype = torch.float32).requires_grad_(not gradbalance)
        self._output_target_lambda = None

        lambda_lr = 1e-3

        if "lr" in optimizer_kwargs: lambda_lr = 1e-2 * optimizer_kwargs["lr"]

        self.optim_lagrangian = None
        if not gradbalance: self.optim_lagrangian = torch.optim.SGD((self.lagrange_multiplier, ), lr = lambda_lr, maximize = True)
        if gradbalance: 
            self.lagrangian_smoothing = 1e-2
            self.lagrange_multiplier.fill_(float("-inf"))

        self.optim = optimizer((self.mm.get_buffer("MASK"),), **optimizer_kwargs)


    def finish(self):
        
        self.mm.revert_to_binary_mask(torch.ones(self.mm.num_prunable, dtype = torch.bool))

        self.m.train(self.original_training_mode)
        if self.leafed_state_dict:
            self.m.load_state_dict(self.leafed_state_dict)
        for param in self.mm.parameters():
            param.requires_grad_(True)
        self.remove_handles()

    def init_hooks(self):
        for layer in self._captures:
            self._handles.append(layer.register_forward_hook(self._hook))
        for layer, func in self._fcaptures:
            self._handles.append(layer.register_forward_hook(
                lambda *args, func=func, **kwargs: self._fhook(func, *args, **kwargs)
            ))

    def remove_handles(self):
        for handle in self._handles: handle.remove()
        self._handles.clear()

    def clear_capture(self):
        self.act_w.clear()

    def _hook(self, *args): raise NotImplementedError
    def _fhook(self, *args): raise NotImplementedError

    def _compute_loss(self, x, y) -> torch.Tensor:
        raise NotImplementedError

    def _scale_learning_rate(self, factor):
        with torch.no_grad():    
            for pg in self.optim.param_groups:
                pg['lr'] *= factor
            if not self.gradbalance: 
                for pg in self.optim_lagrangian.param_groups:
                    pg['lr'] *= factor

    def optimize_step(self, x, y):

        loss = self._compute_loss(x, y) * self._loss_scaler_constant
        sigmoid_mask = torch.sigmoid(self.mm.get_buffer("MASK"))
        sparsity_error = (sigmoid_mask.sum() * self._inv_desired_active - 1) * self._sparsity_scaler_constant 


        if not self.gradbalance:

            lagrangian_loss = loss + self.lagrange_multiplier * sparsity_error
        
            lagrangian_loss.backward()

            self.mm.zero_grad() 

            with torch.no_grad():
                dist.all_reduce(self.lagrange_multiplier.grad, op = dist.ReduceOp.AVG)

            self.optim_lagrangian.step()
            self.optim_lagrangian.zero_grad()

        else: 

            task_grad = torch.autograd.grad(loss, self.mm.get_buffer("MASK"), retain_graph = False)[0]
            sparsity_grad = (
                self._inv_desired_active
                * self._sparsity_scaler_constant
                * sigmoid_mask
                * (1 - sigmoid_mask)
            )

            with torch.no_grad():
                
                if sparsity_error < 0: target_lambda.fill_(0.0)

                else:
                    task_grad_max = task_grad.abs().amax().clamp_(min=1e-12)
                    task_grad_norm = task_grad.div(task_grad_max).norm(2)

                    sparsity_grad_max = sparsity_grad.abs().amax().clamp_(min=1e-12)
                    sparsity_grad_norm = sparsity_grad.div(sparsity_grad_max).norm(2)

                    target_lambda = (task_grad_norm * task_grad_max) / (sparsity_grad_norm * sparsity_grad_max + 1e-12)
                
                    #target_lambda = torch.sign(sparsity_error) * target_lambda

                dist.all_reduce(target_lambda, op = dist.ReduceOp.AVG)

                if self.lagrange_multiplier != float("-inf"): 
                    self.lagrange_multiplier = self.lagrange_multiplier * (1 - self.lagrangian_smoothing) + target_lambda * self.lagrangian_smoothing

                else: 
                    self.lagrange_multiplier = target_lambda

            self.mm.get_buffer("MASK").grad = task_grad + self.lagrange_multiplier * sparsity_grad            

        with torch.no_grad():
            dist.all_reduce(self.mm.get_buffer("MASK").grad, op = dist.ReduceOp.AVG)

        self.optim.step()
        self.optim.zero_grad()

        with torch.no_grad():

            self.loss_tr += loss
            self.count_tr += y.size(dim = 0)
        

        # DEBUG 
        """
        print(f"[rank {self.RANK}] SUM : {self.mm.get_buffer('MASK').mean()} | STD : {self.mm.get_buffer('MASK').std()} | NORM : {self.mm.get_buffer('MASK').norm(2)}")
        """
        

    def optimize_mask(self, train_data: DataLoader, epochs: int, train_cardinality: int, 
                      sampler_offset: int = 0, dynamic_epochs = False, reduce_epochs: List[int] = [], 
                      invert_mask = False, custom_continuous_to_mg: bool = False, anneal_temperature = False,):

        """
        If Dynamic Epochs is True, Epochs argument will be ignored
        """

        logs = defaultdict(dict)
        self.reset_metrics()

        if self.IsRoot: 
            bar = tqdm(total = None if dynamic_epochs else epochs * train_cardinality, unit = "it", colour="white", leave = True)
            last_iter = 0

        break_epoch = False
        epoch = 0

        while not break_epoch:

            self.reset_metrics()
            train_data.sampler.set_epoch(epoch + train_cardinality * sampler_offset)

            if not dynamic_epochs and anneal_temperature:
                self._set_concrete_temperature( self.concrete_temperature * (0.1 / self.concrete_temperature) ** (epoch / (epoch - 1)) )

            for step, (x, y, *_) in enumerate(train_data):
                
                iteration = int(epoch * train_cardinality + step + 1)

                x, y = x.cuda(), y.cuda()
                for T in self.transforms: x = T(x)
                
                self.optimize_step(x, y)

                if (step + 1) % (train_cardinality//30) == 0 or (step + 1 == train_cardinality):

                    self.transfer_metrics()
                    
                    if self.IsRoot:

                        logs[iteration] = self.metric_results()
                        
                        bar.set_postfix_str(f"L: {logs[iteration]['loss']:.4e} | Pred {logs[iteration]['sparsity']:.3f}% | Real {logs[iteration]['true_sparsity']:.3f}% | STD : {self.mm.get_buffer("MASK").std().item():.3f} | Lagrangian {self.lagrange_multiplier.item():.3e} | Tau {self._current_temperature:.2f}",
                                            refresh = False)
                        
                        bar.update(iteration - last_iter)
                        last_iter = iteration

            epoch += 1

            if not dynamic_epochs:
                if epoch >= epochs: break_epoch = True

            else:
                if (self.mm.get_true_active()/self.mm.num_prunable) >= self.spr: break_epoch = True

            if (epoch + 1) in reduce_epochs: self._scale_learning_rate(0.1)


        if self.IsRoot: 
            bar.close()
            print(f"\nExpected Average Sparsity: {self.mm.sparsity.item()}")
            print(f"True On-Gate Sparsity: {100 * (self.mm.get_true_active()/self.mm.num_prunable).item()}")
            print(f"Clipped Sparsity: {(100 * self.spr)}")

        if not custom_continuous_to_mg: ticket = self.mm.get_continuous_ticket(sparsity_d = None if dynamic_epochs else self.spr, invert = invert_mask)

        else: ticket = self.mm.custom_continuous_to_mg(sparsity_d = self.spr)

        return logs, ticket
    

class GraSPConcrete(FrozenConcrete):

    def build(self, desired_sparsity: float, optimizer, 
              optimizer_kwargs: dict, 
              transforms: Tuple[Callable],
              gradbalance = False):
        
        #optimizer_kwargs["maximize"] = True
        super().build(desired_sparsity, optimizer, optimizer_kwargs, transforms, gradbalance)

        #self._sparsity_scaler_constant *= 100
        #self._loss_scaler_constant *= 1e-2 # so loss ~ 5e-3

        #if self.IsRoot: print(self.optim)
        
        for param in self.mm.parameters(): param.requires_grad_(True)        
        self.mm.zero_grad()

    def _compute_loss(self, x, y):

        self.mm.zero_grad()
        loss = F.cross_entropy(self.mm(x), y)
        
        grad_w = torch.autograd.grad(loss, self.mm.parameters(), create_graph = True)
        grad_max_vals = [g.detach().abs().amax() for g in grad_w if g is not None]

        max_val = torch.stack(grad_max_vals).amax().clamp(min=1e-30).div(self.mm.num_learnable) # APPLY MEAN OPERATION HERE
        grad_w = [torch.nan_to_num(grad / max_val, nan=0.0) for grad in grad_w if grad is not None]

        norms = list(torch._foreach_norm(grad_w, 2.0))
        #print(f"Norms: {[torch.isnan(norm).count_nonzero().item() for norm in norms]}")
        total_norm = torch.norm(torch.stack(norms), 2.0) * max_val
        #print(f"Total norm: {total_norm.item()}")
        return -total_norm # maximize
    
class SNIPConcrete(FrozenConcrete):
    
    def build(self, desired_sparsity: float, optimizer, 
              optimizer_kwargs: dict, 
              transforms: Tuple[Callable],
              gradbalance = False):
        
        super().build(desired_sparsity, optimizer, optimizer_kwargs, transforms, gradbalance)
        #self._sparsity_scaler_constant *= 100
        self._loss_scaler_constant *= 100

    def _compute_loss(self, x, y):
        loss = F.cross_entropy(self.mm(x), y)
        return loss.abs()

class LossChangeConcrete(FrozenConcrete):
    
    def build(self, desired_sparsity: float, optimizer, 
              optimizer_kwargs: dict, 
              transforms: Tuple[Callable],
              gradbalance = False):
        
        super().build(desired_sparsity, optimizer, optimizer_kwargs, transforms, gradbalance)
        #self._sparsity_scaler_constant *= 100
        self._loss_scaler_constant *= 10000

    def _compute_loss(self, x, y):
        
        with torch.no_grad():
            self.mm.deactivate_mask()
            lossD = F.cross_entropy(self.mm(x), y)
            self.mm.activate_mask()

        loss = F.cross_entropy(self.mm(x), y)
        return (loss/lossD - 1).abs()

class ActivationConcrete(FrozenConcrete):

    def __init__(self, rank: int, world_size: int, 
                 model: DDP,
                 capture_layers: List[Module] = [],
                 fake_capture_layers: List[Tuple[Module, Callable]] = [],
                 concrete_temperature = 2./3.):
        super().__init__(rank, world_size, model, capture_layers, fake_capture_layers, concrete_temperature = concrete_temperature)
        self.full_activations = []
        """self.dense_mm = copy.deepcopy(self.mm)
        self.dense_mm.reset_ticket()
        self.dense_mm.eval()
        for param in self.dense_mm.parameters(): 
            param.grad = None
            param.requires_grad_(False)"""

    def build(self, desired_sparsity: float, 
              optimizer, optimizer_kwargs: dict, 
              transforms: Tuple[Callable],
              gradbalance = False):
        
        super().build(desired_sparsity, optimizer, optimizer_kwargs, transforms, gradbalance)

        #self._add_dense_to_capture_layers()
        self.init_hooks()

    def _add_dense_to_capture_layers(self):

        name_map = {module: name for name, module in self.mm.named_modules()}
        
        target_captures = [name_map[submodule] for submodule in self._captures]
        self._captures.extend((self.dense_mm.get_submodule(name) for name in target_captures))

        target_fcaptures = [(name_map[submodule], fn) for submodule, fn in self._fcaptures]
        self._fcaptures.extend(((self.dense_mm.get_submodule(name), fn) for name, fn in target_fcaptures))


class KldLogit(ActivationConcrete):

    def __init__(self, *args, reverse = True, **kwargs):
        super().__init__(*args, **kwargs)
        self.reverse_kld = reverse
        print(self.reverse_kld)

    def build(self, desired_sparsity: float, optimizer, 
              optimizer_kwargs: dict, 
              transforms: Tuple[Callable],
              gradbalance = False):
        
        super().build(desired_sparsity, optimizer, optimizer_kwargs, transforms, gradbalance)
        self._loss_scaler_constant *= 100

    def _compute_loss(self, x, y):
        
        with torch.no_grad():
            self.mm.deactivate_mask() 
            dense = self.mm(x).detach()
            #dense.div_(dense.sum(dim = 1, keepdim = True))
            dense = dense.log_softmax(1)
            self.mm.activate_mask()


        sparse = self.mm(x)
        #sparse.div_(sparse.sum(dim = 1, keepdim = True))
        sparse = sparse.log_softmax(1)

        if not self.reverse_kld: out = F.kl_div(sparse, dense, reduction = 'batchmean', log_target = True)
        else: out = F.kl_div(dense, sparse, reduction = 'batchmean', log_target = True)

        return out

    def _hook(self, _, __, output): return #self.act_w.append(output.view(output.shape[0], -1))
    def _fhook(self, func, _, __, output): return #self.act_w.append(func(output).view(output.shape[0], -1))

class NormalizedMseFeatures(ActivationConcrete):
    
    def build(self, desired_sparsity: float, optimizer, 
              optimizer_kwargs: dict, 
              transforms: Tuple[Callable],
              gradbalance = False):
        
        super().build(desired_sparsity, optimizer, optimizer_kwargs, transforms, gradbalance)
        self._loss_scaler_constant *= 100
        if not gradbalance: self.optim_lagrangian.param_groups[0]["lr"] /= 10
    
    def _normalize(self, act: torch.Tensor):
        
        #if (act.ndim == 4): 
        #    mean = act.mean(dim = [2, 3], keepdim = True)
        #    std = act.std(dim = [2, 3], keepdim = True)
        #
        #elif (act.ndim == 2):
        mean = act.mean(dim = 1, keepdim = True)
        std = act.std(dim = 1, keepdim = True)

        return act.sub(mean).div(std + 1e-9)

    def _compute_loss(self, x, y):
        
        self.clear_capture()

        with torch.no_grad(): 
            self.mm.deactivate_mask()
            self.mm(x)
            dense_acts =  [act for act in self.act_w]
            self.clear_capture()
            self.mm.activate_mask()
        
        self.mm(x)
        
        loss = torch.as_tensor(0.0, dtype = torch.float32, device = 'cuda')
        cnt = 0

        for idx, act in enumerate(self.act_w):
            loss += F.mse_loss(self._normalize(act), self._normalize(dense_acts[idx]), reduction = "mean")
            cnt += 1

        self.clear_capture()

        return loss / cnt

    def _hook(self, _, __, output): return self.act_w.append(output.view(output.shape[0], -1))
    def _fhook(self, func, _, __, output): return self.act_w.append(func(output).view(output.shape[0], -1))


class OldKld(ActivationConcrete):

    
    def build(self, desired_sparsity: float, optimizer, 
              optimizer_kwargs: dict, 
              transforms: Tuple[Callable],
              gradbalance = False):
        
        super().build(desired_sparsity, optimizer, optimizer_kwargs, transforms, gradbalance)
        self._loss_scaler_constant *= 100 

    def _compute_loss(self, x, y):
        
        self.clear_capture()

        with torch.no_grad(): 
            self.mm.deactivate_mask()
            self.mm(x)
            dense_acts = torch.cat([act.log_softmax(1) for act in self.act_w], dim = 1)
            self.clear_capture()
            self.mm.activate_mask()
            #dense_acts = dense_acts.div(dense_acts.sum(dim = 1, keepdim = True))
            #dense_acts = dense_acts.log_softmax(1)
            
            
        self.mm(x)
        curr_acts = torch.cat([act.log_softmax(1) for act in self.act_w], dim = 1)
        self.clear_capture()
        #curr_acts = curr_acts.div(curr_acts.sum(dim = 1, keepdim = True))
        #curr_acts = curr_acts.log_softmax(1)

        loss = F.kl_div(curr_acts, dense_acts, log_target = True, reduction = "batchmean")

        return loss

    def _hook(self, _, __, output): return self.act_w.append(output.view(output.shape[0], -1) )#+ 1e-8)
    def _fhook(self, func, _, __, output): return self.act_w.append(func(output).view(output.shape[0], -1) )#+ 1e-8)



class TrajectoryConcrete(FrozenConcrete):

    def __init__(self, rank: int, 
                 world_size: int, 
                 model: DDP,
                 concrete_temperature = 2./3.):
        
        """
        Should be used after training, optimizer state is optimizer.state_dict() output
        type_of_optimizer: "sgd"
        """
        
        super().__init__(rank, world_size, model, concrete_temperature=concrete_temperature)

        self.lottery_weights = [layer.get_parameter(layer.MASKED_NAME) for layer in self.mm.lottery_layers]
        self.weights = self.lottery_weights#list(self.mm.parameters())
        self.lottery_weight_ids = set(id(weight) for weight in self.lottery_weights)
        self.weight_ids = set(id(weight) for weight in self.weights)

        #self._make_optimizer_state(optimizer_state['state'])
        #self.weight_decay = optimizer_state['param_groups'][0]['weight_decay']
        #self.momentum = optimizer_state['param_groups'][0]['momentum']
        #self.dampening = optimizer_state['param_groups'][0]['dampening'] # DOES NOT IMPLEMENT

    def _make_optimizer_state(self, state):
        
        self.momentum_state = {}
        return
        for idx, param in enumerate(self.m.parameters()):
                if id(param) in self.weight_ids: self.momentum_state[id(param)] = state[idx]['momentum_buffer']    

    def build(self, desired_sparsity: float, 
              optimizer, 
              optimizer_kwargs: dict, 
              transforms: Tuple[Callable],
              gradbalance = False):
        
        super().build(desired_sparsity, optimizer, optimizer_kwargs, transforms, gradbalance)
        
        for weight in self.weights: weight.requires_grad_(True)        
        self.mm.zero_grad()

    def _step_comparison_loss(self, step1, step2):
        raise NotImplementedError

    def _compute_loss(self, x, y):

        self.mm.deactivate_mask()
        loss = F.cross_entropy(self.mm(x), y)
        grad_d = torch.autograd.grad(loss, self.weights)
        grad_d_mom = list()
        cnt = 0
        with torch.no_grad():
            for weight, grad in zip(self.weights, grad_d):
                curr_id = id(weight)
                if curr_id in self.lottery_weight_ids:
                    #weight = weight * F.sigmoid(self.mm.lottery_layers[cnt].weight_mask / self.mm.concrete_temperature)
                    cnt += 1
                grad_d_mom.append(grad.detach() )#+ self.momentum * self.momentum_state[curr_id])# + self.weight_decay * weight + self.momentum * self.momentum_state[curr_id])

        self.mm.activate_mask()
        loss = F.cross_entropy(self.mm(x), y)
        grad_s = torch.autograd.grad(loss, self.weights, create_graph = True)
        grad_s_mom = list()
        cnt = 0
        for weight, grad in zip(self.weights, grad_s):
            curr_id = id(weight)
            if curr_id in self.lottery_weight_ids:
                #weight = weight * F.sigmoid(self.mm.lottery_layers[cnt].weight_mask / self.mm.concrete_temperature)
                cnt += 1
            grad_s_mom.append(grad )#+ self.momentum * self.momentum_state[curr_id])#+ self.weight_decay * weight + self.momentum * self.momentum_state[curr_id])
        
        return self._step_comparison_loss(grad_s_mom, grad_d_mom)
    

class StepAlignmentConcrete(TrajectoryConcrete):

    def build(self, desired_sparsity: float, 
              optimizer, 
              optimizer_kwargs: dict, 
              transforms: Tuple[Callable],
              gradbalance = False):
        
        super().build(desired_sparsity, optimizer, optimizer_kwargs, transforms, gradbalance)
        
        self._loss_scaler_constant *= 100

    def _normalize(self, grad: torch.Tensor):
        
        mean = grad.mean()
        std = grad.std()

        return grad.sub(mean).div(std + 1e-9)

    def _step_comparison_loss(self, step1, step2):

        step1 = [grad.sub(grad.mean()).div(grad.std()).view(-1) for grad in step1]
        step2 = [grad.sub(grad.mean()).div(grad.std()).view(-1) for grad in step2]

        """max1 = step1.abs().amax().clamp(min = 1e-30)
        max2 = step2.abs().amax().clamp(min = 1e-30)

        norm1 = step1.div(max1).norm(2)
        norm2 = step2.div(max2).norm(2)"""

        loss = torch.as_tensor(0.0, dtype = torch.float32, device = 'cuda')
        cnt = 0

        for sparse, dense in zip(step1, step2):
            loss += F.mse_loss(sparse, dense, reduction = "mean")
            cnt += 1

        #difference = (step1 - step2)

        #maxc = difference.abs().amax().clamp(min = 1e-8).div(difference.numel())
        #dist = difference.div(maxc).norm(2) * maxc

        #max_shift = maxc.div(torch.sqrt(max1 * max2))

        return loss / cnt
