import gc
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader
from typing import Callable, List, Tuple

from models.base import MaskedModel
import importlib

__all__ = ["SNIP_Pruner", "SynFlow_Pruner", "GraSP_Pruner", "OldKld_Pruner", "MSE_Pruner", "KldLogit_Pruner", "GradMatch_Pruner"]


class SaliencyPruning:
    def __init__(self, rank: int, world_size: int, model: Module | DDP,
                 data_module: str = "data.cifar10",
                 capture_layers: List[Module] = None,
                 fake_capture_layers: List[Tuple[Module, Callable]] = None,):
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
        self.cached_loader = None
        self.data_module = importlib.import_module(data_module)

    def build(self, sparsity_rate: float, transforms: Tuple[Callable],
              input: torch.Tensor | DataLoader, input_sampler_offset: int = None):
        self.spr = sparsity_rate
        self.transforms = transforms
        
        if isinstance(input, DataLoader):
            print("Warning: DataLoader provided. Fetching one batch for single-shot pruning.")
            if input_sampler_offset is not None and hasattr(input, 'sampler'):
                input.sampler.set_epoch(input_sampler_offset)
            data_tuple = next(iter(input)) 
            self.inp = (data_tuple[0], data_tuple[1]) 
            if self.DISTRIBUTED: print("Warning: Single Shot Data in Use; Not Distributed")
            torch.cuda.empty_cache()
            gc.collect()
        elif input is None:
            self.inp = self.get_single_shot_data(is_last = False)
            if self.DISTRIBUTED: print("Warning: Single Shot Data in Use; Not Distributed")
            torch.cuda.empty_cache()
            gc.collect()
        else:
            self.inp = input

        self.leafed_state_dict = self.m.state_dict()
        self.m.eval()
        for param in self.mm.parameters():
            param.grad = None
            param.requires_grad_(False)
        self.init_hooks()

    def finish(self):
        del self.cached_loader
        self.m.train(self.original_training_mode)
        if self.leafed_state_dict:
            self.m.load_state_dict(self.leafed_state_dict)
        for param in self.mm.parameters():
            param.requires_grad_(True)
        self.remove_handles()

    def get_multishot_data(self, batch_count):
        return self.data_module.get_partial_train_loader(self.RANK, self.WORLD_SIZE, batch_count = batch_count, batch_size = 512)

    def get_single_shot_data(self, is_last = True) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.cached_loader != None: loader = self.cached_loader
        else: loader, _ = self.data_module.get_loaders(self.RANK, self.WORLD_SIZE, validation = False)
        if not is_last: self.cached_loader = loader
        sampler_offset = int(torch.randint(1, 2**16, (1,)))
        out_data = self.data_module.custom_fetch_data(loader, 1, sampler_offset=sampler_offset)
        out = (out_data[0][0], out_data[0][1]) 
        if is_last: del self.cached_loader, loader
        return out


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
    def grad_mask(self, *args, **kwargs): raise NotImplementedError

    def _accumulate_grad_saliency(self, x: torch.Tensor, y: torch.Tensor, weights: List[torch.Tensor], grad_w: List[torch.Tensor], loss_func: Callable, **kwargs):
        """
        Generic function to compute and accumulate |grad(Loss)| saliency.
        Loss function is expected to handle the forward pass internally.
        """
        for T in self.transforms: x = T(x)
        
        kwargs["weights"] = weights
        loss = loss_func(x, y, **kwargs)
        
        grads = torch.autograd.grad(loss, weights)

        if len(grad_w) == 0:
            grad_w.extend((grad.abs() for grad in grads))
        else:
            for idx, grad in enumerate(grads):
                grad_w[idx].add_(grad.abs())

    def _get_ticket_from_accumulated_grads(self, weights: List[torch.Tensor], grad_w: List[torch.Tensor], curr_sp: float, is_signed: bool = False, current_mask: torch.Tensor = None):
        """
        Computes the final scores from accumulated gradients and returns the mask (ticket).
        """
        
        if not is_signed:
            grads = [(w.data * g).abs() for w, g in zip(weights, grad_w)]
        else:
            grads = [-w.data * g for w, g in zip(weights, grad_w)] 
            
        scores = torch.cat([g.view(-1) for g in grads]).to(torch.float64)
        
        #if current_mask is None:
        #    current_mask = self.mm.get_buffer("MASK").to(scores.device)

        #scores = scores * current_mask.to(scores.device)            
            
        if self.DISTRIBUTED: dist.all_reduce(scores, op=dist.ReduceOp.SUM)

        ticket = torch.zeros_like(scores, dtype=torch.bool)
        if self.IsRoot:
            num_to_keep = int(curr_sp * scores.numel())
            
            if not is_signed:
                
                threshold = torch.kthvalue(scores, scores.numel() - num_to_keep).values
                ticket = scores.ge(threshold)
            else:

                threshold = torch.kthvalue(scores, num_to_keep).values
                ticket = scores.le(threshold)

            print(f"{curr_sp * 100:.3f}% | Saliency (Sum): {scores.sum()} | Pruning Threshold: {threshold}")

        if self.DISTRIBUTED: dist.broadcast(ticket, src=0)
        
        if current_mask is not None:
            return ticket * self.mm.get_buffer("MASK")
        return ticket


class GraSP_Pruner(SaliencyPruning):
    """
    GraSP-based pruning compatible with a DDP model wrapper.
    Implements the *original* GraSP two-pass algorithm with micro-batching.
    """
    
    def __init__(self, rank: int, world_size: int, model: Module | DDP,
                 data_module: str = "data.cifar10"):
        super().__init__(rank, world_size, model, data_module)

    def _loss_func(self, x, y, weights, inv_temperature, grad_w_sum, **kwargs):

        for T in self.transforms: x = T(x)

        outputs = self.mm(x) * inv_temperature
        loss = F.cross_entropy(outputs, y)
        
        grad_f = torch.autograd.grad(loss, weights, create_graph=True)
        
        return sum(torch.sum(g * h) for g, h in zip(grad_w_sum, grad_f))        


    def _single_shot_grad_mask(self, weights: List[torch.Tensor], grad_w: list, temperature: float, improved: bool, curr_sp: float, micro_batch_size: int):
        x_full, y_full = self.inp
        inv_temperature = 1 / temperature
        N = x_full.shape[0]
        if N < 2: raise ValueError("GraSP requires a batch size of at least 2.")

        grad_w_sum = None 
        self.mm.zero_grad() 
        start = 0
        while start < N:
            end = min(start + micro_batch_size, N)
            x_micro, y_micro = x_full[start:end].cuda(), y_full[start:end].cuda()
            
            for T in self.transforms: x_micro = T(x_micro)

            outputs = self.mm(x_micro) * inv_temperature
            loss = F.cross_entropy(outputs, y_micro)
            
            grad_w_p = torch.autograd.grad(loss, weights, create_graph=False)
            
            if grad_w_sum is None:
                grad_w_sum = list(grad_w_p)
            else:
                for idx in range(len(grad_w_sum)):
                    grad_w_sum[idx].add_(grad_w_p[idx])
            
            start = end
        
        self.mm.zero_grad() 
        
        start = 0
        while start < N:
            end = min(start + micro_batch_size, N)
            x_micro, y_micro = x_full[start:end].cuda(), y_full[start:end].cuda()
            self._accumulate_grad_saliency(x_micro, y_micro, weights, grad_w, self._loss_func, inv_temperature = inv_temperature, grad_w_sum = grad_w_sum)
            start = end
        
        return self._get_ticket_from_accumulated_grads(weights, grad_w, curr_sp, is_signed = not improved)

    def grad_mask(self, temperature: float = 200.0, improved: bool = False, steps: int = 1, micro_batch_size: int = 20):
        """
        If improved = False, it is not improved (original GraSP: keep smallest)
        If improved = True, it is improved (keep largest magnitude)
        """
        
        if steps > 1 and improved == "2": raise ValueError("Not Implemented, Run Seperately.")

        if self.spr == 1.: return torch.ones_like(self.mm.get_buffer("MASK"))

        weights = [layer.weight for layer in self.mm.lottery_layers]
        for param in self.mm.parameters():
            param.requires_grad_(any(param is p for p in weights))
        
        grad_w = list()

        for n in range(steps):
            
            self.mm.zero_grad()

            curr_sp = self.spr ** ((n + 1) / steps)
            
            out = self._single_shot_grad_mask(weights, grad_w, temperature, improved, curr_sp, micro_batch_size)
            self.mm.set_ticket(out)
            
            if n < steps - 1: 
                self.inp = self.get_single_shot_data(is_last = False)
        
        return out

# --- SNIP Pruner ---

class SNIP_Pruner(SaliencyPruning):
    
    def __init__(self, rank: int, world_size: int, model: Module | DDP,
                 data_module: str = "data.cifar10"):
        super().__init__(rank, world_size, model, data_module)
    
    def _loss_func(self, x, y, **kwargs):
        return F.cross_entropy(self.mm(x), y)

    def _single_shot_grad_mask(self, weights, grad_w, curr_sp, micro_batch_size):
        x_full, y_full = self.inp
        N = x_full.shape[0]

        start = 0
        while start < N:
            end = min(start + micro_batch_size, N)
            self._accumulate_grad_saliency(x_full[start:end].cuda(), y_full[start:end].cuda(), weights, grad_w, self._loss_func)
            start += micro_batch_size
        
        return self._get_ticket_from_accumulated_grads(weights, grad_w, curr_sp, is_signed=False)

    def grad_mask(self, steps: int = 1, micro_batch_size: int = 64):
        if self.spr == 1.: return torch.ones_like(self.mm.get_buffer("MASK"))

        self.mm.zero_grad()
        weights = [layer.weight for layer in self.mm.lottery_layers]
        for param in self.mm.parameters():
            param.requires_grad_(any(param is p for p in weights)) 
        
        grad_w = list()

        for n in range(steps):
            self.mm.zero_grad()
            curr_sp = self.spr ** ((n + 1) / steps)
            out = self._single_shot_grad_mask(weights, grad_w, curr_sp, micro_batch_size)
            self.mm.set_ticket(out)
            if n < steps - 1: 
                self.inp = self.get_single_shot_data(is_last = False)
                grad_w.clear()
        
        return out

# --- KldLogit Pruner ---

class KldLogit_Pruner(SaliencyPruning):
    def __init__(self, rank: int, world_size: int, model: Module | DDP,
                 data_module: str = "data.cifar10", reverse = True):
        super().__init__(rank, world_size, model, data_module)
        self.reversekld = reverse

    def _loss_func(self, x, y, ticket, **kwargs):
        
        self.mm.reset_ticket()
        with torch.no_grad(): dense = self.mm(x).detach().log_softmax(1)
        
        self.mm.set_ticket(ticket)
        sparse = self.mm(x).log_softmax(1)

        if not self.reversekld: loss = F.kl_div(sparse, dense, reduction = 'batchmean', log_target = True)   
        else: loss = F.kl_div(dense, sparse, reduction = 'batchmean', log_target = True)   
        
        return loss

    def _single_shot_grad_mask(self, weights, grad_w, curr_sp, micro_batch_size):
        x_full, y_full = self.inp
        N = x_full.shape[0]
        
        current_ticket = self.mm.export_ticket_cpu()

        start = 0
        while start < N:
            end = min(start + micro_batch_size, N)
            
            self._accumulate_grad_saliency(x_full[start:end].cuda(), y_full[start:end].cuda(), weights, grad_w, self._loss_func, ticket=current_ticket)
            
            start += micro_batch_size

        self.mm.set_ticket(current_ticket)

        return self._get_ticket_from_accumulated_grads(weights, grad_w, curr_sp, is_signed=False)

    def grad_mask(self, steps: int = 1, micro_batch_size: int = 64):
        if self.spr == 1.: return torch.ones_like(self.mm.get_buffer("MASK"))

        self.mm.zero_grad()

        self.mm.zero_grad()
        weights = [layer.weight for layer in self.mm.lottery_layers]
        for param in self.mm.parameters():
            param.requires_grad_(any(param is p for p in weights)) 
        
        grad_w = list()

        for n in range(steps):
            curr_sp = self.spr ** ((n + 1) / steps)
            out = self._single_shot_grad_mask(weights, grad_w, curr_sp, micro_batch_size)
            self.mm.set_ticket(out)
            if n < steps - 1: 
                self.inp = self.get_single_shot_data(is_last = False)
                grad_w.clear()
        
        return out

# --- GradMatch Pruner ---

class GradMatch_Pruner(SaliencyPruning):
    
    def __init__(self, rank: int, world_size: int, model: Module | DDP,
                 data_module: str = "data.cifar10"):
        super().__init__(rank, world_size, model, data_module)

    def _loss_func(self, x, y, weights, ticket, **kwargs):
        
        self.mm.reset_ticket()
        taskd = F.cross_entropy(self.mm(x), y)
        with torch.no_grad():
            grad_d = torch.autograd.grad(taskd, weights) 
   
        self.mm.set_ticket(ticket)
        task = F.cross_entropy(self.mm(x), y)
        grad_s = torch.autograd.grad(task, weights, create_graph = True) 

        grad_s = [grad.sub(grad.mean()).div(grad.std() + 1e-12).view(-1) for grad in grad_s]
        grad_d = [grad.detach().sub(grad.mean()).div(grad.std() + 1e-12).view(-1) for grad in grad_d]
        
        mse_loss = torch.as_tensor(0.0, dtype = torch.float32, device = 'cuda')
        for sparse, dense in zip(grad_s, grad_d):
            mse_loss += F.mse_loss(sparse, dense, reduction = "mean")

        return mse_loss


    def _single_shot_grad_mask(self, weights, grad_w, curr_sp, micro_batch_size):
        x_full, y_full = self.inp
        N = x_full.shape[0]
        
        current_ticket = self.mm.export_ticket_cpu()

        start = 0
        while start < N:
            end = min(start + micro_batch_size, N)
            x_micro, y_micro = x_full[start:end].cuda(), y_full[start:end].cuda()

            for T in self.transforms: x_micro = T(x_micro)

            mse_loss = self._loss_func(x_micro, y_micro, weights, current_ticket)
            
            grads = torch.autograd.grad(mse_loss, weights)
            
            if len(grad_w) == 0:
                grad_w.extend((grad.abs() for grad in grads))
            else:
                for idx, grad in enumerate(grads):
                    grad_w[idx].add_(grad.abs())
            # -----------------------------------------------------------

            start += micro_batch_size
        
        self.mm.set_ticket(current_ticket)

        return self._get_ticket_from_accumulated_grads(weights, grad_w, curr_sp, is_signed=False, current_mask=self.mm.get_buffer("MASK"))

    def grad_mask(self, steps: int = 1, micro_batch_size: int = 64):
        if self.spr == 1.: return torch.ones_like(self.mm.get_buffer("MASK"))

        self.mm.zero_grad()
        weights = [layer.weight for layer in self.mm.lottery_layers]
        for param in self.mm.parameters():
            param.requires_grad_(any(param is p for p in weights)) 

        grad_w = list()

        for n in range(steps):
            self.mm.zero_grad()
            curr_sp = self.spr ** ((n + 1) / steps)
            out = self._single_shot_grad_mask(weights, grad_w, curr_sp, micro_batch_size)
            self.mm.set_ticket(out)
            if n < steps - 1: 
                self.inp = self.get_single_shot_data(is_last = False)
                grad_w.clear()

        return out

# --- SynFlow Pruner ---

class SynFlow_Pruner(SaliencyPruning):
    """
    SynFlow is data-independent and uses a single dummy input, so it does not
    use micro-batching on the data sample itself.
    """
    
    def __init__(self, rank: int, world_size: int, model: Module | DDP,
                 data_module: str = "data.cifar10"):
        super().__init__(rank, world_size, model, data_module)

    def _accumulate_saliency(self, inp_shape, weights):
        inp = torch.ones(inp_shape, dtype= torch.float32, device="cuda")
        return torch.autograd.grad(torch.sum(self.mm(inp)), weights)


    def _single_shot_grad_mask(self, weights, curr_sp):
        x, y = self.inp
        inp_shape = [1,] + list(x[0,:].shape)
        
        grad_w = self._accumulate_saliency(inp_shape, weights)
        grads = [(w * g).detach().abs() for w, g in zip(weights, grad_w)]
        self.mm.zero_grad()
        scores = torch.cat([g.view(-1) for g in grads])
        num_to_keep = int((curr_sp) * scores.numel())
        
        threshold = torch.kthvalue(scores, scores.numel() - num_to_keep).values
        ticket = scores.ge(threshold)
        print(f"{curr_sp * 100:.3f}% | Saliency (Sum): {scores.sum()} | Pruning Threshold: {threshold}")
        
        return ticket * self.mm.get_buffer("MASK")

    def grad_mask(self, steps: int = 100, micro_batch_size: int = 1):

        if self.spr == 1.: return torch.ones_like(self.mm.get_buffer("MASK"))

        self.mm.zero_grad()
        weights = [layer.get_parameter(layer.MASKED_NAME) for layer in self.mm.lottery_layers]

        @torch.no_grad()
        def linearize(model):
            signs = dict()
            for name, param in model.named_parameters():
                param.abs_()
                signs[name] = torch.sign(param.data)
                param.requires_grad_(any(param is p for p in weights))
            return signs
        
        @torch.no_grad()
        def unlinearize(model, signs):
            for name, param in model.named_parameters():
                with torch.no_grad():
                    param.data.mul_(signs[name])
                        
        signs = linearize(self.mm)

        for n in range(steps):
            curr_sp = self.spr ** ((n + 1) / steps)
            out = self._single_shot_grad_mask(weights, curr_sp)
            self.mm.set_ticket(out)

        unlinearize(self.mm, signs)

        return out

# --- Activation Saliency Pruning Base ---

class ActivationSaliencyPruning(SaliencyPruning):
    """Base class for activation-based pruning methods.
    Overlay logic is included so that gradients are not calculated at global minima."""
    def __init__(self, rank: int, world_size: int, model: DDP,
                 capture_layers: List[Module],
                 fake_capture_layers: List[Tuple[Module, Callable]],
                 data_module: str = "data.cifar10",):
        super().__init__(rank, world_size, model, data_module, capture_layers, fake_capture_layers)
        self.full_activations = []

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)

    def _make_full_activations(self, micro_batch_size: int): raise NotImplementedError

    def _apply_overlay(self, base_overlay, noise, clamp_range):
        offset = 0
        overlay = torch.clamp(base_overlay + noise, *clamp_range)
        for layer in self.mm.lottery_layers:
            layer.get_parameter(layer.MASKED_NAME).mul_(overlay[offset: offset + layer.MASK_NUMEL].view(layer.MASK_SHAPE))
            offset += layer.MASK_NUMEL

    def _remove_overlay(self, base_overlay, noise, clamp_range):
        with torch.no_grad():
            offset = 0
            inv_overlay = torch.clamp(base_overlay + noise, *clamp_range).double().reciprocal().float()
            for layer in self.mm.lottery_layers:
                layer.get_parameter(layer.MASKED_NAME).mul_(inv_overlay[offset: offset + layer.MASK_NUMEL].view(layer.MASK_SHAPE))
                offset += layer.MASK_NUMEL

    def _get_ticket(self, magnitudes: torch.Tensor, spr: float):
        ticket = torch.zeros_like(magnitudes, dtype=torch.bool)
        if self.IsRoot:
            num_to_keep = int(spr * magnitudes.numel())
            threshold = torch.kthvalue(magnitudes, magnitudes.numel() - num_to_keep).values
            ticket = magnitudes.ge(threshold)
            print(f"{spr * 100:.3f}% | Saliency (Sum): {magnitudes.sum()} | Pruning Threshold: {threshold}")
        if self.DISTRIBUTED: dist.broadcast(ticket, src=0)
        return ticket

    def grad_mask(self, steps: int = 1, micro_batch_size: int = 64):     

        if self.spr == 1.: return torch.ones_like(self.mm.get_buffer("MASK"))

        self._make_full_activations(micro_batch_size)

        for n in range(steps):
            self.mm.zero_grad()
            curr_sp = self.spr ** ((n + 1) / steps)
            out = self._single_shot_grad_mask(curr_sp, micro_batch_size)
            self.mm.set_ticket(out)
            if n < steps - 1: 
                self.inp = self.get_single_shot_data(is_last = False)
                self.full_activations.clear()
                self._make_full_activations(micro_batch_size)
        
        return out

# --- OldKld Pruner ---

class OldKld_Pruner(ActivationSaliencyPruning):
    """Pruning based on KL divergence of activation distributions."""
    def _make_full_activations(self, micro_batch_size: int):
        x_full, *_ = self.inp
        
        micro_acts = []
        N = x_full.shape[0]
        start = 0
        with torch.no_grad():
            while start < N:
                end = min(start + micro_batch_size, N)
                x_micro = x_full[start:end].cuda()
                
                self.mm(x_micro)
                act_mask = torch.cat([act.log_softmax(1) for act in self.act_w], dim = 1)
                micro_acts.append(act_mask)
                self.clear_capture()
                start += micro_batch_size
        
        self.full_activations.append(torch.cat(micro_acts, dim=0))

    def _accumulate_saliency(self, x, full_activations_micro, overlay):
        self.mm(x.cuda())
        curr_acts = torch.cat([act.log_softmax(1) for act in self.act_w], dim = 1)
        kl_loss = F.kl_div(curr_acts, full_activations_micro.cuda(), reduction="batchmean", log_target=True)
        self.clear_capture()
        return torch.autograd.grad(kl_loss, overlay)[0]

    def _single_shot_grad_mask(self, curr_sp, micro_batch_size):
        overlay = torch.ones(self.mm.num_prunable, device=f'cuda:{self.RANK}', requires_grad=True)
        
        x_full, *_ = self.inp
        full_acts_tensor = self.full_activations[0]
        
        magnitudes = torch.zeros(self.mm.num_prunable, device=f'cuda:{self.RANK}', dtype=torch.float64)

        N = x_full.shape[0]
        start = 0
        while start < N:
            end = min(start + micro_batch_size, N)
            x_micro = x_full[start:end]
            acts_micro = full_acts_tensor[start:end]

            noise = torch.randn_like(overlay) * 6e-2
            self._apply_overlay(overlay, noise, (0.8, 1.2))
            
            mag_micro = self._accumulate_saliency(x_micro, acts_micro, overlay).detach().to(torch.float64)
            magnitudes += mag_micro
            
            self._remove_overlay(overlay, noise, (0.8, 1.2))
            
            start += micro_batch_size
        
        return self._get_ticket(magnitudes.abs(), curr_sp)

    def _hook(self, _, __, output): self.act_w.append(output.to(torch.float64).view(output.shape[0], -1) )
    def _fhook(self, func, _, __, output): self.act_w.append(func(output.to(torch.float64)).view(output.shape[0], -1) )


# --- MSE Pruner ---

class MSE_Pruner(ActivationSaliencyPruning):
    """Pruning based on MSE of activation outputs."""
    def _make_full_activations(self, micro_batch_size: int):
        x_full, *_ = self.inp
        
        collected_acts = []
        N = x_full.shape[0]
        start = 0
        with torch.no_grad():
            while start < N:
                end = min(start + micro_batch_size, N)
                x_micro = x_full[start:end].cuda()

                self.mm(x_micro)
                
                if not collected_acts:
                    collected_acts = [[] for _ in self.act_w]
                
                for i, act in enumerate(self.act_w):
                    collected_acts[i].append(act.detach())
                
                self.clear_capture()
                start += micro_batch_size
        
        self.full_activations.append([torch.cat(layer_acts, dim=0) for layer_acts in collected_acts])

    def _accumulate_saliency(self, x, full_activations_micro, overlay):
        
        self.mm(x.cuda())

        mse_loss = torch.as_tensor(0.0, dtype = torch.float32, device = 'cuda')
        for act_idx, act in enumerate(self.act_w):
            
            std_full, mean_full = torch.std_mean(full_activations_micro[act_idx], dim = 0, keepdim = True)
            
            act_norm = act.sub(act.mean(dim = 0, keepdim = True)).div(act.std(dim = 0, keepdim = True) + 1e-12)
            full_norm = full_activations_micro[act_idx].sub(mean_full).div(std_full + 1e-12)

            mse_loss += F.mse_loss(act_norm, full_norm, reduction = "mean")
            
        self.clear_capture()
        return torch.autograd.grad(mse_loss, overlay)[0]

    def _single_shot_grad_mask(self, curr_sp, micro_batch_size):
        overlay = torch.ones(self.mm.num_prunable, device=f'cuda', requires_grad=True)
        
        x_full, *_ = self.inp
        full_acts_list = self.full_activations[0]
        
        magnitudes = torch.zeros(self.mm.num_prunable, device=f'cuda', dtype=torch.float64)

        N = x_full.shape[0]
        start = 0
        while start < N:
            end = min(start + micro_batch_size, N)
            x_micro = x_full[start:end]
            acts_micro_list = [full_act[start:end] for full_act in full_acts_list]

            noise = torch.randn_like(overlay) * 6e-2
            self._apply_overlay(overlay, noise, (0.8, 1.2))
            
            mag_micro = self._accumulate_saliency(x_micro, acts_micro_list, overlay).detach().to(torch.float64)
            magnitudes += mag_micro
            
            self._remove_overlay(overlay, noise, (0.8, 1.2))
            
            start += micro_batch_size
        
        magnitudes = magnitudes.abs()
        return self._get_ticket(magnitudes, curr_sp)

    def _hook(self, _, __, output): self.act_w.append(output.view(output.shape[0], -1))
    def _fhook(self, func, _, __, output): self.act_w.append(func(output).view(output.shape[0], -1))