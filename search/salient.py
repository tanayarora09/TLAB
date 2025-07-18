import gc
import torch
import torch.nn.functional as F
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
from torch.utils.data import DataLoader
from typing import Callable, List, Tuple

from models.base import BaseModel
from data.cifar10 import get_loaders, custom_fetch_data


__all__ = ["SNIP_Pruner", "SynFlow_Pruner", "GraSP_Pruner", "KLD_Pruner", "MSE_Pruner"]


class SaliencyPruning:
    def __init__(self, rank: int, world_size: int, model: Module | DDP,
                 capture_layers: List[Module] = None,
                 fake_capture_layers: List[Tuple[Module, Callable]] = None):
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

    def build(self, sparsity_rate: float, transforms: Tuple[Callable],
              input: torch.Tensor | DataLoader, input_sampler_offset: int = None):
        self.spr = sparsity_rate
        self.transforms = transforms
        self.inp = input

        if self.inp is None:
            self.inp = self.get_single_shot_data()
            self.running = False
            if self.DISTRIBUTED: print("Warning: Single Shot Data in Use; Not Distributed")
            torch.cuda.empty_cache()
            gc.collect()
        else:
            self.running = isinstance(input, DataLoader)
            if self.running and input_sampler_offset is not None and hasattr(self.inp, 'sampler'):
                self.inp.sampler.set_epoch(input_sampler_offset)

        self.leafed_state_dict = self.m.state_dict()
        self.m.eval()
        for param in self.mm.parameters():
            param.grad = None
            param.requires_grad_(False)
        self.init_hooks()

    def finish(self):
        self.m.train(self.original_training_mode)
        if self.leafed_state_dict:
            self.m.load_state_dict(self.leafed_state_dict)
        for param in self.mm.parameters():
            param.requires_grad_(True)
        self.remove_handles()

    def get_single_shot_data(self) -> Tuple[torch.Tensor, torch.Tensor]:
        loader, _ = get_loaders(self.RANK, self.WORLD_SIZE)
        sampler_offset = int(torch.randint(1, 2**16, (1,)))
        return custom_fetch_data(loader, 1, sampler_offset=sampler_offset)[0]

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


class GraSP_Pruner(SaliencyPruning):
    """GraSP-based pruning compatible with a DDP model wrapper."""
    def __init__(self, rank: int, world_size: int, model: Module | DDP):
        super().__init__(rank, world_size, model)

    def _accumulate_saliency(self, x, y, weights, temperature):
        N = x.shape[0]
        if N < 2: raise ValueError("GraSP requires a batch size of at least 2.")

        loss1 = F.cross_entropy(self.mm(x[:N//2]) / temperature, y[:N//2])
        grad_w = torch.autograd.grad(loss1, weights, create_graph=True)
        
        loss2 = F.cross_entropy(self.mm(x[N//2:]) / temperature, y[N//2:])
        grad_w_p = torch.autograd.grad(loss2, weights)

        total_grad = [g1 + g2 for g1, g2 in zip(grad_w, grad_w_p)]
        z = sum(torch.sum(g * h) for g, h in zip(grad_w, total_grad))
        z.backward()

    def _single_shot_grad_mask(self, weights, temperature: float, improved: str):
        x, y = self.inp
        x, y = x.cuda(), y.cuda()
        for T in self.transforms: x = T(x)
        self._accumulate_saliency(x, y, weights, temperature)
        grads = [-w.data * w.grad for w in weights]
        scores = torch.cat([g.view(-1) for g in grads]).to(torch.float64)
        num_to_keep = int((self.spr) * scores.numel())
        
        if improved == "0":
            threshold = torch.kthvalue(scores, num_to_keep).values
            ticket = scores.le(threshold)
            print(f"Saliency (Sum): {scores.sum()} | Pruning Threshold: {threshold}")
        
        elif improved == "1":
            scores.abs_()
            threshold = torch.kthvalue(scores, scores.numel() - num_to_keep).values
            ticket = scores.ge(threshold)
            print(f"Saliency (Sum): {scores.sum()} | Pruning Threshold: {threshold}")
        
        elif improved == "2":
            threshold = torch.kthvalue(scores, num_to_keep).values
            ticket1 = scores.le(threshold)
            print(f"Saliency (Sum): {scores.sum()} | Pruning Threshold: {threshold}")
            scores.abs_()
            threshold = torch.kthvalue(scores, scores.numel() - num_to_keep).values
            ticket2 = scores.ge(threshold)
            print(f"Saliency (Sum): {scores.sum()} | Pruning Threshold: {threshold}")
                
        
        if improved == "2": return (ticket1, ticket2)
        return ticket

    def _running_grad_mask(self, weights, temperature: float, improved: str):
        for x, y, *_ in self.inp:
            x, y = x.cuda(), y.cuda()
            for T in self.transforms: x = T(x)
            self._accumulate_saliency(x, y, weights, temperature)

        grads = [-w.data * w.grad for w in weights]
        scores = torch.cat([g.view(-1) for g in grads]).to(torch.float64)
        
        if self.DISTRIBUTED: dist.all_reduce(scores, op=dist.ReduceOp.SUM)

        ticket = torch.zeros_like(scores, dtype=torch.bool)
        if self.IsRoot:
            
            num_to_keep = int(self.spr * scores.numel())

            if improved == "0":
                threshold = torch.kthvalue(scores, num_to_keep).values
                ticket = scores.le(threshold)
            
            elif improved == "1":
                scores.abs_()
                threshold = torch.kthvalue(scores, scores.numel() - num_to_keep).values
                ticket = scores.ge(threshold)
            
            elif improved == "2":
                threshold = torch.kthvalue(scores, num_to_keep).values
                ticket1 = scores.le(threshold)
                scores.abs_()
                threshold = torch.kthvalue(scores, scores.numel() - num_to_keep).values
                ticket2 = scores.ge(threshold)
                ticket = torch.stack((ticket1, ticket2))

            else: raise ValueError()

        if self.DISTRIBUTED: dist.broadcast(ticket, src=0)
        if improved == "2": return torch.unbind(ticket) 
        return ticket

    def grad_mask(self, temperature: float = 200.0, improved: str = "0"):
        """
        If improved = "0", it is not improved
        If improved = "1", it is improved
        If improved = "2", both are turned as a tuple: (not_improved, improved)
        """
        self.mm.zero_grad()
        
        if self.spr == 1.: return torch.ones_like(self.mm.get_buffer("MASK"))

        weights = [layer.weight for layer in self.mm.lottery_layers]
        for param in self.mm.parameters():
            param.requires_grad_(any(param is p for p in weights))

        if not self.running:
            return self._single_shot_grad_mask(weights, temperature, improved)
        else:
            return self._running_grad_mask(weights, temperature, improved)


class SNIP_Pruner(SaliencyPruning):
    def __init__(self, rank: int, world_size: int, model: Module | DDP):
        super().__init__(rank, world_size, model)

    def _accumulate_saliency(self, x, y, weights, grad_w):
        loss = F.cross_entropy(self.mm(x), y)
        if len(grad_w) == 0:
            grad_w.extend((grad.abs() for grad in torch.autograd.grad(loss, weights)))
        else:
            for idx, grad in enumerate(torch.autograd.grad(loss, weights)):
                grad_w[idx] += grad.abs()

    def _single_shot_grad_mask(self, weights, grad_w):
        x, y = self.inp
        x, y = x.cuda(), y.cuda()
        for T in self.transforms: x = T(x)
        self._accumulate_saliency(x, y, weights, grad_w)
        grads = [(w.data * g).abs() for w, g in zip(weights, grad_w)]
        scores = torch.cat([g.view(-1) for g in grads]).to(torch.float64)
        num_to_keep = int((self.spr) * scores.numel())
        
        threshold = torch.kthvalue(scores, scores.numel() - num_to_keep).values
        ticket = scores.ge(threshold)
        print(f"Saliency (Sum): {scores.sum()} | Pruning Threshold: {threshold}")
        
        return ticket

    def _running_grad_mask(self, weights, grad_w):
        for x, y, *_ in self.inp:
            x, y = x.cuda(), y.cuda()
            for T in self.transforms: x = T(x)
            self._accumulate_saliency(x, y, weights, grad_w)

        grads = [(w.data * g).abs() for w, g in zip(weights, grad_w)]
        scores = torch.cat([g.view(-1) for g in grads]).to(torch.float64)
        
        if self.DISTRIBUTED: dist.all_reduce(scores, op=dist.ReduceOp.SUM)

        ticket = torch.zeros_like(scores, dtype=torch.bool)
        if self.IsRoot:
            
            num_to_keep = int(self.spr * scores.numel())
            threshold = torch.kthvalue(scores, scores.numel() - num_to_keep).values
            ticket = scores.ge(threshold)

        if self.DISTRIBUTED: dist.broadcast(ticket, src=0)
        return ticket

    def grad_mask(self):

        if self.spr == 1.: return torch.ones_like(self.mm.get_buffer("MASK"))

        self.mm.zero_grad()
        weights = [layer.weight for layer in self.mm.lottery_layers]
        for param in self.mm.parameters():
            param.requires_grad_(any(param is p for p in weights)) 
        
        grad_w = list()

        if not self.running:
            return self._single_shot_grad_mask(weights, grad_w)
        else:
            return self._running_grad_mask(weights, grad_w)


class SynFlow_Pruner(SaliencyPruning):
    def __init__(self, rank: int, world_size: int, model: Module | DDP):
        super().__init__(rank, world_size, model)

    def _accumulate_saliency(self, x, y, weights):
        loss = self.mm(torch.ones(x.shape, dtype=x.dtype, device=x.device)).sum()
        loss.backward()
        return None#[weight.grad.to(torch.float64) for weight in weights]


    def _single_shot_grad_mask(self, weights, curr_sp):
        x, y = self.inp
        x, y = x.cuda(), y.cuda()
        for T in self.transforms: x = T(x)
        grad_w = self._accumulate_saliency(x, y, weights)
        grads = [(w.data.to(torch.float64) * w.grad.data.to(torch.float64)).abs() for w in weights]
        self.mm.zero_grad()
        scores = torch.cat([g.view(-1) for g in grads]) * self.mm.get_buffer("MASK")
        num_to_keep = int((curr_sp) * scores.numel())
        
        threshold = torch.kthvalue(scores, scores.numel() - num_to_keep).values
        ticket = scores.ge(threshold)
        print(f"{curr_sp * 100:.3f}% | Saliency (Sum): {scores.sum()} | Pruning Threshold: {threshold} | Current Sparsity: {self.mm.sparsity:.3f}%")
        
        return ticket * self.mm.get_buffer("MASK")

    def grad_mask(self):

        if self.spr == 1.: return torch.ones_like(self.mm.get_buffer("MASK"))

        self.mm.zero_grad()
        weights = [layer.get_parameter(layer.MASKED_NAME) for layer in self.mm.lottery_layers]
        signs = dict()

        for pidx, param in enumerate(self.mm.parameters()):
            signs[pidx] = torch.sign(param)
            with torch.no_grad(): param.abs_()
            param.requires_grad_(any(param is weight for weight in weights))

        for n in range(100):
            curr_sp = self.spr ** ((n + 1) / 100)
            out = self._single_shot_grad_mask(weights, curr_sp)
            self.mm.set_ticket(out)
        
        with torch.no_grad():
            for pidx, param in enumerate(self.mm.parameters()):
                param.data.mul_(signs[pidx])
                        
        return out


class ActivationSaliencyPruning(SaliencyPruning):
    """Base class for activation-based pruning methods.
    Overlay logic is included so that gradients are not calculated at global minima."""
    def __init__(self, rank: int, world_size: int, model: DDP,
                 capture_layers: List[Module],
                 fake_capture_layers: List[Tuple[Module, Callable]]):
        super().__init__(rank, world_size, model, capture_layers, fake_capture_layers)
        self.full_activations = []

    def build(self, *args, **kwargs):
        super().build(*args, **kwargs)
        self._make_full_activations()

    def _make_full_activations(self): raise NotImplementedError

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


class KLD_Pruner(ActivationSaliencyPruning):
    """Pruning based on KL divergence of activation distributions."""
    def _make_full_activations(self):
        data_iterator = self.inp if self.running else [self.inp]
        for x, *_ in data_iterator:
            with torch.no_grad():
                self.mm(x.cuda())
            #for act in self.act_w: 
            #    act += 1e-8
            #    act.div_(act.sum(dim=1, keepdim=True) + 1e-8)
            act_mask = torch.cat([act.log_softmax(1) for act in self.act_w], dim = 1)
            #act_mask.div_(act_mask.sum(dim=1, keepdim=True))
            self.full_activations.append((act_mask).cpu())
            self.clear_capture()

    def _accumulate_saliency(self, x, full_activations):
        self.mm(x.cuda())
        curr_acts = torch.cat([act.log_softmax(1) for act in self.act_w], dim = 1)
        #curr_acts.div_(curr_acts.sum(dim=1, keepdim=True) + 1e-8)
        #curr_acts = (curr_acts).log_softmax(1)
        kl_loss = F.kl_div(curr_acts, full_activations.cuda(), reduction="batchmean", log_target=True)
        (kl_loss).backward()
        self.clear_capture()

    def _get_ticket(self, magnitudes):
        ticket = torch.zeros_like(magnitudes, dtype=torch.bool)
        if self.IsRoot:
            num_to_keep = int(self.spr * magnitudes.numel())
            threshold = torch.kthvalue(magnitudes, magnitudes.numel() - num_to_keep).values
            ticket = magnitudes.ge(threshold)
        if self.DISTRIBUTED: dist.broadcast(ticket, src=0)
        return ticket

    def _single_shot_grad_mask(self):
        overlay = torch.ones(self.mm.num_prunable, device=f'cuda:{self.RANK}', requires_grad=True)
        self._apply_overlay(overlay, torch.randn_like(overlay) * 6e-2, (0.8, 1.2))
        self._accumulate_saliency(self.inp[0], self.full_activations[0])
        self._remove_overlay(overlay, torch.randn_like(overlay) * 6e-2, (0.8, 1.2))
        magnitudes = (-1 * overlay.grad).detach().to(torch.float64)
        return self._get_ticket(magnitudes)

    def _running_grad_mask(self):
        overlay = torch.ones(self.mm.num_prunable, device=f'cuda:{self.RANK}', requires_grad=True)
        for idx, (x, *_) in enumerate(self.inp):
            self._apply_overlay(overlay, torch.randn_like(overlay) * 6e-2, (0.8, 1.2))
            self._accumulate_saliency(x, self.full_activations[idx])
            self._remove_overlay(overlay, torch.randn_like(overlay) * 6e-2, (0.8, 1.2))
            if self.DISTRIBUTED: dist.barrier()
        magnitudes = (-1 * overlay.grad).detach().to(torch.float64)
        if self.DISTRIBUTED: dist.all_reduce(magnitudes, op=dist.ReduceOp.SUM)
        return self._get_ticket(magnitudes)

    def grad_mask(self):
        
        if self.spr == 1.: return torch.ones_like(self.mm.get_buffer("MASK"))

        if not self.running: return self._single_shot_grad_mask()
        else: return self._running_grad_mask()

    def _hook(self, _, __, output): self.act_w.append(output.to(torch.float64).view(output.shape[0], -1) )#+ 1e-8)
    def _fhook(self, func, _, __, output): self.act_w.append(func(output.to(torch.float64)).view(output.shape[0], -1) )#+ 1e-8)


class MSE_Pruner(ActivationSaliencyPruning):
    """Pruning based on MSE of activation outputs."""
    def _make_full_activations(self):
        data_iterator = self.inp if self.running else [self.inp]
        for x, *_ in data_iterator:
            with torch.no_grad():
                self.mm(x.cuda())
            self.full_activations.append([act.detach().cpu() for act in self.act_w])
            self.clear_capture()

    def _accumulate_saliency(self, x, full_activations):
        self.mm(x.cuda())
        #curr_activations = torch.cat(self.act_w, dim=1)
        
        mse_loss = torch.as_tensor(0.0, dtype = torch.float32, device = 'cuda')
        for act_idx, act in enumerate(self.act_w):
            mse_loss += F.mse_loss(act, full_activations[act_idx].cuda(), reduction = "mean")

        mse_loss.backward()
        self.clear_capture()

    def _get_ticket(self, magnitudes):
        ticket = torch.zeros_like(magnitudes, dtype=torch.bool)
        if self.IsRoot:
            num_to_keep = int(self.spr * magnitudes.numel())
            threshold = torch.kthvalue(magnitudes, magnitudes.numel() - num_to_keep).values
            ticket = magnitudes.ge(threshold)
        if self.DISTRIBUTED: dist.broadcast(ticket, src=0)
        return ticket

    def _single_shot_grad_mask(self):
        overlay = torch.ones(self.mm.num_prunable, device=f'cuda', requires_grad=True)
        self._apply_overlay(overlay, torch.randn_like(overlay) * 6e-2, (0.8, 1.2))
        self._accumulate_saliency(self.inp[0], self.full_activations[0])
        self._remove_overlay(overlay, torch.randn_like(overlay) * 6e-2, (0.8, 1.2))
        magnitudes = overlay.grad.detach().to(torch.float64).abs()
        return self._get_ticket(magnitudes)

    def _running_grad_mask(self):
        overlay = torch.ones(self.mm.num_prunable, device=f'cuda', requires_grad=True)
        for idx, (x, *_) in enumerate(self.inp):
            self._apply_overlay(overlay, torch.randn_like(overlay) * 6e-2, (0.8, 1.2))
            self._accumulate_saliency(x, self.full_activations[idx])
            self._remove_overlay(overlay, torch.randn_like(overlay) * 6e-2, (0.8, 1.2))
            if self.DISTRIBUTED: dist.barrier()
        magnitudes = overlay.grad.detach().to(torch.float64).abs()
        if self.DISTRIBUTED: dist.all_reduce(magnitudes, op=dist.ReduceOp.SUM)
        return self._get_ticket(magnitudes)

    def grad_mask(self):
        
        if self.spr == 1.: return torch.ones_like(self.mm.get_buffer("MASK"))

        if not self.running: return self._single_shot_grad_mask()
        else: return self._running_grad_mask()

    def _hook(self, _, __, output): self.act_w.append(output.view(output.shape[0], -1))
    def _fhook(self, func, _, __, output): self.act_w.append(func(output).view(output.shape[0], -1))

