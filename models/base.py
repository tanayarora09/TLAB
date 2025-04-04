import torch
from torch import nn
import torch.distributed as dist

import numpy as np
import h5py

from models.LotteryLayers import Lottery


class BaseModel(nn.Module):

    """
    Overhead for Models:

    Models MUST use LotteryLayers for all layers that must be pruned.

    Expected Structure:

    - Model

    -- Blocks 

    --- Conv / Linear / Norm / Activation / Other Layers that Data is Passed Through

    Expected Name of Parameters:

    $BLOCK.$LAYER.$PARAM

    """

    def init_base(self, rank: int, world_size: int):

        """
        Call in __init__ after initializing layers.
        """

        self.RANK = rank

        self.WORLD_SIZE = world_size
        self.DISTRIBUTED = world_size > 1

        self._prunable_count = 0

        self.lottery_layers = tuple([layer for name, block in self.named_children() for name, layer in block.named_children() if isinstance(layer, Lottery)])

        for layer in self.lottery_layers:
            self._prunable_count += layer.MASK_NUMEL

        self.register_buffer("MASK", torch.ones(self._prunable_count, dtype = torch.bool, 
                                                device = "cuda", requires_grad = False), persistent = False)
        
        offset = 0       
        for layer in self.lottery_layers:
            layer.update_mask(self.get_buffer("MASK"), offset)
            offset += layer.MASK_NUMEL


    
    def set_ticket(self, mask: torch.Tensor, zero_out = False, realzo = False) -> None:
        with torch.no_grad():
            if mask.numel() != self.num_prunable: raise ValueError("Mask must have correct number of parameters.")

            self.get_buffer("MASK").copy_(mask)

            #if realzo:
            #    for layer in self.lottery_layers:
            #        getattr(layer, 'weight').data *= getattr(layer, "weight_mask")

            if False: #zero_out:
                for name, module in self.named_modules():
                    if isinstance(module, nn.BatchNorm2d): 
                        module.reset_parameters()
                        module.reset_running_stats()

            if False: # zero_out: 
                for layer in self.lottery_layers: 
                    getattr(layer, "weight").mil_(getattr(layer, "weight_mask"))

#    def reset_batchnorm(self):
#        self.apply(lambda module: if isinstance(module, nn.BatchNorm2d): module.reset_parameters())
    
    def reset_ticket(self) -> None:

        self.get_buffer("MASK").fill_(True)

    @property
    def sparsity(self):
        return self.sparsity_d * 100

    @property
    def sparsity_d(self):
        return (self.get_buffer("MASK").count_nonzero() / self.num_prunable)


    @property
    def num_prunable(self): return self._prunable_count

    ### ------------------------------------- SERIALIZATION -------------------------
    
    @torch.no_grad()
    def print_layerwise_sparsities(self):
        if self.RANK != 0: return
        for bname, block in self.named_children():
            for name, layer in block.named_children():
                if isinstance(layer, Lottery):
                    mask = getattr(layer, "weight_mask")
                    print(f"{bname} | {mask.sum()/mask.numel() * 100} | {mask.numel()}")

    @torch.no_grad()
    def export_layerwise_sparsities(self):
        out = dict()
        for i, layer in enumerate(self.lottery_layers):
            mask = getattr(layer, "weight_mask")
            out[i] = (mask.sum()/mask.numel()).item()
        return out

    @torch.no_grad()
    def count_channels(self):
        if self.RANK != 0: return
        for bname, block in self.named_children():
            for name, layer in block.named_children():
                if isinstance(layer, Lottery):
                    weight = getattr(layer, "weight_mask") * getattr(layer, "weight")
                    nonzero = (weight.abs().sum(dim=(1, 2, 3)) > 0).sum()
                    print(f"{bname} | {nonzero} | {weight.shape[0]}")


    pls = print_layerwise_sparsities
    cc = count_channels

    @torch._dynamo.disable
    @torch.no_grad()
    def export_ticket_cpu(self) -> torch.Tensor:

        return self.get_buffer("MASK").detach().cpu()

    
    @torch._dynamo.disable
    @torch.no_grad()
    def export_ticket(self, name: str, root: int = 0, entry_name: str = "mask", ticket: torch.Tensor = None) -> None:
        if not (self.RANK == root): return
        with h5py.File(f"./logs/TICKETS/{name}.h5", 'a') as f:
            if ticket is not None: 
                f.create_dataset(entry_name, data = ticket.cpu().numpy())
            f.create_dataset(entry_name, data = self.export_ticket_cpu().numpy())

    @torch._dynamo.disable         
    @torch.no_grad()
    def load_ticket(self, name: str, root: int = 0, entry_name: str = "mask") -> torch.Tensor:
        """
        ROOT = 0
        """
        
        if self.RANK == root or not self.DISTRIBUTED:
            with h5py.File(f"./logs/TICKETS/{name}.h5", 'r') as f:
                data = torch.as_tensor(f[entry_name][:], device = "cuda", dtype = torch.bool)
        elif self.DISTRIBUTED:
            data = torch.zeros(self.num_prunable, device = "cuda", dtype = torch.bool) 

        if self.DISTRIBUTED:
            dist.barrier(device_ids = [self.RANK])

            dist.broadcast(data, src=root)

        return data

    @torch._dynamo.disable         
    @torch.no_grad()
    def load_ticket_to_model(self, name: str, root: int = 0, entry_name: str = "mask") -> None:
        """
        ROOT = 0
        """
        if self.RANK == root:
            with h5py.File(f"./logs/TICKETS/{name}.h5", 'r') as f:
                data = torch.as_tensor(f[entry_name][:], device = "cuda")
        else:
            data = torch.empty(self.num_prunable, device = "cuda") 

        dist.barrier(device_ids = [self.RANK])

        dist.broadcast(data, src=root)

        self.set_ticket(data)

    ### --------------------------------------- PRUNING ----------------------------

    def prune_by_mg(self, rate: float, iteration: float, root: int = 0) -> None:
        with torch.no_grad():

            if self.RANK == root or not self.DISTRIBUTED:

                all_magnitudes = (torch.cat([(layer.get_parameter(layer.MASKED_NAME)).detach().view(-1) for layer in self.lottery_layers], 
                                        dim = 0) * self.get_buffer("MASK")).abs_().cpu()
        
                threshold = np.quantile(all_magnitudes.numpy(), q = 1.0 - rate ** iteration, method = "higher")

                ticket = all_magnitudes.ge(threshold).cuda()

            elif self.DISTRIBUTED: 
                ticket = torch.zeros(self.num_prunable, dtype = torch.bool, device = "cuda")

            if self.DISTRIBUTED:

                dist.barrier(device_ids = [self.RANK])

                dist.broadcast(ticket, src = root)

            self.set_ticket(ticket)

            return 
    
    #@torch.no_grad()
    def prune_random(self, rate: float, distributed: bool, root: int = 0) -> None:
        """
        Prune 20% randomly -> rate = 0.8
        """
        with torch.no_grad():

            if not distributed or (self.RANK == root):
                
                ticket = self.get_buffer("MASK").clone()

                nonzero_indices = ticket.nonzero(as_tuple=True)[0]

                num_to_prune = int(len(nonzero_indices) * (1.0 - rate))

                prune_indices = nonzero_indices[torch.randperm(len(nonzero_indices))[:num_to_prune]]

                ticket[prune_indices] = False

            elif distributed: 
                ticket = torch.zeros_like(self.get_buffer("MASK"), device = "cuda", dtype = torch.bool)
            

            if distributed: 
                dist.barrier(device_ids = [self.RANK])
                dist.broadcast(ticket, src = root)

            self.set_ticket(ticket)
            
            return

    def prune_random_given_layerwise(self, layerwise_sparsities: dict, distributed: bool, root: int = 0):
        """
        Requires A Ticket Reset
        """
        with torch.no_grad():
            
            if not distributed or (self.RANK == root):
                ticket = list()
                self.reset_ticket()
                for i, layer in enumerate(self.lottery_layers):
                    mask = getattr(layer, "weight_mask").view(-1).clone()#.contiguous()
                    N = mask.numel()
                    prune_indices = torch.randperm(N)[: int((1.0 - layerwise_sparsities[i]) * N)]
                    mask[prune_indices] = False
                    ticket.append(mask)
                ticket = torch.cat(ticket).cuda()
            
            elif distributed: 
                ticket = torch.zeros_like(self.get_buffer("MASK"), device = 'cuda', dtype = torch.bool)

            if distributed:
                dist.barrier(device_ids = [self.RANK])
                dist.broadcast(ticket, src = root)

            self.set_ticket(ticket)

        return


    #@torch.compile
    def merge_tickets(self, t1: torch.Tensor, t2: torch.Tensor, t1_weight: float, t2_weight: float) -> torch.Tensor:
        """
        Merges two tickets stochastically (genetic breeding)

        Keep all with both, sample rest with weight going to better fitness

        O(N) Runtime and Space Complexity
        """
        return merge_tickets_graphed(t1, t2, torch.as_tensor(t1_weight), torch.as_tensor(t2_weight))

    #@torch.compile
    def mutate_ticket(self, ticket: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
        """
        Mutates a given ticket by randomly swapping temperature * min(true_values, false_values) values
        """

        return mutate_ticket_graphed(ticket, torch.as_tensor(temperature))
    

    def check_layer_collapse(self):
        """
        Not Very Helpful.
        """
        with torch.no_grad():
            not_collapse = True
            for layer in self.lottery_layers:
                not_collapse &= getattr(layer, "weight_mask").any().item()
        return not not_collapse

        
@torch.compile
def merge_tickets_graphed(t1: torch.Tensor, t2: torch.Tensor, t1w: torch.Tensor, 
                          t2w: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        child = t1 & t2
        remaining = (t1.sum() - child.sum()).int()
        if remaining == 0: return child
        available_indices = torch.bitwise_xor(t1, t2).nonzero().view(-1)
        sample = (t1w * t1.float() + t2w * t2.float())[available_indices]
        sample.div_(sample.sum())
        child[available_indices[torch.multinomial(sample, remaining, replacement = False)]] = True
        return child
        


@torch.compile
def mutate_ticket_graphed(ticket: torch.Tensor, temperature: torch.Tensor):

    with torch.no_grad():

        ticket = ticket.clone()
        ntrue = ticket.count_nonzero()
        nfalse = ticket.numel() - ntrue

        swaps = (temperature * min(ntrue, nfalse)).int()

        if swaps == 0: return ticket

        true_indices = ticket.nonzero().view(-1)
        false_indices = (~ticket).nonzero().view(-1)
        
        swap_true_indices = true_indices[torch.randperm(true_indices.numel())[:swaps]]
        swap_false_indices = false_indices[torch.randperm(false_indices.numel())[:swaps]]

        ticket[swap_true_indices] = False
        ticket[swap_false_indices] = True

        return ticket