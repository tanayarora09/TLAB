import torch
from torch import nn
import torch.distributed as dist

import numpy as np
import h5py

from models.LotteryLayers import Lottery

import gc


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

    def init_base(self, rank: int):

        """
        Call in __init__ after initializing layers.
        """

        self.RANK = rank

        self._prunable_count = 0

        self.migrated = False

        self.lottery_layers = tuple([layer for name, block in self.named_children() for name, layer in block.named_children() if isinstance(layer, Lottery)])

        for layer in self.lottery_layers:
            self._prunable_count += layer.MASK_NUMEL

        self.register_buffer("MASK", torch.ones(self._prunable_count, dtype = torch.bool, 
                                                device = "cuda", requires_grad = False), persistent = False)
        
        offset = 0       
        for layer in self.lottery_layers:
            layer.update_mask(self.get_buffer("MASK"), offset)
            offset += layer.MASK_NUMEL


    @torch.no_grad()
    def set_ticket(self, mask: torch.Tensor) -> None:
        
        if mask.numel() != self.num_prunable: raise ValueError("Mask must have correct number of parameters.")

        self.get_buffer("MASK").copy_(mask)

    
    def reset_ticket(self) -> None:

        self.get_buffer("MASK").fill_(True)

    @property
    def sparsity(self):
        return self.sparsity_d * 100

    @property
    def sparsity_d(self):
        if not self.migrated: return (self.get_buffer("MASK").count_nonzero() / self.num_prunable)
        total = 0
        for layer in self.lottery_layers: total += layer.weight._nnz()
        return (torch.as_tensor(total, device = "cuda") / self.num_prunable)

    @property
    def num_prunable(self): return self._prunable_count

    ### ------------------------------------- SERIALIZATION -------------------------

    @torch._dynamo.disable
    @torch.no_grad()
    def export_ticket_cpu(self) -> torch.Tensor:

        return self.get_buffer("MASK").detach().cpu()

    
    @torch._dynamo.disable
    @torch.no_grad()
    def export_ticket(self, name: str, root: int = 0, entry_name: str = "mask") -> None:
        if not (self.RANK == root): return
        with h5py.File(f"./logs/TICKETS/{name}.h5", 'a') as f:
            f.create_dataset(entry_name, data = self.export_ticket_cpu().numpy())

    @torch._dynamo.disable         
    @torch.no_grad()
    def load_ticket(self, name: str, root: int = 0, entry_name: str = "mask") -> torch.Tensor:
        """
        ROOT = 0
        """
        
        if self.RANK == root:
            with h5py.File(f"./logs/TICKETS/{name}.h5", 'r') as f:
                data = torch.as_tensor(f[entry_name][:], device = "cuda", dtype = torch.bool)
        else:
            data = torch.zeros(self.num_prunable, device = "cuda", dtype = torch.bool) 

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

    def migrate_to_sparse(self): 
        self.lottery_layers = list()

        for bname, block in self.named_children():
            replace = list()
            for name, layer in block.named_children():
                if isinstance(layer, Lottery):
                    replace.append((name, layer))
            for nm, lyr in replace:
                new_layer = lyr.migrate_to_sparse()
                self.lottery_layers.append(new_layer)
                delattr(block, nm)
                setattr(block, nm, new_layer)
                
        self.lottery_layers = tuple(self.lottery_layers)

        self.migrated = True
        delattr(self, "MASK")

        #dist.barrier(device_ids = [self.RANK])

        torch.cuda.empty_cache()
        gc.collect()

        #dist.barrier(device_ids = [self.RANK])

        return

    def prune_by_mg_rand(self, rate: float, quant_rate: float) -> torch.Tensor:

        """
        Only if ticket is dense -

        Quant_rate = % To Consider (e.x. 0.8)


        """

        if rate == 0: return

        with torch.no_grad():

            all_magnitudes = (torch.cat([(layer.get_parameter(layer.MASKED_NAME)).detach().view(-1) for layer in self.lottery_layers], 
                                        dim = 0) * self.get_buffer("MASK")).abs_()

            threshold = np.quantile(all_magnitudes.cpu().numpy(), q = (1.0 - quant_rate), method = "higher")

            consider = torch.nonzero(all_magnitudes < threshold, as_tuple=True)[0].to("cuda")

            num_to_prune = int(self.num_prunable * (1.0 - rate))

            prune_indices = torch.randperm(len(consider), device = "cuda")[:num_to_prune]

            prune = consider[prune_indices]

            self.get_buffer("MASK")[prune] = False

        return 
        


    def prune_by_mg(self, rate: float, iteration: float, root: int = 0) -> None:
        with torch.no_grad():

            if self.RANK == root:

                all_magnitudes = (torch.cat([(layer.get_parameter(layer.MASKED_NAME)).detach().view(-1) for layer in self.lottery_layers], 
                                        dim = 0) * self.get_buffer("MASK")).abs_().cpu()
        
                threshold = np.quantile(all_magnitudes.numpy(), q = 1.0 - rate ** iteration, method = "higher")

                ticket = all_magnitudes.ge(threshold).cuda()

            else: 
                ticket = torch.zeros(self.num_prunable, dtype = torch.bool, device = "cuda")

            dist.barrier(device_ids = [self.RANK])

            dist.broadcast(ticket, src = root)

            self.set_ticket(ticket)

            return 
    
    #@torch.no_grad()
    def prune_random(self, rate: float, distributed: bool, root: int = 0) -> None:
        """
        Prune 20% randomly -> rate = 0.8
        """

        if rate == 0: return

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

    def merge_tickets(self, t1: torch.Tensor, t2: torch.Tensor, t1_weight: torch.Tensor, t2_weight: torch.Tensor) -> torch.Tensor:
        """
        Merges two tickets stochastically (genetic breeding)

        Keep all with both, sample rest with weight going to better fitness

        O(N) Runtime and Space Complexity
        """
        return merge_tickets_graph(t1, t2, t1_weight, t2_weight)

    def mutate_ticket(self, ticket: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
        """
        Mutates a given ticket by randomly swapping temperature * min(true_values, false_values) values
        """
        return mutate_ticket_graph(ticket, temperature)

@torch.compile
def merge_tickets_graph(t1: torch.Tensor, t2: torch.Tensor, t1_weight: float, t2_weight: float) -> torch.Tensor:
    with torch.no_grad():
        child = t1 & t2
        remaining = (t1.sum() - child.sum()).int()
        if (remaining == 0): return child
        available_indices = torch.bitwise_xor(t1, t2).nonzero().view(-1)
        sample_probs = (t1_weight * t1.float() + t2_weight * t2.float())[available_indices]
        sample_probs /= sample_probs.sum()
        child[available_indices[torch.multinomial(sample_probs, remaining, replacement=False)]] = True
        return child


@torch.compile
def mutate_ticket_graph(ticket: torch.Tensor, temperature: torch.Tensor) -> torch.Tensor:
    with torch.no_grad():
        ticket = ticket.clone()
        ntrue = ticket.count_nonzero()
        nfalse = ticket.numel() - ntrue
        swaps = (temperature * min(ntrue, nfalse)).int()
        if swaps == 0: return ticket
        true_indices = ticket.nonzero().view(-1)
        false_indices = (~ticket).nonzero().view(-1)
        swap_true_indices = true_indices[torch.randperm(len(true_indices))[:swaps]]
        swap_false_indices = false_indices[torch.randperm(len(false_indices))[:swaps]]
        ticket[swap_true_indices] = False
        ticket[swap_false_indices] = True
        return ticket
