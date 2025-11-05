import torch
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

from data.cifar10 import *

from utils.serialization_utils import logs_to_pickle, save_tensor
from utils.training_utils import plot_logs

from training.VGG import VGG_CNN
from search.salient import DGTS
from models.vgg import VGG, BaseModel

import json
import pickle
import gc

import h5py

def make_interpolated_weights(ckpts: list, alpha: float):

    state_dict1, state_dict2 = ckpts
    interpolated_state_dict = dict()

    for key in state_dict1.keys():
        if key in state_dict2:
            interpolated_state_dict[key] = alpha * state_dict2[key] + (1 - alpha) * state_dict1[key]
        else:
            raise KeyError(f"Key {key} not found in both checkpoints")
    
    return interpolated_state_dict



@torch.no_grad()
def get_loss(rank, model, dv, transforms):

    def correct_k(output: torch.Tensor, labels: torch.Tensor, topk: int = 1) -> torch.Tensor:
        """
        Returns number of correct prediction.
        Deprecates output tensor.
        """
        with torch.no_grad():
            _, output = output.topk(topk, 1)
            output.t_()
            output.eq_(labels.view(1, -1).expand_as(output))
            return output[:topk].view(-1).to(torch.int64).sum(0)

    #loss_fn = torch.nn.CrossEntropyLoss(reduction = "sum").to('cuda')
    
    model.eval()
    loss_tr = torch.as_tensor(0, dtype = torch.float64, device = 'cuda')
    count_tr = torch.as_tensor(0, dtype = torch.int64, device = 'cuda')

    for x, y, *_ in dv:
        
        x, y = x.cuda(), y.cuda()
        for T in transforms: x = T(x)

        output = model(x)
        loss = correct_k(output, y)
        loss_tr += loss
        count_tr += y.size(dim=0)

    dist.barrier(device_ids = [rank])

    dist.all_reduce(loss_tr, op = dist.ReduceOp.SUM)
    dist.all_reduce(count_tr, op = dist.ReduceOp.SUM)

    return loss_tr.div(count_tr).detach().item()

    

"""
    def evaluate(self, test_data: torch.utils.data.DataLoader) -> None:

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
"""

def calculate_alpha_error(rank, model, dv, transforms, alpha: float, ckpts: list, mean_error: float):

    ckpt = make_interpolated_weights(ckpts, alpha)
    model.load_state_dict(ckpt)

    dist.barrier(device_ids = [rank])

    loss = get_loss(rank, model, dv, transforms)

    dist.barrier(device_ids = [rank])

    print(loss, mean_error)

    return loss

def main(rank, world_size, name: str, args: list, lock, shared_list, **kwargs):

    EXPERIMENT = int(name[-1])

    SPIDXS = [int(sp) for sp in args]

    out = dict()

    for spe in SPIDXS:
        out[spe] = dict()

        curr_name = name[:-1] + f"_{spe}_{EXPERIMENT}"

        print(curr_name)

        resize = torch.jit.script(Resize().to('cuda'))
        normalize = torch.jit.script(Normalize().to('cuda'))
        center_crop = torch.jit.script(CenterCrop().to('cuda'))

        model = VGG(depth = 19, rank = rank, world_size = world_size).to("cuda")

        model.set_ticket(model.load_ticket(curr_name))

        model = DDP(model, 
                    device_ids = [rank],
                    output_device = rank, 
                    gradient_as_bucket_view = True)
        
        _, dv = get_loaders(rank, world_size, batch_size = 512) 

        del _

        ckpt1 = {k:v.cuda() for k,v in torch.load(f"logs/WEIGHTS/final_{curr_name}_0.pt", map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}, 
                           weights_only = True)['model'].items()}
        
        ckpt2 = {k:v.cuda() for k,v in torch.load(f"logs/WEIGHTS/final_{curr_name}_1.pt", map_location = {'cuda:%d' % 0: 'cuda:%d' % rank}, 
                    weights_only = True)['model'].items()}

        model.load_state_dict(ckpt1)

        error1 = get_loss(rank, model, dv, (resize, normalize, center_crop, ))

        model.load_state_dict(ckpt2)

        error2 = get_loss(rank, model, dv, (resize, normalize, center_crop, ))

        print(curr_name, error1, error2)


        for point in range(21):

            alpha = point/20

            err = calculate_alpha_error(rank, model, dv, (resize, normalize, center_crop, ),
                                                    alpha, [ckpt1, ckpt2], (error1 + error2)/2)

            print(alpha, err)

            out[spe][alpha] = err
            
        
        if rank == 0: logs_to_pickle(out, f"interpolation_{name}", suffix = "alphas")
            
    
        


    """
    while not T._pruned:

        tmp_name = f"{name}_{T.mm.sparsity.item()*0.8:.1f}"

        logs = T.fit(dt, dv, EPOCHS, CARDINALITY, tmp_name, 
                     save = True if (T.mm.sparsity_d.item() == 1.00) else False,
                     rewind_iter = 20000,
                     verbose = False, 
                     sampler_offset = sampler_offset,
                     validate = False)
        
        T.init_capture_hooks()

        ticket, fitness = collect_and_search(dt, T)

        T.remove_handles()

        T.prune_model(ticket)
        T.fitnesses.append((T.mm.sparsity.item(), fitness))

        if T.mm.sparsity_d <= T.desired_sparsity:
            T._pruned = True

        sampler_offset += 1

        if (rank == 0):

            logs_to_pickle(logs, tmp_name)
            T.mm.export_ticket(name, entry_name = f"{T.mm.sparsity.item():.1f}")

        torch.distributed.barrier(device_ids = [rank])

        T.load_ckpt(f"{name[:-5] + "0.055"}_{80.0:.1f}", "rewind")

        T.build(sparsity_rate = 0.8, experiment_args = args[1:], type_of_exp = 2,
            optimizer = torch.optim.SGD(T.m.parameters(), 0.1, momentum = 0.9, weight_decay = 1e-3),
            loss = torch.nn.CrossEntropyLoss(reduction = "sum").to('cuda'),
            collective_transforms = (resize, normalize), train_transforms = (dataAug,),
            eval_transforms = (center_crop,), final_collective_transforms = tuple(),
            scale_loss = True, gradient_clipnorm = 2.0)

        torch.distributed.barrier(device_ids = [rank])

    T.mm.export_ticket(name)

    logs_final = T.fit(dt, dv, EPOCHS, CARDINALITY, name, 
                     save = False, verbose = False,
                     sampler_offset = sampler_offset)

    if (rank == 0):
        
        plot_logs(logs_final, EPOCHS, name, steps = CARDINALITY, start = 0)
        
        logs_to_pickle(logs_final, name)

    torch.distributed.barrier(device_ids = [rank])"""