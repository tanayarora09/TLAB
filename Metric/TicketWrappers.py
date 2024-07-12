import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
import torchinfo
import time
from collections import defaultdict
import inspect
import LotteryLayers
import h5py
import math
from contextlib import contextmanager

@contextmanager
def conditional_no_grad(condition):
    if condition:
        with torch.no_grad():
            yield
    else:
        yield

class TicketCNN:

    ### INIT FUNCTIONS

    def __init__(self, model: nn.Module, rank: int):
        super(TicketCNN, self).__init__()
        self.m = model
        self.L2_NORM = None
        self._grads = None
        self._rewind = None
        self._init = None
        self._iterations = []
        self.ROOT = rank == 0
        self.RANK = rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.m(x)

    def build(self, optim: torch.optim.Optimizer, 
              data_augmentation_transform: nn.Module, 
              resize_transform: nn.Module,
              normalize_transform: nn.Module,
              evaluate_transform: nn.Module,
              loss = F.cross_entropy, weight_decay = 0.0,
              scaler: bool = True, clipnorm: float = 2.0):
        
        self.criterion = loss
        self.optim = optim

        if self.ROOT: self.reset_cpu_metrics()
        self.loss_tr = torch.as_tensor(0.0, dtype = torch.float64, device = 'cuda')
        self.acc_tr = torch.as_tensor(0.0, dtype = torch.float64, device = 'cuda')
        self.bcount = torch.as_tensor(0, dtype = torch.int64, device = 'cuda')

        self.L2_sum = torch.as_tensor(0.0, device = 'cuda')

        self.L2_NORM = weight_decay
        self.use_amp = scaler
        self.reset_scaler()
        self.clipnorm = clipnorm

        self.daug = data_augmentation_transform
        self.evalT = evaluate_transform
        self.resize = resize_transform
        self.normalize = normalize_transform

        self._grads = list(defaultdict())
        self.curr_L2 = defaultdict(list)

        self.m = self.m

    ### EVAL FUNCTIONS
    
    @torch.no_grad()
    def correct_k(self, out: torch.Tensor, y: torch.Tensor, topk: int = 1) -> torch.Tensor:
        _, out = out.topk(topk, 1, True, True)
        out = out.t()
        correct = out.eq(y.view(1, -1).expand_as(out))
        correct = correct[:topk].view(-1).float().sum(0, keepdim = True)
        return correct.squeeze()

    @torch.compile
    def test_step(self, x: torch.Tensor, y: torch.Tensor, id: str, _is_first: bool = False, _calculate_grad_dist: bool = False, epoch: int = 0) -> None:
        with conditional_no_grad((not _calculate_grad_dist)):
            
            output = self.m(x)
            loss = self.criterion(output, y) + self.L2_NORM * self.calculate_reg_loss()
            
            self.loss_tr += loss
            self.acc_tr += self.correct_k(output, y, 1).div(len(x))
            self.bcount += 1
            
            if not _calculate_grad_dist: return

            self.optim.zero_grad(True)
            loss.backward()

            if _is_first: 
                self._grads[id] = self.capture_grads()
                return
            
            for map in self._grads:
                if id in map:
                    all_weights = torch.cat([(self.m.get_parameter(name[:-5]).grad * buffer).reshape([-1]) for name, buffer in self.m.named_buffers() if name.endswith("mask")], dim = 0)
                    all_prev_weights = torch.cat([(map[id][name[:-5]].to('cuda').grad * buffer).reshape([-1]) for name, buffer in self.m.named_buffers() if name.endswith("mask")], dim = 0)
                    dist = torch.norm(all_weights - all_prev_weights, p = 2)
                    self.curr_L2[id] = dist.detach().cpu()
    
    @torch.compile
    def evaluate(self, dv: torch.utils.data.DataLoader, _is_first: bool = False, _calcualte_grad_dist: bool = False, epoch: int = 0) -> None:
        
        with conditional_no_grad((not _calcualte_grad_dist)):

            self.m.eval()
            
            self.reset_gpu_metrics()
            if self.ROOT: self.reset_cpu_metrics()
            
            if _calcualte_grad_dist: 
                self.curr_L2.clear()

            dv.sampler.set_epoch(0)

            for step, (x, y, id) in enumerate(dv):
                
                x, y = x.to('cuda'), y.to('cuda')
                
                x = self.resize(x)
                x = self.evalT(x)
                #x = self.daug(x)
                self.normalize(x)
                
                self.test_step(x, y, id[0], _is_first, _calcualte_grad_dist, epoch)
                
                self.collect_metrics()

                if self.ROOT:
                    self.transfer_metrics()

                self.reset_gpu_metrics()
            
            if _calcualte_grad_dist:
                self.sync_grads()

    
    @torch.no_grad()
    def get_eval_results(self) -> dict[str, float]:
        return {"accuracy": (self.cacc / self.cbcnt),
                "loss": (self.closs / self.cbcnt)}


    ### TRAIN FUNCTIONS

    @torch.no_grad()
    def reduce_learning_rate(self, factor: int):
        for pg in self.optim.param_groups:
            pg['lr'] /= factor
    
    @torch.compile
    def calculate_reg_loss(self):

        self.L2_sum.fill_(0.0)
        
        for name, param in self.m.named_parameters():
        
            if "norm" in name:
                continue
        
            self.L2_sum += torch.norm(param.detach(), 2)

        return self.L2_sum

    @torch.compile
    @torch.no_grad()
    def mask_grads(self):
        
        for name, mask in self.m.named_buffers():
        
            if not name.endswith("mask"): continue

            grad = self.m.get_parameter(name[:-5]).grad

            grad.mul_(mask)
    
    @torch.compile
    def train_step(self, x: torch.Tensor, y: torch.Tensor, id: str, accum: bool = True, accum_steps: int = 1) -> None:

        with torch.autocast(device_type = 'cuda', dtype = torch.float16, enabled = self.use_amp):
        
            output = self.m(x)
        
            loss = self.criterion(output, y)
        
            loss += self.L2_NORM * self.calculate_reg_loss()

            loss /= accum_steps

        if not accum:
            
            with self.m.no_sync():
                self.scaler.scale(loss).backward()#retain_graph = True)
            
            return
        
        self.scaler.scale(loss).backward()#retain_graph = True)

        self.scaler.unscale_(self.optim)
        nn.utils.clip_grad_norm_(self.m.parameters(), max_norm = self.clipnorm)
        
        self.mask_grads()

        self.scaler.step(self.optim)
        self.scaler.update()

        self.optim.zero_grad(True)

        with torch.no_grad():

            self.loss_tr += loss
            
            #print(torch.sum(torch.argmax(output.softmax(dim = 1), dim = 1).eq(y)))

            self.acc_tr += self.correct_k(output, y, 1).div(len(x))
            
            self.bcount += 1

        return

    #@torch.compile
    def train_one(self, dt: torch.utils.data.DataLoader, dv: torch.utils.data.DataLoader, epochs: int,
                  cardinality: int, name: str, accumulation_steps: int = 1, is_first: bool = False) -> defaultdict:

        logs = defaultdict(dict)
        
        self.reset_gpu_metrics()

        if self.ROOT: 
            self.save_init(name = name) # turn off if IMP
            self.reset_cpu_metrics()
        

        best_val_loss = 2**17
        
        start_time = None
        val_start_time = None

        for epoch in range(epochs):
            
            if self.ROOT: torch._print(f"\033[38;2;255;0;0m\nStarting epoch {epoch + 1}\n\033[0m")
            
            start_time = time.time() 
            
            self.m.train()
            dt.sampler.set_epoch(epoch)
            
            if ((epoch == 79) or (epoch == 119)): self.reduce_learning_rate(10)
            accum = False
            
            dist.barrier()
            
            for step, (x, y, id) in enumerate(dt): # Enumerate

                iter = int(epoch * cardinality + step + 1)
                accum = ((step + 1) % accumulation_steps == 0) or (step + 1 == cardinality)

                #print(x.shape)

                x, y = x.to('cuda'), y.to('cuda')
                x = self.daug(self.resize(x))
                self.normalize(x)

                self.train_step(x, y, id[0], accum, accumulation_steps) # Forward, Backward if accum == True

                if accum: 
                    
                    self.collect_metrics()

                    #torch._print(str(step))

                    if self.ROOT: # Log, ROOT drags process
                        
                        #torch._print(str(self.acc_tr))
                        #torch._print(str(self.bcount))

                        self.transfer_metrics()
                        #torch._print(str(self.cacc))
                        #torch._print(f"\033[38;2;255;0;0m{str(self.cbcnt)}\033[0m")

                        logs[iter] = self.get_eval_results() # Add Logs; Have to Use Python bc of Bit Overflow
                        
                        if (step + 1) % 49 == 0 or (step + 1 == cardinality): # Log
                            torch._print(f"\033[38;2;255;255;255m----  Status at {math.ceil((step + 1) / 49):.0f}/8: ----     Accuracy: {logs[iter]['accuracy']:.4f}   --  Loss: {logs[iter]['loss']:.5f} --\033[0m")
                        
                        if (iter == 500) and self.ROOT: # Save Rewind
                            self.save_rewind(name = name)
                    
                    self.reset_gpu_metrics()

            if self.ROOT: torch._print(f"\033[38;2;255;255;0mTraining stage took {(time.time() - start_time):.1f} seconds.\033[0m")

            val_start_time = time.time()
            dv.sampler.set_epoch(epoch)

            self.evaluate(dv, is_first, True)
            
            if self.ROOT:
                
                logs[(epoch + 1) * cardinality].update({('val_' + k): v for k, v in self.get_eval_results().items()}) 
                
                torch._print(f"\033[38;2;255;0;0m\n||| EPOCH {epoch + 1} |||\n\033[0m") #LOG
                
                for k, v in logs[(epoch + 1) * cardinality].items():
                    torch._print(f"\033[38;2;0;255;255m\t{k}: {v:.6f}\033[0m")            
                
                if logs[(epoch + 1) * cardinality]["val_loss"] < best_val_loss and self.ROOT:
                    best_val_loss = logs[(epoch + 1) * cardinality]["val_loss"]
                    torch._print(f"\n\033[38;2;255;0;255m -- UPDATING BEST WEIGHTS TO {epoch + 1} -- \033[0m\n")
                    self.save_ckpt(f"./WEIGHTS/best_{name}.h5")
                
                self.reset_cpu_metrics()

                torch._print(f"\033[38;2;255;255;0mValidation stage took {(time.time() - val_start_time):.1f} seconds. \nTotal for Epoch: {(time.time() - start_time):.1f} seconds.\033[0m")

            self.reset_gpu_metrics()
            
        return logs

    ### PRUNING FUNCTIONS

    def train_IMP(self, dt: torch.utils.data.DataLoader, dv: torch.utils.data.DataLoader, epochs: int, 
                  prune_rate: float, iters: int, cardinality: int, name: str):

        """
        dt : Train_data
        dv: Validation_data
        epochs: Epochs_per_train_cycle
        prune_rate: pruning rate (i.e. if you want to remove 20% of weights per iterations, prune_rate = 0.8)
        iters: iterations to prune. prune_rate ** iters == final_sparsity
        cardinality: cardinality of the train data
        name: name of experiment - will log in output files
        """

        if self.ROOT: self.save_init(name) 
        
        logs = defaultdict(dict)
        logs[0] = self.train_one(dt, dv, epochs, cardinality, name + "_0", is_first = True) # train full model first, so loop ends with train
        self.sync_grads()
        for iter in range(1, iters):
            self.prune_by_rate_mg(prune_rate, iter)
            self.export_ticket(f"{name}_{iter}")
            self.load_vars(f"rewind_{name}_0")
            self.train_one(dt, dv, epochs, cardinality, name + f"_{iter}")
        
        return logs, self._mask

    def prune_by_rate_mg(self, rate, iter):
        all_weights = torch.cat([(self.m.get_parameter(name[:-5]) * buffer).reshape([-1]) for name, buffer in self.m.named_buffers() if name.endswith("mask")], dim = 0)
        threshold = all_weights.abs_().quantile(1.0 - rate**iter)
        for name, buffer in self.m.named_buffers():
            if not name.endswith("mask"): continue
            buffer.copy_((self.m.get_parameter(name[:-5]).abs().gt(threshold)).to(torch.float32))

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

    ###SERIALIZATION FUNCTIONS

    @torch.no_grad()
    def save_rewind(self, name):
        self.save_ckpt(f"./WEIGHTS/rewind_{name}.pth.tar")
    
    @torch.no_grad()
    def save_init(self, name):
        self.save_ckpt(f"./WEIGHTS/init_{name}.pth.tar")
    
    @torch.no_grad()
    def save_ckpt(self, fp):
        if not self.ROOT: raise ValueError("Must only be called from root.")
        ckpt = {'model': self.m.module.state_dict(),
                'optim': self.optim.state_dict(),
                'scaler': self.scaler.state_dict()}
        torch.save(ckpt, fp)

    @torch.no_grad()
    def load_vars(self, name):
        ckpt = torch.load(f"./WEIGHTS/{name}.pth.tar", map_location = {'cuda:%d' % 0: 'cuda:%d' % self.RANK})
        self.m.module.load_state_dict(ckpt['model'])
        self.optim.load_state_dict(ckpt['optim'])
        self.scaler.load_state_dict(ckpt['scaler'])

    @torch.no_grad()
    def export_ticket(self, name):
        with h5py.File(f"./TICKETS/{name}.h5", 'w') as f:
            for module in self.m.children():
                if module.__class__ in (inspect.getmembers(LotteryLayers, inspect.isclass)):
                    for n, buffer in module.named_buffers:
                        if not n.endswith("mask"): continue
                        f.create_dataset(n, data = buffer.detach().cpu().numpy())
                        
    @torch.no_grad()
    def load_ticket(self, name):
        with h5py.File(f"./TICKETS/{name}.h5", 'r') as f:
            for module in self.m.children():
                for n, buffer in module.named_buffers:
                    if not n.endswith("mask"): continue
                    buffer.copy_(f[n][:])

    def summary(self, batch_size):
        torchinfo.summary(self.m.module, (batch_size, 3, 224, 224))
    
    ### Helper Init

    @torch.no_grad()
    def reset_scaler(self):
        self.scaler = torch.amp.GradScaler(device = 'cuda', enabled = self.use_amp)

    @torch.no_grad()
    def reset_cpu_metrics(self):
        self.closs = 0.0
        self.cacc = 0.0
        self.cbcnt = 0

    @torch.no_grad()
    def transfer_metrics(self):
        self.closs += self.loss_tr.detach().cpu().item()
        self.cacc += self.acc_tr.detach().cpu().item()
        self.cbcnt += self.bcount.detach().cpu().item()

    @torch.no_grad()
    def reset_gpu_metrics(self):
        self.loss_tr.fill_(0.0)
        self.acc_tr.fill_(0.0)
        self.bcount.fill_(0)
    
    @torch.no_grad()
    def collect_metrics(self):
        dist.all_reduce(self.loss_tr)
        dist.all_reduce(self.acc_tr)
        dist.all_reduce(self.bcount)