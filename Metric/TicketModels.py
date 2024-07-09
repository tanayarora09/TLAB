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


class TicketCNN:

    ### INIT FUNCTIONS

    def __init__(self, model: nn.Module, rank: int):
        super(TicketCNN, self).__init__()
        self.m = model
        self.L2_NORM = None
        self._mask_list = None
        self._rewind = None
        self._init = None
        self._iterations = []
        self.ROOT = rank == 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.m(x)

    def build(self, optim: torch.optim.Optimizer, 
              data_augmentation_transform: nn.Module, 
              preprocess_transform: nn.Module, 
              loss = F.cross_entropy, weight_decay = 0.0,
              scaler: bool = True, clipnorm: int = 0.1):
        
        self.criterion = loss
        self.optim = optim

        self.loss_tr = torch.as_tensor(0.0, device = 'cuda')
        self.acc_tr = torch.as_tensor(0, device = 'cuda')
        self.bcount = torch.as_tensor(0, device = 'cuda')

        self.L2_NORM = weight_decay
        self.use_amp = scaler
        self.reset_scaler()
        self.clipnorm = clipnorm

        self.daug = data_augmentation_transform
        self.preprocess = preprocess_transform

        self.m = self.m

    @torch.no_grad()
    def reset_scaler(self):
        self.scaler = torch.amp.GradScaler(enabled = self.use_amp)

    @torch.no_grad()
    def reset_metrics(self):
        self.loss_tr.fill_(0.0)
        self.acc_tr.fill_(0)
        self.bcount.fill_(0)
    
    @torch.no_grad()
    def collect_metrics(self):
        dist.all_reduce(self.loss_tr)
        dist.all_reduce(self.acc_tr)
        #dist.all_reduce(self.bcount) #dont think this is necessary

    ### EVAL FUNCTIONS
    
    @torch.no_grad()
    def accuracy_k(self, out: torch.Tensor, y: torch.Tensor, topk: int = 1) -> torch.Tensor:
        _, out = out.topk(topk, 1, True, True)
        out = out.t()
        correct = out.eq(y.view(1, -1).expand_as(out))
        correct = correct[:topk].view(-1).float().sum(0, keepdim = True)
        return correct.squeeze()

    @torch.compile
    @torch.no_grad()
    def test_step(self, x: torch.Tensor, y: torch.Tensor) -> None:
        output = self.m(x)
        self.loss_tr += self.criterion(output, y)
        self.loss_tr += self.L2_NORM * self.calculate_reg_loss()
        self.acc_tr += self.accuracy_k(output, y, 1)
        self.bcount += len(y)
    
    @torch.compile
    @torch.no_grad()
    def evaluate(self, dv: torch.utils.data.DataLoader) -> dict[str, float]:
        
        self.m.eval()
        self.reset_metrics()
        dv.sampler.set_epoch(0)

        for step, (x, y) in enumerate(dv):
            if self.ROOT:
                if step % 25 == 0:
                    torch._print(str(step))
            x, y = x.to('cuda'), y.to('cuda')
            x = self.preprocess(x)
            self.test_step(x, y)
            #self.collect_metrics()

        self.collect_metrics()

        return {"val_accuracy": (self.acc_tr.div(self.bcount).detach().cpu().item()),
                "val_loss": (self.loss_tr.div(self.bcount).detach().cpu().item())}

    ### TRAIN FUNCTIONS

    def reduce_learning_rate(self, factor: int):
        for pg in self.optim.param_groups:
            pg['lr'] /= factor

    def calculate_reg_loss(self):
        sum = torch.as_tensor(0.0, device = 'cuda')
        for name, param in self.m.named_parameters():
            if "norm" in name:
                continue
            sum += torch.norm(param)
        return sum
    
    @torch.compile
    @torch.no_grad()
    def mask_grads(self):
        print("Masking Grads")
        for name, mask in self.m.named_buffers():
            if not name.endswith("mask"): continue
            self.m.get_parameter(name[:-5]).grad.mul_(mask)
            
    @torch.compile
    def train_step(self, x: torch.Tensor, y: torch.Tensor, accum: bool) -> None:

        with torch.autocast(device_type = 'cuda', dtype = torch.float16, enabled = self.use_amp):
            output = self.m(x)
            loss = self.criterion(output, y)
            loss += self.L2_NORM * self.calculate_reg_loss()

        if not accum:
            with self.m.no_sync():
                self.scaler.scale(loss).backward()
            return
        
        self.scaler.scale(loss).backward()

        self.unscale_(self.optim)
        nn.utils.clip_grad_norm_(self.m.parameters(), max_norm = self.clipnorm)
        self.mask_grads()

        self.scaler.step(self.optim)
        self.scaler.update()

        self.optim.zero_grad(True)

        with torch.no_grad():
            self.loss_tr += loss
            self.acc_tr += self.accuracy_k(output, y, 1)
            self.bcount += len(y)

        return

    #@torch.compile
    def train_one(self, dt: torch.utils.data.DataLoader, dv: torch.utils.data.DataLoader, epochs: int, cardinality: int, name: str, accumulation_steps: int = 1) -> defaultdict:

        logs = defaultdict(dict)
        if self.ROOT: self.save_init(name = name)
        best_val_loss = 2**17
        start_time = None
        val_start_time = None

        for epoch in range(epochs):
            torch._print(f"Starting epoch {epoch + 1}")
            start_time = time.time() 
            self.m.train()
            dt.sampler.set_epoch(epoch + 1)
            if ((epoch == 79) or (epoch == 119)): self.reduce_learning_rate(10)
            accum = False
            dist.barrier()
            for step, (x, y) in enumerate(dt): # Enumerate

                iter = int(epoch * cardinality + step + 1)
                accum = ((step + 1) % accumulation_steps == 0) or (step + 1 == cardinality)

                x, y = x.to('cuda'), y.to('cuda')
                x = self.daug(self.preprocess(x))
                
                self.train_step(x, y, accum) # Forward, Backward if accum == True

                if accum: 
                    
                    self.collect_metrics()

                    if self.ROOT: # Log, ROOT drags process

                        logs[iter] = {"loss": self.loss_tr.div(self.bcount).detach().cpu().item(), "accuracy": self.acc_tr.div(self.bcount).detach().cpu().item()} # Add Logs
                        
                        if (step + 1) % 98 == 0 or (step == 781): # Log
                            torch._print(f"---  Status at {torch.ceil((step + 1) / 50):.0f}/8: ---     Accuracy: {logs[iter]['accuracy']:.4f}   --  Loss: {logs[iter]['loss']:.5f}")
                        
                        if (iter == 500) and self.ROOT: # Save Rewind
                            self.save_rewind(name = name)


            torch._print(f"\033[93mTraining stage took {(time.time() - start_time):.1f} seconds.\033[0m")

            val_start_time = time.time()
            dv.sampler.set_epoch(epoch + 1)
            val_metrics = self.evaluate(dv)
            
            logs[(epoch + 1) * cardinality].update(val_metrics)
            
            torch._print(f"\033[97m||| EPOCH {epoch + 1} |||\033[0m") #LOG
            for k, v in logs[(epoch + 1) * cardinality].items():
                torch._print(f"\033[96m\t\t{k}: {v:.6f}\033[0m")
            
            if logs[(epoch + 1) * cardinality]["val_loss"] < best_val_loss and self.ROOT:
                torch._print(f"\n\033[97m -- UPDATING BEST WEIGHTS TO {epoch + 1} -- \033[0m\n")
                self.save_vars_to_file(f"./WEIGHTS/best_{name}.h5")

            self.reset_metrics()
            
            torch._print(f"\033[93mValidation stage took {(time.time() - val_start_time):.1f} seconds. \nTotal for Epoch: {(time.time() - start_time):.1f} seconds.\033[0m")
     
        return logs

    ###PRUNING FUNCTIONS

    """
        def train_IMP(self, dt, dv, epochs, prune_rate, iters, cardinality, name, strategy = None):

        """'''
        dt : Train_data
        dv: Validation_data
        epochs: Epochs_per_train_cycle
        prune_rate: pruning rate (i.e. if you want to remove 20% of weights per iterations, prune_rate = 0.8)
        iters: iterations to prune. prune_rate ** iters == final_sparsity
        cardinality: cardinality of the train data
        name: name of experiment - will log in output files
        strategy: multi-gpu strategy
        '''"""

        self.call(K.ops.zeros((1,224,224,3)), training = False) # Force build weights if not already
        self.save_init(name) 
        
        logs = defaultdict(dict)
        iter = 0
        step_logs = self.train_one(dt, dv, epochs, cardinality, name, strategy = strategy) # train full model first, so loop ends with train
        logs[iter] = step_logs
        for iter in range(1, iters):
            self.prune_by_rate_mg(prune_rate, iter)
            self.load_weights_from_obj_or_file(name, rewind = True)
            self.train_one(dt, dv, epochs, cardinality, name, strategy = strategy)
        return logs, self._mask

    def prune_by_rate_mg(self, rate, iter):
        all_weights = K.ops.concatenate([K.ops.reshape(self.mask_to_kernel(m) * m, [-1]) for m in self._mask_list], axis = 0)
        threshold = K.ops.quantile(K.ops.abs(all_weights), 1.0 - rate**iter)
        for m in self._mask_list:
            m.assign(tf.cast(K.ops.abs(self._mask_to_kernel(m)) > threshold, dtype = tf.float32))
    """


    ###SERIALIZATION FUNCTIONS

    @torch.no_grad()
    def save_rewind(self, name):
        self.save_ckpt(f"./WEIGHTS/rewind_{name}.pth.tar")
    
    @torch.no_grad()
    def save_init(self, name):
        self.save_ckpt(f"./WEIGHTS/init_{name}.pth.tar")
    
    @torch.no_grad()
    def save_ckpt(self, fp):
        ckpt = {'model': self.m.module.state_dict(),
                'optim': self.optim.state_dict(),
                'scaler': self.scaler.state_dict()}
        torch.save(ckpt, fp)

    @torch.no_grad()
    def load_vars(self, fp):
        ckpt = torch.load(fp) ### ADD MAP LOCATION
        self.m.module.load_state_dict(ckpt['model'])
        self.optim.load_state_dict(ckpt['optim'])
        self.scaler.load_state_dict(ckpt['scaler'])

    @torch.no_grad()
    def export_ticket(self, fp):
        with h5py.File(fp, 'w') as f:
            for module in self.m.children():
                if module.__class__ in (inspect.getmembers(LotteryLayers, inspect.isclass)):
                    for name, buffer in module.named_buffers:
                        if not name.endswith("mask"): continue
                        f.create_dataset(name, data = buffer.detach().cpu().numpy())
                        
    @torch.no_grad()
    def load_ticket(self, fp):
        with h5py.File(fp, 'r') as f:
            for module in self.m.children():
                for name, buffer in module.named_buffers:
                    if not name.endswith("mask"): continue
                    buffer.copy_(f[name][:])

    def summary(self, batch_size):
        torchinfo.summary(self.m.module, (batch_size, 3, 224, 224))
