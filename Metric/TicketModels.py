import torch
from torch import nn
from torch.nn import functional as F
import torch.distributed as dist
import torchinfo
import time
from collections import defaultdict


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
        self.RANK = rank

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.m(x)

    def build(self, optim: torch.optim.Optimizer, data_augmentation_transform: nn.Module, preprocess_transform: nn.Module, loss = F.cross_entropy, weight_decay = 0.0):
        self.criterion = loss
        self.loss_tr = torch.as_tensor(0.0, device = 'cuda')
        self.acc_tr = torch.as_tensor(0, device = 'cuda')
        self.optim = optim
        self.L2_NORM = weight_decay
        self.daug = data_augmentation_transform
        self.preprocess = preprocess_transform
        self.m = self.m
    ### EVAL FUNCTIONS
    
    @torch.compile
    @torch.no_grad()
    def test_step(self, x: torch.Tensor, y: torch.Tensor) -> None:
        output = self.m(x)
        self.loss_tr += self.criterion(output, y)
        self.loss_tr += self.L2_NORM * self.calculate_reg_loss()
        self.acc_tr += torch.sum(torch.argmax(output.softmax(dim = 1), dim = 1).eq(y))
    
    @torch.compile
    @torch.no_grad()
    def evaluate(self, dv: torch.utils.data.DataLoader, cardinality: int) -> dict[str, float]:
        
        self.m.eval()
        
        self.loss_tr.fill_(0.0)
        self.acc_tr.fill_(0)
        
        for step, (x, y) in enumerate(dv):
            x, y = x.to('cuda'), y.to('cuda')
            x = self.preprocess(x)
            self.test_step(x, y)
        
        return {"val_accuracy": (self.acc_tr.div(cardinality).detach().cpu().item()),
                "val_loss": (self.loss_tr.div(cardinality).detach().cpu().item())}

    ### TRAIN FUNCTIONS

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
    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.optim.zero_grad(True)
        output = self.m(x)
        loss = self.criterion(output, y)
        loss += self.L2_NORM * self.calculate_reg_loss()
        loss.backward()
        self.mask_grads()
        self.optim.step()
        with torch.no_grad():
            self.loss_tr += loss
            self.acc_tr += torch.sum(torch.argmax(output.softmax(dim = 1), dim = 1).eq(y))
        return

    #@torch.compile
    def train_one(self, dt: torch.utils.data.DataLoader, dv: torch.utils.data.DataLoader, epochs: int, cardinality: int, validation_cardinality: int, name: str) -> defaultdict:

        logs = defaultdict(dict)
        if self.RANK == 0:
            self.save_init(name = name)
        best_val_loss = 2**16
        start_time = None
        val_start_time = None

        for epoch in range(epochs):
            torch._print(f"Starting epoch {epoch + 1}")
            start_time = time.time() 
            self.m.train()
            dt.sampler.set_epoch(epoch + 1)
            dv.sampler.set_epoch(epoch + 1)
            dist.barrier()
            for step, (x, y) in enumerate(dt): # Enumerate
                iter = int(epoch * cardinality + step + 1)
                x, y = x.to('cuda'), y.to('cuda')
                x = self.daug(self.preprocess(x))
                self.train_step(x, y) # Update Step
                logs[iter] = {"loss": self.loss_tr.div(step + 1).detach().cpu().item(), "accuracy": self.acc_tr.div(step + 1).detach().cpu().item()} # Add Logs
                if (step + 1) % 50 == 0 or (step == 390): # Log
                    torch._print(f"---  Status at {torch.ceil((step + 1) / 50):.0f} / 8: ---     Accuracy: {logs[iter]['accuracy']:.4f}   --  Loss: {logs[iter]['loss']:.5f}")
                if (iter == 1250) and self.RANK == 0: # Save Rewind
                    self.save_rewind(name = name)

            torch._print(f"\033[93mTraining stage took {(time.time() - start_time):.1f} seconds.\033[0m")

            val_start_time = time.time()
            val_metrics = self.evaluate(dv, validation_cardinality)
            
            logs[(epoch + 1) * cardinality].update(val_metrics)
            
            torch._print(f"\033[97m||| EPOCH {epoch + 1} |||\033[0m") #LOG
            for k, v in logs[(epoch + 1) * cardinality].items():
                torch._print(f"\033[96m\t\t{k}: {v:.6f}\033[0m")
            
            if logs[(epoch + 1) * cardinality]["val_loss"] < best_val_loss and self.RANK == 0:
                torch._print(f"\n\033[97m -- UPDATING BEST WEIGHTS TO {epoch + 1} -- \033[0m\n")
                self.save_vars_to_file(f"./WEIGHTS/best_{name}.h5")

            with torch.no_grad():
                self.loss_tr.fill_(0.0)
                self.acc_tr.fill_(0)
            
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
        torch.save(self.m.module.state_dict(), f"./WEIGHTS/rewind{name}.pth.tar")

    @torch.no_grad()
    def save_init(self, name):
        torch.save(self.m.module.state_dict(), f"./WEIGHTS/init_{name}.pth.tar")
    
    @torch.no_grad()
    def load_vars(self, fp):
        self.m.module.load_state_dict(torch.load(fp))
            
    def summary(self, batch_size):
        torchinfo.summary(self.m.module, (batch_size, 3, 224, 224))