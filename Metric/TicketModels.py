import torch
from torch import nn
from torch.nn import functional as F
import torchinfo
import time
from collections import defaultdict
import h5py


class TicketCNN(nn.Module):

    ### INIT FUNCTIONS

    def __init__(self):
        super(TicketCNN, self).__init__()
        self.L2_NORM = None
        self._mask_list = None
        self._rewind = None
        self._init = None
        self._iterations = []

    def forward(self):
        pass

    def compile_model(self, optim: torch.optim.Optimizer, data_augmentation_transform: nn.Module, extra_transform: nn.Module,  loss = F.cross_entropy, weight_decay = 0.0):
        self.criterion = loss
        self.loss_tr = torch.as_tensor(0.0)
        self.acc_tr = torch.as_tensor(0)
        self.optim = optim
        self.L2_NORM = weight_decay
        self.daug = data_augmentation_transform
        self.preprocess = extra_transform 

    ### TRAIN and EVAL FUNCTIONS

    def calculate_reg_loss(self):
        sum = torch.as_tensor(0.0)
        for name, param in self.named_parameters():
            if "norm" in name:
                continue
            sum += torch.norm(param)
        return sum
    
    @torch.no_grad()
    def mask_grads(self):
        print("Masking Grads")
        for name, mask in self.named_buffers():
            if not name.endswith("mask"): continue #shouldn't ever occur
            self.get_parameter(name[:-5]).grad.mul_(mask)


    def train_step(self, x: torch.Tensor, y: torch.Tensor) -> None:
        self.optim.zero_grad()
        output = self(x)
        loss = self.criterion(output, y)
        loss += self.L2_NORM * self.calculate_reg_loss()
        loss.backward()
        self.mask_grads()
        self.optim.step()
        self.loss_tr += loss
        self.acc_tr += torch.sum(torch.argmax(output.softmax(dim = 1), dim = 1).eq(y))
        return

    @torch.no_grad()
    def test_step(self, x: torch.Tensor, y: torch.Tensor) -> None:
        output = self(x)
        self.loss_tr += self.criterion(output, y)
        self.loss_tr += self.L2_NORM * self.calculate_reg_loss()
        self.acc_tr += torch.sum(torch.argmax(output.softmax(dim = 1), dim = 1).eq(y))

    def train_one(self, dt: torch.utils.data.DataLoader, dv: torch.utils.data.DataLoader, epochs: int, cardinality: int, validation_cardinality: int, name: str) -> defaultdict:

        logs = defaultdict(dict)
        self.save_init(name = name)
        best_val_loss = 2**16
        start_time = None
        val_start_time = None

        for epoch in range(epochs):
            torch._print(f"Starting epoch {epoch + 1}")
            start_time = time.time() 
            self.train()
            for step, (x, y) in enumerate(dt): # Enumerate
                iter = int(epoch * cardinality + step + 1)
                x, y = x.to('cuda'), y.to('cuda')
                x = self.daug(self.preprocess(x))
                self.train_step(x, y) # Update Step
                logs[iter] = {"loss": self.loss_tr.div(step + 1).item(), "accuracy": self.acc_tr.div(step + 1).item()} # Add Logs
                if (step + 1) % 50 == 0 or (step == 390): # Log
                    torch._print(f"---  Status at {torch.ceil((step + 1) / 50):.0f} / 8: ---     Accuracy: {logs[iter]['accuracy']:.4f}   --  Loss: {logs[iter]['loss']:.5f}")
                if (iter == 1250): # Save Rewind
                    self.save_rewind(name = name)

            self.loss_tr = torch.as_tensor(0.0) # Reset Metrics
            self.acc_tr = torch.as_tensor(0)

            torch._print(f"\033[93mTraining stage took {(time.time() - start_time):.1f} seconds.\033[0m")

            val_start_time = time.time()
            self.eval()
            for step, (x, y) in enumerate(dv): #Validation
                self.test_step(x, y)
            
            logs[(epoch + 1) * cardinality].update({"val_loss": self.loss_tr.div(validation_cardinality).item(),
                                                    "accuracy": self.acc_tr.div(validation_cardinality).item()})
            
            torch._print(f"\033[97m||| EPOCH {epoch + 1} |||\033[0m") #LOG
            for k, v in logs[(epoch + 1) * cardinality].items():
                torch._print(f"\033[96m     {k}: {v:.6f}\033[0m")
            
            if logs[(epoch + 1) * cardinality]["val_loss"] < best_val_loss:
                torch._print(f"\n\033[97m -- UPDATING BEST WEIGHTS TO {epoch + 1} -- \033[0m\n")
                self.save_vars_to_file(f"./WEIGHTS/best_{name}.h5")

            self.loss_tr = torch.as_tensor(0.0) # Reset Metrics
            self.acc_tr = torch.as_tensor(0)
            
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

    def save_rewind(self, name):
        if self._rewind: return
        self.save_vars_to_file(f"./WEIGHTS/rewind_{name}.pt")
        self._rewind = self.return_vars()

    def save_init(self, name):
        if self._init: return
        self.save_vars_to_file(f"./WEIGHTS/init_{name}.pt")
        self._init = self.return_vars()
    
    def return_vars(self):
        return [(name, weight.detach().cpu()) for (name, weight) in self.named_parameters]

    def save_vars_to_file(self, fp):
        torch.save(self.state_dict(), fp)

    @torch.no_grad()
    def load_vars(self, rewind = True, fp = None):
        if fp: 
            self.load_state_dict(torch.load(fp))
        elif rewind:
            for (name, weight) in self._rewind:
                self.get_parameter(name).copy_(weight)
        else: 
            for (name, weight) in self._init:
                self.get_parameter(name).copy_(weight)
            
    def summary(self, batch_size):
        torchinfo.summary(self, (batch_size, 3, 224, 224))