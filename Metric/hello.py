import torch
from torch import distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from torch.utils import benchmark
import numpy as np 
import time

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

from data import get_cifar, setup_distribute, cleanup_distribute, DataAugmentation, ResizeAndNormalize
from Helper import save_individual_image
from VGG import VGG19
from TicketModels import TicketCNN

import torch._dynamo as dynamo
dynamo.config.capture_scalar_outputs = True


from posix import fspath
torch.compiler.allow_in_graph(fspath) # temp solution to graph break in vision dataset

def temp(rank, world_size):

    torch.cuda.set_device(rank)

    setup_distribute(rank, world_size)
    
    dt, dv = get_cifar(rank, world_size) 

    dataAug = torch.jit.script(DataAugmentation())
    preprocess = torch.jit.script(ResizeAndNormalize())

    torch._print("Got Data")

    dynamo.reset()

    model = VGG19()
    
    model.cuda()

    model = DDP(model, device_ids = [rank], gradient_as_bucket_view = True)

    model = torch.compile(model, mode = "max-autotune")
    
    torch._print("Got Cuda Model")

    T = TicketCNN(model, rank)

    del model

    T.build(torch.optim.SGD(T.m.module.parameters(), 0.01, momentum = 0.9), 
                        data_augmentation_transform = dataAug,
                        preprocess_transform = preprocess, 
                        weight_decay = 1e-6)
    

    print(all([param.device == torch.cuda.current_device] for name, param in T.m.named_buffers()))
    print(all([param.device == torch.cuda.current_device] for name, param in T.m.named_parameters()))
    T.summary(128)
    
    dt = iter(dt)
    dv = iter(dv)
    start = time.time()
    print(dynamo.explain(T.forward(next(dt)[0].to('cuda'))))
    print(time.time() - start)
    """
    def timed(fn):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        end.record()
        torch.cuda.synchronize()
        return result, start.elapsed_time(end) / 1000

    torch._print("Starting Inference")

    N_ITERS = 18
    
    eager_times = []
    for i in range(N_ITERS):
        inp = next(t) if (N_ITERS % 2 == 0) else next(v)
        _, eager_time = timed(lambda: model(inp))
        eager_times.append(eager_time)
        print(f"eager eval time {i}: {eager_time}")

    print("~" * 10)

    compile_times = []
    for i in range(N_ITERS):
        inp = next(t) if (N_ITERS % 2 == 0) else next(v)
        _, compile_time = timed(lambda: model(inp))
        compile_times.append(compile_time)
        print(f"compile eval time {i}: {compile_time}")
    print("~" * 10)


    eager_med = np.median(eager_times)
    compile_med = np.median(compile_times)
    speedup = eager_med / compile_med
    assert(speedup > 1)
    print(f"(eval) eager median: {eager_med}, compile median: {compile_med}, speedup: {speedup}x")
    print("~" * 10)
    """
    """
    model.compile()
    #model = torch.compile(model)
    
    timer = benchmark.Timer(
    stmt="model.evaluate(dt, 391); model.evaluate(dt, 391); model.evaluate(dt, 391); model.evaluate(dt, 391); model.evaluate(dt, 391); model.evaluate(dt, 391); model.evaluate(dt, 391); model.evaluate(dt, 391); model.evaluate(dt, 391); model.evaluate(dt, 391); model.evaluate(dt, 391); model.evaluate(dt, 391); model.evaluate(dt, 391); model.evaluate(dt, 391); model.evaluate(dt, 391)",
    globals={"model": model, "dt": dt},
    label="Evaluation Time")

    print(timer.blocked_autorange(min_run_time = 330.0))
    """
    cleanup_distribute()

def main():
    world_size = torch.cuda.device_count()
    mp.spawn(temp, args=(world_size,), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()