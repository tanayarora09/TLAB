
from utils.data_utils import *
from data.cifar10 import get_loaders, DataAugmentation, Resize, Normalize

def main(rank, world_size, name):

    dt, dv = get_loaders(rank, world_size, iterate = False)

    dataAug = torch.jit.script(DataAugmentation().to('cuda'))
    resize = torch.jit.script(Resize().to('cuda'))
    normalize = torch.jit.script(Normalize().to('cuda'))

    view_data(dt, rank, (resize, normalize, dataAug))