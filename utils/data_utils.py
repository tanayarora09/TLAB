import matplotlib.pyplot as plt
import torch
import numpy as np

from typing import List

def save_individual_image(image, fp):
    image = image.permute(1, 2, 0).detach().cpu().numpy()
    image = (image * [0.2023, 0.1994, 0.2010]) + [0.4914, 0.4822, 0.4465]
    image /= np.abs(image).max()
    plt.imshow(image)
    plt.axis("off")
    plt.savefig(fp, bbox_inches='tight')
    plt.close()

def view_data(dataloader, rank, transforms):

    for batch, (x, y, *_) in enumerate(dataloader):

        x, y = x.to('cuda'), y.to('cuda')


        for T in transforms:
            x = T(x)

        for i in range(len(x)):
            save_individual_image(x[i], f"./logs/DATAVIEWING/{rank}_{int(batch)}_{i}.jpg")

        if batch % 20 == 0 and rank == 0:
            print(batch)

    return

def jitToList1D(x: torch.Tensor) -> List[int]:
    result: List[int] = []
    for i in x:
        result.append(i.item())
    return result

@torch.jit.script
def jitToList2D(x: torch.Tensor) -> List[List[int]]:
    result: List[List[int]] = []
    for i in x:
        result.append(jitToList1D(i))
    return result