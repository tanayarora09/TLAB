import matplotlib.pyplot as plt
import torch

def save_individual_image(image, fp):
    image /= torch.max(image)
    image = image.permute(1, 2, 0).detach().cpu().numpy()
    image = (image * [0.2023, 0.1994, 0.2010]) + [0.4914, 0.4822, 0.4465]
    plt.imshow(image)
    plt.axis("off")
    plt.savefig(fp, bbox_inches='tight')
    plt.close()

def view_data(dataloader, rank, transforms):

    for batch, (x, y) in enumerate(dataloader):

        x, y = x.to('cuda'), y.to('cuda')

        print(rank, x.device)

        for T in transforms:
            x = T(x)

        for i in range(len(x)):
            save_individual_image(x[i], f"./logs/DATAVIEWING/{rank}_{int(batch)}_{i}.png")

        if batch % 20 == 0 and rank == 0:
            print(batch)

    return
