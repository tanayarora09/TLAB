
import data.cifar10 as cifar

def main(rank, world_size, name):

    dt, dv = cifar.get_loaders(rank, world_size)


    for step, (x, y, id) in enumerate(dt):

        if step < 390:
            continue

        print(f"RANK {rank} || STEP {step} || IDS: {id}")


    for step, (x, y, id) in enumerate(dt):

        if step < 390:
            continue

        print(f"RANK {rank} || STEP {step} || IDS: {id}")
