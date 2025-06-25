'''import pickle

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt 
from matplotlib.ticker import FuncFormatter
def format_func(value: float, tick_number: float) -> str:
    if value.is_integer(): return f"{int(value)}"
    return f"{value:.1f}"



"""for sp in range(1, 21):

    #for idx in range(5)

    with open(f"logs/PICKLES/ITERATIVE_DEEP0_160.0,160.0,160.0,6.0,0.0116_{(100 * 0.8**sp):.1f}_fitnesses.pickle", "rb") as f:
        logs = pickle.load(f)

    epochs, klds, sparsities = zip(*logs)
    repochs, rklds = list(), list()
    #sparsities = [item * 100 for item in sparsities]
    """"""
    ridx = 0
    first = None
    for i, item in enumerate(divs):
        if len(item) == 3:
            if first == None: first = item[0]
            divs[i] = (item[0] + ridx, item[1])
            ridx += 1 

    epochs, klds = zip(*divs)
    """"""
    klds = [item/336 for item in klds]
    curr = len(klds) - 1
    while epochs[curr-1] < epochs[curr]:
        rklds.append(klds[curr])
        repochs.append(epochs[curr])
        curr -= 1
    rklds.append(klds[curr])
    repochs.append(epochs[curr])

    
    plt.figure()
    plt.plot(repochs, rklds, label = "VGG", color = "blue")
    plt.yscale("log")
    plt.xlabel("Epochs")
    plt.ylabel("Kullback-Leibler Divergence - GTS")

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.savefig(f"logs/PICKLES/ITERATIVE0_{sp}.png")
    plt.close()


"""""""""
prunes = list()
prune_klds = list()

for sp in range(1, 21):
    with open(f"logs/PICKLES/REWIND_MULTI0_160.0,90.0,160.0,4.0,6.0,0.0116_{(100 * 0.8**sp):.1f}_fitnesses.pickle", "rb") as f:
        divs = pickle.load(f)

    epochs, klds, sparsities = zip(*divs)
    klds = [item/336 for item in klds]

    minidx = 0
    for idx in range(len(klds)):
        if klds[idx] < klds[minidx]: minidx = idx

    prunes.append(sparsities[minidx])
    prune_klds.append(klds[minidx])

    plt.plot(epochs, klds, color = "blue")
    #for prune in prunes:
        #plt.axvline(x = prune, color = 'red', linestyle = '--')
    #plt.yscale("log")
    plt.title(f"REWIND_MULTI_{(100 *0.8**sp):.1f}")
    plt.xlabel("Epochs")
    plt.ylabel("Divergence")

    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.savefig(f"logs/PICKLES/REWIND_MULTI_{sp}.png")
    plt.close()

plt.plot(prunes, prune_klds, color = "blue")
plt.xscale('log', base=0.8)
plt.yscale("log")
plt.title("REWIND_MULTI_PRUNES")
plt.xlabel("Sparsity")
plt.ylabel("Divergence")

plt.xticks(([100,] + prunes)[::4]) 

plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

plt.savefig(f"logs/PICKLES/REWIND_MULTI.png")
plt.close()'''
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
from matplotlib.ticker import FuncFormatter
import matplotlib.ticker as ticker

formatter = ticker.ScalarFormatter(useMathText=True)

def format_func(value: float, tick_number: float) -> str:
    if value.is_integer(): return f"{int(value)}"
    return f"{value:.1f}"

for rep in range(1,2):

    prunes = []
    prune_klds = []
    all_epochs = []
    all_klds = []

    # First pass to determine axis limits
    x_min, x_max = float("inf"), float("-inf")
    y_min, y_max = float("inf"), float("-inf")

    for sp in range(1, 21):
        with open(f"logs/PICKLES/REWINDA_EQUAL{rep}_160.0,90.0,160.0,10.0,5.0,0.0116_{(100 * 0.8**sp):.1f}_fitnesses.pickle", "rb") as f:
            divs = pickle.load(f)

        epochs, klds, sparsities = zip(*divs)
        epochs = [item/391 for item in epochs]
        print(epochs)
        klds = [item/336 for item in klds]
        
        x_min, x_max = min(x_min, min(epochs)), max(x_max, max(epochs))
        y_min, y_max = min(y_min, min(klds)), max(y_max, max(klds))
        
        minidx = min(range(len(klds)), key=lambda idx: klds[idx])
        prunes.append(sparsities[minidx])
        prune_klds.append(klds[minidx])
        
        all_epochs.append(epochs)
        all_klds.append(klds)

    # Second pass to plot with consistent axes
    for sp, (epochs, klds) in enumerate(zip(all_epochs, all_klds), start=1):
        plt.plot(epochs, klds, color="blue")
        plt.title(f"Density: {(100 * 0.8**sp):.1f}%")
        plt.xlabel("Epochs")
        plt.ylabel("Fitness")
        #plt.yscale('log')
        plt.grid(True, which="both", linestyle="--", linewidth=0.5)
        plt.xlim(x_min, x_max)
        plt.gca().yaxis.set_major_formatter(formatter)
        #plt.ylim(y_min, y_max)
        
        plt.savefig(f"logs/PICKLES/iterative_REWIND_{rep}_{sp:02d}.png")
        plt.close()

    # Final plot for prunes
    plt.plot(prunes, prune_klds, color="blue")
    plt.xscale('log', base=0.8)
    plt.yscale('log')
    plt.title("REWIND PRUNES")
    plt.xlabel("Sparsity")
    plt.ylabel("Divergence")
    plt.xticks(([100,] + prunes)[::4])
    plt.gca().xaxis.set_major_formatter(FuncFormatter(format_func))
    #plt.legend()
    plt.grid(True, which="both", linestyle="--", linewidth=0.5)

    plt.savefig(f"logs/PICKLES/iterative_REWIND_{rep}.png")
    plt.close()
