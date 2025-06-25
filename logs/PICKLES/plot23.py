
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt 
from matplotlib.ticker import FuncFormatter

def format_func(value: float, tick_number: float) -> str:
    if value.is_integer(): return f"{int(value)}"
    return f"{value:.1f}"

prunes = []
prune_klds = []
all_epochs = []
all_klds = []

with open(f"logs/PICKLES/FITNESS_MONITOR_fitnesses.pickle", "rb") as f:
    divs = pickle.load(f)

    epochs, klds = zip(*divs)
    epochs = [item/391 for item in epochs]
    klds = [item/336 for item in klds]
    

plt.plot(epochs, klds, color="blue")
plt.title("Fitness With Respect to Epoch on VGG Genetic Search")
plt.xlabel("Epochs")
plt.ylabel("KL Divergence (LOG)")
plt.yscale('log')
#plt.grid(True, which="both", linestyle="--", linewidth=0.5)
#plt.ylim(y_min, y_max)

plt.savefig(f"logs/PICKLES/FitnessMonitor.png")
plt.close()
