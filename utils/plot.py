import numpy as np
import pickle
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

def plot_logs_cm(logs_dict, epochs, name, steps=391, validate = False):

    """
    Plot Logs Returned by Training Run
    Saved at PLOTS/$metric_plot_$name.jpg
    """
    

    plt.figure(figsize=(8, 6))

    for sp in logs_dict.keys():
    
        print(sp)
        color, logs = logs_dict[sp]

        loss = [logs[epoch * steps]['loss'] for epoch in epochs]   
        
        # Plot loss
        plt.plot(epochs, loss, color=color, label=f'Sparsity {sp:.2f}', alpha=0.8)
        plt.title('Loss')
        plt.xlabel('Epochs')
        plt.legend()
        #plt.ylim(0, 15)

    plt.savefig(f'./loss_plot_{name}.jpg')
    plt.close()

    plt.figure(figsize=(8, 6))

    for sp in logs_dict.keys():

        print(sp)

        color, logs = logs_dict[sp]

        acc = [logs[epoch * steps]['accuracy'] for epoch in epochs]

        # Plot accuracy
        plt.figure()
        plt.plot(epochs, acc, color=color, label=f'Sparsity {sp:.2f}', alpha=0.8)
        plt.title('Accuracy')
        plt.xlabel('Epochs')
        plt.ylim(0, 1.0)
        plt.legend()
        
    plt.savefig(f'./accuracy_plot_{name}.jpg')
    plt.close()


# Generate example data
num_graphs = 13  # Number of different sparsities

color_sparsities = np.linspace(0.05, 0.95, num_graphs)  # Varying from 0.1 to 0.9

# Normalize sparsity values for colormap
norm = mcolors.Normalize(vmin=min(color_sparsities), vmax=max(color_sparsities))
cmap = cm.viridis

epochs = range(1, 79 + 1)

logs_dict = dict()

for spars in range(1, num_graphs + 1):
    sp = 100 * 0.8 ** spars 
    with open(f"./logs/PICKLES/NAIVE_MIDDLE_TRUE0_160.0,100.0,160.0,31279.0,5.0,0.055_{sp:.1f}_logs.pickle", 'rb') as f:
        logs = pickle.load(f)
    logs_dict[sp] = (cmap(norm(color_sparsities[spars - 1])), logs)

plot_logs_cm(logs_dict, epochs, "tempra")


