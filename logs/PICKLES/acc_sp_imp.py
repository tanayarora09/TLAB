import matplotlib
import matplotlib.scale
matplotlib.use("Agg")

import matplotlib.axes
import matplotlib.pyplot as plt 

import pickle

with open("KL_2502_logs.pickle", "rb") as f:
    logs = pickle.load(f)


sparsities = sorted([100 * (0.8 ** im) for im in logs.keys()], reverse = True)

final_accuracies = list()
best_accuracies = list()

for im in range(20):
    
    best_accuracy = 0

    for epoch in range(1, 161):
        acc = logs[im][epoch * 391]["val_accuracy"]
        if (epoch == 160): final_accuracies.append(acc * 100)
        if acc > best_accuracy: best_accuracy = acc
        
    best_accuracies.append(best_accuracy * 100)

fig, ax = plt.subplots()

# Plotting
ax.plot(sparsities, final_accuracies, label="Final", color="blue")
ax.plot(sparsities, best_accuracies, label="Best", color="orange")


# Custom logarithmic scale with base 0.8

#scale = matplotlib.scale.LogScale(ax, base = 0.8)
ax.set_xscale("log")

# Reverse x-axis
ax.invert_xaxis()

# Labels and title
plt.title("VGG")
plt.ylabel("Test Accuracy")
plt.xlabel("Sparsity")
plt.legend()
plt.savefig("tmp.png")
plt.close()