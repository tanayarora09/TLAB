import pickle
import numpy as np
import matplotlib.pyplot as plt 


def plot_logs(logs_list, num_epochs, name, steps=391, start=0):
    """
    Plot Logs from Multiple Training Runs
    Saves plots at PLOTS/loss_plot_$name.jpg and PLOTS/accuracy_plot_$name.jpg
    """
    epochs = range(start + 1, num_epochs + 1)
    
    # Extract metrics for all logs
    loss_vals = []
    val_loss_vals = []
    acc_vals = []
    val_acc_vals = []
    
    for logs in logs_list:
        loss_vals.append([logs[epoch * steps]['loss'] for epoch in epochs])
        val_loss_vals.append([logs[epoch * steps]['val_loss'] for epoch in epochs])
        acc_vals.append([logs[epoch * steps]['accuracy'] for epoch in epochs])
        val_acc_vals.append([logs[epoch * steps]['val_accuracy'] for epoch in epochs])
    
    # Convert to NumPy arrays for easier computation
    loss_vals = np.array(loss_vals)
    val_loss_vals = np.array(val_loss_vals)
    acc_vals = np.array(acc_vals)
    val_acc_vals = np.array(val_acc_vals)
    
    # Compute mean and std
    loss_mean, loss_std = np.mean(loss_vals, axis=0), np.std(loss_vals, axis=0)
    val_loss_mean, val_loss_std = np.mean(val_loss_vals, axis=0), np.std(val_loss_vals, axis=0)
    acc_mean, acc_std = np.mean(acc_vals, axis=0), np.std(acc_vals, axis=0)
    val_acc_mean, val_acc_std = np.mean(val_acc_vals, axis=0), np.std(val_acc_vals, axis=0)
    
    # Plot loss
    plt.figure()
    plt.plot(epochs, loss_mean, label='Training Loss', linewidth=1)
    plt.fill_between(epochs, loss_mean - loss_std, loss_mean + loss_std, alpha=0.2)
    plt.plot(epochs, val_loss_mean, label='Validation Loss', linewidth=1)
    plt.fill_between(epochs, val_loss_mean - val_loss_std, val_loss_mean + val_loss_std, alpha=0.2)
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.savefig(f'./logs/PLOTS/loss_plot_{name}.jpg')
    plt.close()
    
    # Plot accuracy
    plt.figure()
    plt.plot(epochs, acc_mean, label='Training Accuracy', linewidth=1)
    plt.fill_between(epochs, acc_mean - acc_std, acc_mean + acc_std, alpha=0.2)
    plt.plot(epochs, val_acc_mean, label='Validation Accuracy', linewidth=1)
    plt.fill_between(epochs, val_acc_mean - val_acc_std, val_acc_mean + val_acc_std, alpha=0.2)
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.savefig(f'./logs/PLOTS/accuracy_plot_{name}.jpg')
    plt.close()

for sp in range(20, 24, 2):
    logs_list = list()
    for exp in range(8):
          print(sp, exp)
          with open(f"./logs/PICKLES/RANDOM_WD_LESS{exp}_{sp}.0_logs.pickle", "rb") as f:
                logs = pickle.load(f)
                if len(logs.keys()) != 0: logs_list.append(logs)
    plot_logs(logs_list, 160, f"RANDOM_HALF_{sp}.0")
