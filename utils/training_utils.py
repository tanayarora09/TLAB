import matplotlib.pyplot as plt

def plot_logs(logs, num_epochs, name, steps=391, start = 0, validate = True):

    """
    Plot Logs Returned by Training Run
    Saved at PLOTS/$metric_plot_$name.jpg
    """

    epochs = range(start + 1, num_epochs + 1)
    acc = [logs[epoch * steps]['accuracy'] for epoch in epochs]
    loss = [logs[epoch * steps]['loss'] for epoch in epochs]
    if validate:
        val_acc = [logs[epoch * steps]['val_accuracy'] for epoch in epochs]
        val_loss = [logs[epoch * steps]['val_loss'] for epoch in epochs]
    
    """
    iterations = range(1, num_epochs * steps + 1)
    learning_rate = [logs[iter]["learning_rate"] for iter in iterations]

    # Plot learning rate
    
    plt.plot(iterations, learning_rate, label="learning_rate", linewidth=0.5)
    plt.title("Learning Rate")
    plt.xlabel("Iterations")
    plt.legend()
    plt.savefig(f'./logs/PLOTS/learning_rate_plot_{name}.jpg')
    plt.close()
    """
    
    # Plot loss
    plt.plot(epochs, loss, label='training_loss', linewidth=0.5)
    if validate: plt.plot(epochs, val_loss, label='validation_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    #plt.ylim(0, 15)
    plt.savefig(f'./logs/PLOTS/loss_plot_{name}.jpg')
    plt.close()

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, acc, label='training_accuracy')
    if validate: plt.plot(epochs, val_acc, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.savefig(f'./logs/PLOTS/accuracy_plot_{name}.jpg')
    plt.close()
