import matplotlib.pyplot as plt

def plot_logs(logs, num_epochs, name, steps=196):

    """
    Plot Logs Returned by Training Run
    Saved at PLOTS/$metric_plot_$name.jpg
    """

    epochs = range(1, num_epochs + 1)
    val_loss = [logs[epoch * steps]['val_loss'] for epoch in epochs]
    acc = [logs[epoch * steps]['accuracy'] for epoch in epochs]
    val_acc = [logs[epoch * steps]['val_accuracy'] for epoch in epochs]
    loss = [logs[epoch * steps]['loss'] for epoch in epochs]

    """
    iterations = range(1, num_epochs * steps + 1)
    learning_rate = [logs[iter]["learning_rate"] for iter in iterations]

    # Plot learning rate
    
    plt.plot(iterations, learning_rate, label="learning_rate", linewidth=0.5)
    plt.title("Learning Rate")
    plt.xlabel("Iterations")
    plt.legend()
    plt.savefig(f'./PLOTS/learning_rate_plot_{name}.jpg')
    plt.close()
    """
    
    # Plot loss
    plt.plot(epochs, loss, label='training_loss', linewidth=0.5)
    plt.plot(epochs, val_loss, label='validation_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    #plt.ylim(0, 15)
    plt.savefig(f'./PLOTS/loss_plot_{name}.jpg')
    plt.close()

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, acc, label='training_accuracy')
    plt.plot(epochs, val_acc, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylim(0, 1.0)
    plt.legend()
    plt.savefig(f'./PLOTS/accuracy_plot_{name}.jpg')
    plt.close()