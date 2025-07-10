import matplotlib.pyplot as plt

def plot_logs_concrete(logs, name):

    """
    Plot Logs Returned by Concrete Run
    Saved at PLOTS/$metric_concrete_$name.jpg
    """

    iterations = sorted(logs.keys())
    loss = [abs(logs[iteration]['loss']) for iteration in iterations]
    expected_sparsities = [100. - logs[iteration]['sparsity'] for iteration in iterations]
    
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
    plt.plot(iterations, loss, label='training_loss', linewidth=0.5)
    plt.title('Loss')
    plt.xlabel('Iterations')
    #plt.legend()
    plt.ylim(bottom = 0)
    plt.savefig(f'./logs/PLOTS/loss_concrete_{name}.jpg')
    plt.close()

    plt.plot(iterations, expected_sparsities, label='sparsities', linewidth=0.5)
    plt.title('Expected Sparsity')
    plt.xlabel('Iterations')
    #plt.legend()
    #splt.ylim(0, 100)
    plt.savefig(f'./logs/PLOTS/sparsities_concrete_{name}.jpg')
    plt.close()