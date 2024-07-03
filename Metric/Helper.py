import matplotlib.pyplot as plt  
import matplotlib
matplotlib.use('Agg')
import numpy as np
import pickle
import h5py


def create_optimizer_args(trial):
    kwargs = {}
    kwargs["init_lr"] = trial.suggest_float("sgd_init_learning_rate", 5e-3, 8e-2, log = True)
    kwargs["momentum"] = trial.suggest_float("sgd_momentum", 25e-2, 99e-2, log = True)
    kwargs["weight_decay"] = trial.suggest_float("sgd_weight_decay", 1e-6, 5e-4, log = True)
    kwargs["crop_size"] = trial.suggest_int("crop_size", 150, 210, log = True)
    kwargs["num_hidden"] = trial.suggest_int("hidden_layers", 0, 5, step = 1)
    for i in range(kwargs["num_hidden"]):
        kwargs[f"hidden_{i}"] = trial.suggest_int(f"hidden_size_{i}", 128, 4096, log = True)
    kwargs["drop"] = trial.suggest_float("dropout_rate", 5e-2, 5e-1, log = True)
    kwargs["L2_Reg"] = trial.suggest_float("L2_rate", 5e-5, 1e-3, log = True)
    return kwargs


def plot_loss_curves(history):
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']

    epochs = range(len(history.history['loss']))

    # Plot loss
    plt.plot(epochs, loss, label='training_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()

    # Plot accuracy
    plt.figure()
    plt.plot(epochs, accuracy, label='training_accuracy')
    plt.plot(epochs, val_accuracy, label='val_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()

    plt.show()

def plot_logs(logs, num_epochs, name, steps=391): # SAVES TO FILE
    epochs = range(1, num_epochs + 1)
    val_loss = [logs[str(epoch * steps)]['val_loss'] for epoch in epochs]
    acc = [logs[str(epoch * steps)]['accuracy'] for epoch in epochs]
    val_acc = [logs[str(epoch * steps)]['val_accuracy'] for epoch in epochs]
    loss = [logs[str(epoch * steps)]['loss'] for epoch in epochs]

    iterations = range(1, num_epochs * steps + 1)
    learning_rate = [logs[str(iter)]["learning_rate"] for iter in iterations]

    # Plot learning rate
    plt.plot(iterations, learning_rate, label="learning_rate", linewidth=0.5)
    plt.title("Learning Rate")
    plt.xlabel("Iterations")
    plt.legend()
    plt.savefig(f'./PLOTS/learning_rate_plot_{name}.jpg')
    plt.close()

    # Plot loss
    plt.plot(epochs, loss, label='training_loss', linewidth=0.5)
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    plt.ylim(0, 15)
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

def save_images_from_dataset(dt, batches):
    def save_individual_image(image, fp):
        image = image.permute(1, 2, 0).cpu().numpy()
        image = (image * [0.2023, 0.1994, 0.2010]) + [0.4914, 0.4822, 0.4465]
        image = np.clip(image, 0, 1)
        plt.imshow(image)
        plt.axis("off")
        plt.savefig(fp, bbox_inches='tight')
        plt.close()
    for step, (images, labels) in enumerate(dt):
        if step < batches:
            for i in range(len(images)):
                save_individual_image(images[i], f"./data_viewing/{i}.jpg")
        else:
            break

def logs_from_pickle(name):
    with open(f"./PICKLES/{name}_logs.pickle", 'rb') as file:
        return pickle.load(file)

def logs_to_pickle(logs, name):
    with open(f"./PICKLES/{name}_logs.pickle", 'wb') as file:
        pickle.dump(logs, file, protocol=pickle.HIGHEST_PROTOCOL)

def explore_h5(fp):
    def print_structure(name, obj):
        if isinstance(obj, h5py.Group):
            print(f"Group: {name}")
            for key, val in obj.attrs.items():
                print(f" Attribute: {key} -> {val}")
        elif isinstance(obj, h5py.Dataset):
            print(f"Dataset: {name}, shape: {obj.shape}, dtype: {obj.dtype}")
            for key, val in obj.attrs.items():
                print(f" Attribute: {key} -> {val}")
    with h5py.File(fp, 'r') as f:
        f.visititems(print_structure)

"""def get_mnist():
    preprocess = lambda image, label: (tf.cast(image, tf.float32), tf.one_hot(label, 10))    
    dt, dv = tfds.load('mnist', split=['train','test'], as_supervised = True, shuffle_files=True)
    dt = dt.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).shuffle(1024).batch(32).prefetch(tf.data.AUTOTUNE)
    dv = dv.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE).shuffle(1024).batch(32).prefetch(tf.data.AUTOTUNE)
    return dt, dv
"""