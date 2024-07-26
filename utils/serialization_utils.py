import pickle
import h5py
import torch

def logs_from_pickle(name):
    with open(f"./logs/PICKLES/{name}_logs.pickle", 'rb') as file:
        return pickle.load(file)

def logs_to_pickle(logs, name):
    with open(f"./logs/PICKLES/{name}_logs.pickle", 'wb') as file:
        pickle.dump(logs, file, protocol=pickle.HIGHEST_PROTOCOL)

@torch._dynamo.disable
def save_tensor(tensor, sub_directory: str, name: str, id: str):
    """
    Save tensor to tensor_swap directory under sub directory
    """
    with h5py.File(f"./tmp/swap/{sub_directory}/{name}.h5", 'a') as f:
        if id in f:
            data = f[id]
            data[...] = tensor.numpy()
        else:    
            f.create_dataset(id, data = tensor.numpy())

@torch._dynamo.disable
def read_tensor(sub_directory: str, name: str, id: str):
    """
    Read tensor from tensor_swap directory under sub_directory
    """
    with h5py.File(f"./tmp/swap/{sub_directory}/{name}.h5", 'r') as f:
        return torch.as_tensor(f[id][:])

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
