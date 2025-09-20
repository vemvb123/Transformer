from typing import Dict
import os
import pickle
import matplotlib.pyplot as plt
import jax
from jax import random, jit, value_and_grad
import jax.numpy as jnp
import math
from typing import Callable, Dict, Tuple
from jax.typing import ArrayLike
from jax import Array
import numpy as np
import matplotlib.pyplot as plt



# for saving model weights
def save_model_weights(params: Dict, filename: str) -> bool:
    try:
        with open(filename, 'wb') as f:
            pickle.dump(params, f)
        return True
    except Exception as e:
        print(f"Error saving model weights: {e}")
        return False


# load parameters from file
def load_model(file_path: str) -> Dict:
    with open(file_path, 'rb') as f:
        loaded_weights = pickle.load(f)

    return loaded_weights




# plot some metric over time
def plot_over_time(list_of_items, filename, label, title, xlabel, ylabel):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(list_of_items)), list_of_items, label=label, color='b', linestyle='-', marker='o')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.legend(loc='upper right')

    plt.savefig(filename)




# get devices avalible to jax
def detemine_device():
    # Define the device
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.has_mps or torch.backends.mps.is_available() else "cpu"
    print("Using device:", device)
    if (device == 'cuda'):
        print(f"Device name: {torch.cuda.get_device_name(device.index)}")
        print(f"Device memory: {torch.cuda.get_device_properties(device.index).total_memory / 1024 ** 3} GB")
    elif (device == 'mps'):
        print(f"Device name: <mps>")
    else:
        print("NOTE: If you have a GPU, consider using it for training.")
        print("      On a Windows machine with NVidia GPU, check this video: https://www.youtube.com/watch?v=GMSjDTU8Zlc")
        print("      On a Mac machine, run: pip3 install --pre torch torchvision torchaudio torchtext --index-url https://download.pytorch.org/whl/nightly/cpu")
    device = torch.device(device)
    return device



# convert data to numpy
def convert_to_numpy(batch):
    return {
            "src": batch["encoder_input"].numpy(),  # Using correct key
            "tgt": batch["decoder_input"].numpy(),  # Using correct key
            "label": batch["label"].numpy(),
            "encoder_mask": batch["encoder_mask"].numpy(),
            "decoder_mask": batch["decoder_mask"].numpy()
        }


# convert data to jax
def convert_to_jax(batch):
    return {
        "tgt": jnp.array(batch["decoder_input"].cpu().numpy()),  # Convert to JAX array
        "label": jnp.array(batch["label"].cpu().numpy()),        # Convert to JAX array
        "mask": jnp.array(batch["decoder_mask"].cpu().numpy()),  # Convert to JAX array
        "src": jnp.array(batch["encoder_input"].cpu().numpy()),
        "encoder_mask": jnp.array(batch["encoder_mask"].cpu().numpy())
    }






