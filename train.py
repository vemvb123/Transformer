from functools import partial
from jax.sharding import Mesh, NamedSharding, PartitionSpec
import torch.distributed as dist
from jax import pmap, lax
from utils import save_model_weights, load_model

from metrics import *
from utils import *

import os

from data import *
from jax import random, jit, value_and_grad
import optax  # optimizer library for JAX
from tqdm import tqdm
import torch
from configs/config import get_config
from init_forward import *
from flax import jax_utils

from data_wiki import get_data, get_max_token_len, get_pad_token

import torch
import torch.distributed as dist

import time
import jax.distributed as jdist
import os

import os
import torch.distributed as dist


import time


# print various information about ax version, and devices avalible to jax
# jax.distributed.initialize(local_device_ids=range(4))
print("jax.__version__", jax.__version__)

print(os.environ.get("CUDA_VISIBLE_DEVICES"))

print("jax.device_count()", jax.device_count())
print("jax.local_device_count()", jax.local_device_count())
print("jax.devices()", jax.devices())

# jax.distributed.initialize(num_processes=num_processes)
print("jax.__version__", jax.__version__)
print("......")
jax.config.update("jax_enable_x64", True)
print(".......")
print(os.environ.get("CUDA_VISIBLE_DEVICES"))
print("-----------")
print(jax.devices(), jax.local_devices())
print("---------")
print("jax.device_count()", jax.device_count())
print("==============")
print("jax.local_device_count()", jax.local_device_count())
print("===================")
print("jax.devices()", jax.devices())
print("++++++++++++++++++++++++++")

def GPU_is_available():
    """Check available GPUs in JAX and PyTorch."""
    print("JAX version:", jax.__version__)
    print("Available JAX devices:", jax.devices())

    devices = jax.devices()
    for device in devices:
        print(device.device_kind)

    print("----")

    print("\nPyTorch CUDA Info:")
    print("Torch version:", torch.__version__)
    print("CUDA available:", torch.cuda.is_available())
    print("Total GPUs visible to PyTorch:", torch.cuda.device_count())

    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

GPU_is_available()

def jax_gpu():
    print(jax.default_backend())
    print(jax.devices)

print("JAX")
jax_gpu()


# get configuration
config = get_config()




# get dataset
train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt, pad_token = get_ds(config)

# get vocabulary size
src_vocab_size = tokenizer_src.get_vocab_size()
tgt_vocab_size = tokenizer_tgt.get_vocab_size() # 22463





# for calculating loss
def loss_fn(params, batch, dropout_rng):
    # retrive data from the batch
    data = batch["tgt"] 
    label = batch["label"]
    mask = batch["mask"]

    src = batch["src"]
    encoder_mask = batch["encoder_mask"]

    # compute logits through forward pass
    logits = model_forward(params, src, data, dropout_rng, config, encoder_mask, mask, False)
    # compute loss between logits and label
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, label).mean()
    return loss, logits


# do a train step
@jit
def train_step(params, opt_state, input_data, label, mask, src, encoder_mask, dropout_rng):
    # combine various data
    batch = {
        "tgt": input_data,
        "label": label,
        "mask": mask,

        "src": src,
        "encoder_mask": encoder_mask
    }
    # calculate loss
    (loss, logits), grads = value_and_grad(loss_fn, has_aux=True)(params, batch, dropout_rng)
    # update parameters through backpropagation
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss, logits


# initilize model parameters and optimizer
rng = random.PRNGKey(0)
params = init_model_params(rng, config, src_vocab_size, tgt_vocab_size, config["seq"])
optimizer = optax.adam(config["lr"])
opt_state = optimizer.init(params)


# print total amount of parameters
def print_parameters(params):
    amt_params = 0
    for key, value in params.items():
        if key == "encoders" or key == "decoders":
            for stack in params[key]:
                for ikey, ivalue in stack.items():
                    print(ikey)
                    for akey, avalue in ivalue.items():
                        print(f"{akey}: {avalue.shape}")
                        amt_params += jnp.size(avalue)
        elif key == "final_proj":
            print(key)
            for bkey, bvalue in params[key].items():
                print(f"{bkey}, {bvalue.shape}")
                amt_params += jnp.size(bvalue)
        else:
            print(key)
            print(f"{key}: {value.shape}")
            amt_params += jnp.size(value)
    return amt_params

print("stacks: ", config["stacks"])
print("amount of parameters: ", print_parameters(params)  )




# TRAINING
# loop through epochs
for epoch in range(config["num_epochs"]):
    # various metrics
    epoch_loss = 0.0
    ppl = 0.0
    bleu = 0.0
    num_batches = 0
    start_time = time.time()   # <-- start timing here

    # initilize batch iterator
    batch_iterator = tqdm(train_dataloader, desc=f"Epoch {epoch:02d}")
    # loop through each batch
    for batch in batch_iterator:
        rng, dropout_rng = random.split(rng)
       
        # convert to jax arrays
        batch_pmap = convert_to_jax(batch)
        
        # do a trains step - pass batch into model with a forward pass, compute loss, do backpropagation
        params, opt_state, loss, logits = train_step(
                params, opt_state, 
                batch_pmap["tgt"],
                batch_pmap["label"],
                batch_pmap["mask"],
                batch_pmap["src"],
                batch_pmap["encoder_mask"],
                dropout_rng)

        # for calculating average loss through the epoch
        epoch_loss += loss
        # compute perplexity
        ppl_t = compute_perplexity(logits, batch_pmap["label"], pad_token)
        ppl += ppl_t
        
        end_time = time.time()
        epoch_duration = end_time - start_time
        num_batches += 1

        # shows metrics during training
        batch_iterator.set_postfix({
            "loss": f"{loss:6.3f}",
            "PPL": f"{ppl_t:6.3f}",
            "avg_ppl": f"{ppl / num_batches}",
            "avg_loss": f"{epoch_loss / num_batches}",
            "epoch time": f"{epoch_duration:.2f} seconds"
        })
    
    avg_loss = epoch_loss / num_batches
    avg_ppl = ppl / num_batches







# potensielt.. TODO
def loss_fn(params, batch, dropout_rng):
    data = batch["tgt"] 
    label = batch["label"]
    mask = batch["mask"]
    
    src = batch["src"]
    decoder_mask = batch["decoder_mask"]

    logits = model_forward(params, src, data, dropout_rng, config, mask, decoder_mask, False)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, label).mean()
    return loss, logits




@jit
def simple_forward(tgt, label, mask, src, encoder_mask, dropout_rng):
    logits = model_forward(params, src, tgt, dropout_rng, config, encoder_mask, mask, False)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, label).mean()
    return loss, logits




# TESTING
print("Testing")
for epoch in range(1):
    epoch_loss = 0.0
    ppl = 0.0
    bleu = 0.0
    num_batches = 0

    start_time = time.time()   # <-- start timing here

    batch_iterator = tqdm(val_dataloader, desc=f"Epoch {epoch:02d}")
    
    for batch in batch_iterator:
        rng, dropout_rng = random.split(rng)
        
        # TODO: Hvis jeg bruker jax array, så må jeg kanskje håndtere PPL regning annerledes
        batch_pmap = convert_to_jax(batch)

        # print_shape_batch_pmap(batch_pmap)
        # print(batch_pmap["tgt"].shape)

        loss, logits = simple_forward(
                batch_pmap["tgt"],
                batch_pmap["label"],
                batch_pmap["mask"],
                batch_pmap["src"],
                batch_pmap["encoder_mask"],
                dropout_rng)

        epoch_loss += loss
       
        ppl_t = compute_perplexity(logits, batch_pmap["label"], pad_token)
        # print("done computing")
        """
        for i in range(num_devices):
            avg_ppl_t += compute_perplexity(logits[i], batch_pmap["label"][i], pad_token)
        ppl_t = avg_ppl_t / num_devices        
        """
        ppl += ppl_t

        end_time = time.time()   # <-- end timing here
        epoch_duration = end_time - start_time

        num_batches += 1
        batch_iterator.set_postfix({
            "loss": f"{loss:6.3f}",
            "PPL": f"{ppl_t:6.3f}",
            "avg_ppl": f"{ppl / num_batches}",
            "avg_loss": f"{epoch_loss / num_batches}",
            "epoch time": f"{epoch_duration:.2f} seconds"
        })
    
    avg_loss = epoch_loss / num_batches
    avg_ppl = ppl / num_batches






# saving model parameters
print("saving model params")
save_model_weights(params, "/cluster/datastore/vemundvb/tra_sist/project/params_norm_weights.pkl")

