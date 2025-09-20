import matplotlib.pyplot as plt
import jax
from jax import random, jit, value_and_grad
import jax.numpy as jnp
import math
from typing import Callable, Dict, Tuple
from jax.typing import ArrayLike
from jax import Array
import numpy as np


def compute_perplexity(logits, targets, pad_token):
    device = jax.devices()[0] # Get the first GPU device
    logits = jax.device_put(logits, device)
    targets = jax.device_put(targets, device)
    pad_token = pad_token.item()
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    # Select the log probabilities of the true targets
    target_log_probs = jnp.take_along_axis(log_probs, targets[..., None],
    axis=-1).squeeze(-1)
    if pad_token is not None:
        mask = targets != pad_token # Boolean mask: True for valid tokens, 
    valid_token_count = jnp.sum(mask) # Correct way to count non-padding
    # If there are no valid tokens, return NaN or some other value 
    if valid_token_count == 0:
        return jnp.nan # Return NaN for sequences with no valid tokens
    # Compute the mean log prob only over valid tokens
    mean_log_prob = jnp.sum(target_log_probs * mask) / valid_token_count
    else:
        mean_log_prob = jnp.mean(target_log_probs) # Default: mean over all to
    perplexity = jnp.exp(-mean_log_prob) # Compute perplexity
    return perplexity



