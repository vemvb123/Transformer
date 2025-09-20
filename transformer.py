import jax
from jax import random, jit, value_and_grad
import jax.numpy as jnp
import math
from typing import Callable, Dict, Tuple
from jax.typing import ArrayLike
from jax import Array
import numpy as np

# from JAX documentation
# https://docs.jax.dev/en/latest/jax.typing.html
# Used to check if variables are from jax typing, and to convert to jax typing.
def type_correction(x: ArrayLike) -> Array:
    # Runtime type validation, Python 3.10 or newer:
    if not isinstance(x, ArrayLike):
        raise TypeError(f"Expected arraylike input; got {x}")

    # Runtime type validation, any Python version:
    if not (isinstance(x, (np.ndarray, Array)) or np.isscalar(x)):
        raise TypeError(f"Expected arraylike input; got {x}")

    # Convert input to jax.Array:
    x_arr = jnp.asarray(x)

    # return an Array
    return x_arr



# dropout layer
# used to apply dropout to the data
def dropout(
        x: jnp.ndarray, rate: 
        float, rng: random.PRNGKey, 
        deterministic: bool = False
        ) -> jnp.ndarray:
    
    # converting to correct type
    x_type = type_correction(x)
    
    # checking wether to apply dropout
    if deterministic or rate == 0.0:
        return x

    # probability of not dropping a value in the x
    keep_prob = 1.0 - rate
    # making a mask to use for dropout
    mask = random.bernoulli(rng, p=keep_prob, shape=x.shape)
    # applying dropout by converting values to 0.0
    return jnp.where(mask, x_type / keep_prob, 0.0)



# initilization of embedding layer
def init_embeddings(
        rng: random.PRNGKey, 
        vocab_size: int, 
        d_model: int
        ) -> jnp.ndarray:
    
    # matrix (vocab_size, d_model)
    a = random.normal(rng, (vocab_size, d_model))
    return a

# forward passing x through embedding layer
def embedding_forward(
        embedding_matrix: jnp.ndarray, 
        x: jnp.ndarray, 
        d_model: int
        ) -> jnp.ndarray:
    
    # type correction
    x_type = type_correction(x)

    # x: (batch, seq) and output: (batch, seq, d_model)
    return embedding_matrix[x_type] * jnp.sqrt(d_model)



# initlizing positional encoding layer
def compute_positional_encoding(
        d_model: int, 
        seq_len: int
        ) -> jnp.ndarray:
    # initilizing vector for positions
    # vector [0 ,1, 2 ... seq_len] 
    position = jnp.arange(seq_len)[:, None]
    # computing division term
    div_term = jnp.exp(jnp.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    # pe matrix of shape (sequence length, embedding size)
    pe = jnp.zeros((seq_len, d_model))
    # applying positional encoding to positions
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
    return pe

# applying positional encoding to data
def positional_encoding_forward(
        x: jnp.ndarray, 
        pe: jnp.ndarray, 
        dropout_rng: random.PRNGKey, 
        dropout_rate: float, 
        deterministic: bool = False
        ) -> jnp.ndarray:
    # type correction
    x_type = type_correction(x)
    pe_type = type_correction(pe)

    # x: (batch, seq_len, d_model); pe: (seq_len, d_model)
    x_type = x_type + pe_type[None, :x_type.shape[1], :]
    # applying dropout to layer
    return dropout(x_type, rate=dropout_rate, rng=dropout_rng, deterministic=deterministic)



# -----------------------
# Layer Normalization (stateless)
# -----------------------
"""
def layer_norm(x: jnp.ndarray, eps: float = 1e-6) -> jnp.ndarray:
    x_type = type_correction(x)
    mean = jnp.mean(x_type, axis=-1, keepdims=True)
    variance = jnp.mean((x_type - mean) ** 2, axis=-1, keepdims=True)
    return (x_type - mean) / jnp.sqrt(variance + eps)

"""

# initilizing normalization layer
def init_layer_norm(features_d: int):
    # array of with (embedding size) amount of values 
    a = jnp.ones((features_d,), dtype=jnp.float32)
    b = jnp.zeros((features_d,), dtype=jnp.float32)
    return {"a": a, "b": b}

# passing data through normalization layer
def layer_norm_forward(
               x: jnp.ndarray, 
               params: Dict[str, jnp.ndarray],
               eps: float = 1e-6) -> jnp.ndarray:
    # refering to normalization weights
    a, b = params["a"], params["b"]
    # type correction
    x_type = type_correction(x)
    # terms for normalization
    mean = jnp.mean(x_type, axis=-1, keepdims=True)
    variance = jnp.mean((x_type - mean) ** 2, axis=-1, keepdims=True)
    # applying normalization
    return a * (x_type - mean) / jnp.sqrt(variance + eps) + b





# initilizing feed forward layer
def init_feedforward_params(
        rng: random.PRNGKey, 
        d_model: int, 
        d_ff: int
        ) -> Dict[str, jnp.ndarray]:
    
    rng, w1_rng, w2_rng = random.split(rng, 3)
    # weight matrix (embedding size, feed forward layer size)
    W1 = random.normal(w1_rng, (d_model, d_ff)) * math.sqrt(2. / d_model)
    # bias vector of feed forward size
    b1 = jnp.zeros((d_ff,))
    # weight matrix (feed forward layer size, embedding size)
    W2 = random.normal(w2_rng, (d_ff, d_model)) * math.sqrt(2. / d_ff)
    # bias vector of embedding size
    b2 = jnp.zeros((d_model,))
    return {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}


# passing data through feed forward layer
def feedforward_forward(params: Dict[str, jnp.ndarray],
                        x: jnp.ndarray,
                        dropout_rng: random.PRNGKey,
                        dropout_rate: float,
                        deterministic: bool = False) -> Tuple[jnp.ndarray, random.PRNGKey]:
    # type correction
    x_type = type_correction(x)
    # refering to weights
    W1, b1, W2, b2 = params["W1"], params["b1"], params["W2"], params["b2"]
    # applying weights, and ReLU activation function
    x_ff = jax.nn.relu(jnp.dot(x_type, W1) + b1)
    # applying dropout
    dropout_rng, new_rng = random.split(dropout_rng)
    x_ff = dropout(x_ff, rate=dropout_rate, rng=dropout_rng, deterministic=deterministic)
    # applying weights
    out = jnp.dot(x_ff, W2) + b2
    return out, new_rng


# initilizing multihead attention layer
def init_multihead_params(
        rng: random.PRNGKey, 
        d_model: int, 
        num_heads: int
        ) -> Dict[str, jnp.ndarray]:
    # ensuring the amount of heads must be divisible by the number of specified heads
    assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
    rng, q_rng, k_rng, v_rng, out_rng = random.split(rng, 5)
    # initilizing weight matrices for query, key and values (embedding size, embedding size)
    Wq = random.normal(q_rng, (d_model, d_model)) / math.sqrt(d_model)
    Wk = random.normal(k_rng, (d_model, d_model)) / math.sqrt(d_model)
    Wv = random.normal(v_rng, (d_model, d_model)) / math.sqrt(d_model)
    Wo = random.normal(out_rng, (d_model, d_model)) / math.sqrt(d_model)
    return {'Wq': Wq, 'Wk': Wk, 'Wv': Wv, 'Wo': Wo}


# passing data through multihead attention layer
def multihead_forward(params: Dict[str, jnp.ndarray],
                      x: jnp.ndarray,
                      mask: jnp.ndarray,
                      dropout_rng: random.PRNGKey,
                      dropout_rate: float,
                      num_heads: int,
                      deterministic: bool = False) -> jnp.ndarray:
    # type correction
    x_type = type_correction(x)

    # getting embedding size    
    d_model = x.shape[-1]
    # calculating size of each head
    d_k = d_model // num_heads
    
    # refering to weight matrices
    Wq, Wk, Wv, Wo = params["Wq"], params["Wk"], params["Wv"], params["Wo"]
    # refering to batch size and sequence length
    batch_size, seq_len, _ = x_type.shape
    # applying weight matrices to input data
    Q = jnp.dot(x_type, Wq)
    K = jnp.dot(x_type, Wk)
    V = jnp.dot(x_type, Wv)

    # function for reshaping tensor, and splitting across heads
    def split_heads(tensor):
        # tensor: (batch size, sequence length, number of heads, d_k)
        tensor = tensor.reshape(batch_size, seq_len, num_heads, d_k)
        # reshapes into: (batch size, number of heads, sequence length, d_k)
        return tensor.transpose(0, 2, 1, 3)
    # splits Q, K, V across the heads
    Q, K, V = split_heads(Q), split_heads(K), split_heads(V)
    # calculates attention scores, (batch size, number of heads, sequence length, sequence length)
    scores = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) / jnp.sqrt(d_k)
    # applies mask over scores
    if mask is not None:
        mask_type = type_correction(mask)
        scores = jnp.where(mask_type == 0, -1e9, scores)
    # scaling scores to probabilities
    attn_weights = jax.nn.softmax(scores, axis=-1)
    # applies dropout to scores
    attn_weights = dropout(attn_weights, rate=dropout_rate, rng=dropout_rng, deterministic=deterministic)
    # calculates V across attention weights
    attn = jnp.matmul(attn_weights, V)
    # reshapes attention scores (batch size, number of heads, sequence length, d_k),
    # and merges the heads
    attn = attn.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, d_model)
    # projecting outputs into a single tensor
    return jnp.dot(attn, Wo)



# initilizing cross attention layer
def init_cross_attention_params(rng: random.PRNGKey, d_model: int, num_heads: int) -> Dict[str, jnp.ndarray]:
    # initilizing cross attention weights
    return init_multihead_params(rng, d_model, num_heads)


# forward passing input through cross attention layer
def cross_attention_forward(params: Dict[str, jnp.ndarray],
                            query: jnp.ndarray,
                            key: jnp.ndarray,
                            value: jnp.ndarray,
                            mask: jnp.ndarray,
                            dropout_rng: random.PRNGKey,
                            dropout_rate: float,
                            num_heads: int,
                            deterministic: bool = False) -> jnp.ndarray:
    # type correction for the query, key, and value
    query_type = type_correction(query)
    key_type = type_correction(key)
    value_type = type_correction(value)

    # getting size of embedding
    d_model = query_type.shape[-1]
    # size of head head
    d_k = d_model // num_heads
    Wq, Wk, Wv, Wo = params["Wq"], params["Wk"], params["Wv"], params["Wo"]
    # get size of batch, sequence size for decoder
    batch_size, tgt_seq_len, _ = query_type.shape
    # get size of sequence length for encoder
    src_seq_len = key_type.shape[1]
    # applying weight matrices to query, key, value
    Q = jnp.dot(query_type, Wq)
    K = jnp.dot(key_type, Wk)
    V = jnp.dot(value_type, Wv)
    # function for reshaping tensor, and splitting across heads
    def split_heads(tensor, seq_len):
        # tensor: (batch, sequence, heads, d_k)
        tensor = tensor.reshape(batch_size, seq_len, num_heads, d_k)
        # reshapes into: (batch, heads, sequence, d_k)
        return tensor.transpose(0, 2, 1, 3)
    # splits Q, K, V across the heads
    Q = split_heads(Q, tgt_seq_len)
    K = split_heads(K, src_seq_len)
    V = split_heads(V, src_seq_len)
    # calculates attention scores, (batch, heads, sequence, sequence length)
    scores = jnp.matmul(Q, K.transpose(0, 1, 3, 2)) / jnp.sqrt(d_k)
    # applies mask over scores
    if mask is not None:
        mask_type = type_correction(mask)
        scores = jnp.where(mask_type == 0, -1e9, scores)
    # scaling scores to probabilities
    attn_weights = jax.nn.softmax(scores, axis=-1)
    # applies dropout to scores
    attn_weights = dropout(attn_weights, rate=dropout_rate, rng=dropout_rng, deterministic=deterministic)
    # calculates V across attention weights
    attn = jnp.matmul(attn_weights, V)
    # reshapes attention scores (batch, heads, decoder sequence, d_k),
    # and merges the heads
    attn = attn.transpose(0, 2, 1, 3).reshape(batch_size, tgt_seq_len, d_model)
    # projecting outputs into a single tensor
    return jnp.dot(attn, Wo)




def compute_positional_encoding(d_model, seq_len):
    position = jnp.arange(seq_len)[:, None]
    div_term = jnp.exp(jnp.arange(0, d_model, 2) * (-jnp.log(10000.0) / d_model))
    pe = jnp.zeros((seq_len, d_model))
    pe = pe.at[:, 0::2].set(jnp.sin(position * div_term))
    pe = pe.at[:, 1::2].set(jnp.cos(position * div_term))
    return pe

