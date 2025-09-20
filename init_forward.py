
import jax
from jax import random, jit, value_and_grad
import jax.numpy as jnp
import math
from typing import Callable, Dict, Tuple
from jax.typing import ArrayLike
from jax import Array
import numpy as np

from transformer import *


# for forward passing data through the transformer
def model_forward(params: Dict,
                  src: jnp.ndarray,
                  tgt_inp: jnp.ndarray,
                  dropout_rng: random.PRNGKey,
                  config: Dict,
                  src_mask: jnp.ndarray,
                  tgt_mask: jnp.ndarray,
                  deterministic: bool = False
                  ) -> jnp.ndarray:
    """
    params: dictionary of all parameters.
    src: source token ids (batch, src_seq_len)
    tgt_inp: target input token ids (batch, tgt_seq_len)
    dropout_rng: PRNG key for dropout. 
    config: dictionary with keys: d_model, dropout_rate, num_heads, etc.
    src_mask: mask for source 
    tgt_mask: mask for target
    deterministic: wether to apply dropout
    """
    # refer to embedding size, dropout rate, number of heads
    d_model = config["d_model"]
    dropout_rate = config["dropout_rate"]
    num_heads = config["h"]
    # get batch size, sequence length of source
    batch = None
    src_seq_len = None
    tgt_seq_len = None
    try:
        batch, src_seq_len = src.shape
        # get target sequence length
        _, tgt_seq_len = tgt_inp.shape
    except Exception:
        src_seq_len = src.shape
        tgt_seq_len = tgt_inp.shape



    # --- Encoder ---
    # apply embedding to source id tokens
    src_emb = embedding_forward(params["embedding_src"], src, d_model)
    # apply positional encoding
    src_emb = src_emb + params["pe_src"][None, :src_seq_len, :]
    # output for encoder, which will later be passed to decoder
    encoder_out = None
    for i in range(config["stacks"]):
        dropout_rng, attn_rng = random.split(dropout_rng)
        # pass through multihead attention
        enc_attn_out = multihead_forward(params["encoders"][i]["self_attn"],
                                     src_emb,
                                     mask=src_mask,
                                     dropout_rng=attn_rng,
                                     dropout_rate=dropout_rate,
                                     num_heads=num_heads,
                                     deterministic=deterministic)
        # apply residual connection and normalization
        enc_out = layer_norm_forward(src_emb + enc_attn_out, params["encoders"][i]["attn_norm"])
        # pass through feed forward
        dropout_rng, ff_rng = random.split(dropout_rng)
        ff_out, _ = feedforward_forward(params["encoders"][i]["ff"], enc_out, ff_rng, dropout_rate, deterministic)
        # apply residual connection and normalization
        encoder_out = layer_norm_forward(enc_out + ff_out, params["encoders"][i]["ff_norm"])


    # --- Decoder ---
    # apply embedding to target id tokens
    tgt_emb = embedding_forward(params["embedding_tgt"], tgt_inp, d_model)
    # apply positional encoding
    tgt_emb = tgt_emb + params["pe_tgt"][None, :tgt_seq_len, :] 
    # output for encoder, which will later be passed to decoder
    dec_out = None
    for i in range(config["stacks"]):
        # self attention
        dropout_rng, dec_self_attn_rng = random.split(dropout_rng)
        dec_self_attn_out = multihead_forward(params["decoders"][i]["self_attn"],
                                          tgt_emb,
                                          mask=tgt_mask,
                                          dropout_rng=dec_self_attn_rng,
                                          dropout_rate=dropout_rate,
                                          num_heads=num_heads,
                                          deterministic=deterministic)
        # dec_out = layer_norm(tgt_emb + dec_self_attn_out)
        dec_out = layer_norm_forward(tgt_emb + dec_self_attn_out, params["decoders"][i]["self_attn_norm"])

        # cross attention
        dropout_rng, cross_attn_rng = random.split(dropout_rng)
        cross_attn_out = cross_attention_forward(params["decoders"][i]["cross_attn"],
                                             dec_out,
                                             encoder_out,
                                             encoder_out,
                                             mask=src_mask, # Tror at her skal det være src_mask
                                             dropout_rng=cross_attn_rng,
                                             dropout_rate=dropout_rate,
                                             num_heads=num_heads,
                                             deterministic=deterministic)
        dec_out = layer_norm_forward(dec_out + cross_attn_out, params["decoders"][i]["cross_attn_norm"])

        # feed forward
        dropout_rng, dec_ff_rng = random.split(dropout_rng)
        ff_dec_out, _ = feedforward_forward(params["decoders"][i]["ff"], dec_out, dec_ff_rng, dropout_rate, deterministic)
        dec_out = layer_norm_forward(dec_out + ff_dec_out, params["decoders"][i]["ff_norm"])
 
    
    # --- Final Projection ---
    logits = jnp.dot(dec_out, params["final_proj"]["W"]) + params["final_proj"]["b"]
    return logits



# for passing through decoder-only layers
def decoder_only_forward(
                        params: Dict,
                        input: jnp.ndarray,
                        mask: jnp.ndarray, # TODO: vet ikke om skal være dne datatypen...
                        dropout_rng: random.PRNGKey,
                        config: Dict,
                        deterministic: bool = False
                        ) -> jnp.ndarray:
    """
    params: dictionary of weights for the layers
    input: input data - token ids
    config: dictionary with keys: d_model, dropout_rate, num_heads, etc. 
    vocab_size: vocabulary size 
    seq_len: longest sequence 
    """
    # refer to embedding size, dropout rate, number of heads
    d_model = config["d_model"]
    dropout_rate = config["dropout_rate"]
    num_heads = config["h"]
    seq_len = config["seq"]
    # refer to longest sequence length
    # _, seq_len = input.shape

    # apply embedding to source id tokens
    emb = embedding_forward(params["embedding"], input, d_model)
    # apply positional encoding
    emb = emb + params["pe"][None, :seq_len, :]
    # forward pass through stacked layers
    decoder_data = emb
    for i in range( config["stacks"] ):
        # pass through multihead attention
        dropout_rng, dec_self_attn_rng = random.split(dropout_rng)
        dec_self_attn_out = multihead_forward(params["decoders"][i]["self_attn"],
                                              decoder_data,
                                              dropout_rng=dec_self_attn_rng,
                                              dropout_rate=dropout_rate,
                                              num_heads=num_heads,
                                              deterministic=deterministic,
                                              mask=mask # TODO puttet inn maske
                                              )
        # apply residual connection and normalization
        dec_out = layer_norm(decoder_data + dec_self_attn_out)
        # pass through feed forward
        dropout_rng, dec_ff_rng = random.split(dropout_rng)
        ff_dec_out, _ = feedforward_forward(params["decoders"][i]["ff"], dec_out, dec_ff_rng, dropout_rate, deterministic) 
        # apply residual connection and normalization
        dec_out = layer_norm(dec_out + ff_dec_out) # TODO: bruker mye dec_out...
        decoder_data = dec_out

    # --- Final Projection ---
    logits = jnp.dot(decoder_data, params["final_proj"]["W"]) + params["final_proj"]["b"]
    return logits




# for initilizing the full transformer model
def init_model_params(
        rng: random.PRNGKey, 
        config: Dict, 
        src_vocab_size: int, 
        tgt_vocab_size: int,
        seq_len: int
        ) -> Dict:
    """ 
    config: dictionary with keys: d_model, dropout_rate, num_heads, etc.
    src_vocab_size: vocabulary size of source
    tgt_vocab_size: vocabulary size of target
    seq_len: longest sequence
    """
    # refer to embedding size
    d_model = config["d_model"]
    # Initialize embeddings
    rng, embed_rng_src, embed_rng_tgt = random.split(rng, 3)
    embedding_src = init_embeddings(embed_rng_src, src_vocab_size, d_model)
    embedding_tgt = init_embeddings(embed_rng_tgt, tgt_vocab_size, d_model)
    # Compute fixed positional encodings
    pe_src = compute_positional_encoding(d_model, seq_len)
    pe_tgt = compute_positional_encoding(d_model, seq_len)
    # arrays for stacking layers
    encoders = []
    decoders = []

    # Encoder - stacking encoder layers
    for i in range(config["stacks"]):
        
        rng, enc_attn_rng, ff_enc_rng = random.split(rng, 3)
        # initilizing multihead attention layer
        enc_self_attn_params = init_multihead_params(enc_attn_rng, d_model, config["h"])
        # initilizing normalization layer
        enc_attn_norm = init_layer_norm( config["d_model"] )
        # initilizing feed forward layer
        ff_enc_params = init_feedforward_params(ff_enc_rng, d_model, config["d_ff"])
        # initilizing normalization layer
        ff_enc_norm = init_layer_norm( config["d_model"] )
        # stacking encoder layers
        encoders.append( {
            "self_attn": enc_self_attn_params,
            "attn_norm": enc_attn_norm,
            "ff": ff_enc_params,
            "ff_norm": ff_enc_norm
        } )
    
    # Decoder - stacking decoder layers
    for i in range(config["stacks"]):
        # Decoder modules
        rng, dec_self_attn_rng, cross_attn_rng, ff_dec_rng = random.split(rng, 4) 
        # initilizing multihead attention layer
        dec_self_attn_params = init_multihead_params(dec_self_attn_rng, d_model, config["h"])
        # initilizing normalization layer
        dec_self_attn_norm = init_layer_norm( config["d_model"] )
        # initilizing cross attention layer
        cross_attn_params = init_cross_attention_params(cross_attn_rng, d_model, config["h"])
        # initilizing normalization layer
        cross_attn_norm = init_layer_norm( config["d_model"] )
        # initilizing feed forward layer
        ff_dec_params = init_feedforward_params(ff_dec_rng, d_model, config["d_ff"])
        # initilizing normalization layer
        ff_dec_norm = init_layer_norm( config["d_model"] )
        # stacking decoder layers
        decoders.append( {
            "self_attn": dec_self_attn_params,
            "self_attn_norm": dec_self_attn_norm,
            "cross_attn": cross_attn_params,
            "cross_attn_norm": cross_attn_norm,
            "ff": ff_dec_params,
            "ff_norm": ff_dec_norm
        } )

    # Final projection: from d_model to tgt_vocab_size
    rng, proj_rng = random.split(rng)
    W_proj = random.normal(proj_rng, (d_model, tgt_vocab_size)) / jnp.sqrt(d_model)
    b_proj = jnp.zeros((tgt_vocab_size,))
    # stacking all layers into a final transformer model
    return {
        "embedding_src": embedding_src,
        "embedding_tgt": embedding_tgt,
        "pe_src": pe_src,
        "pe_tgt": pe_tgt,
        "encoders": encoders,
        "decoders": decoders,
        "final_proj": {"W": W_proj, "b": b_proj}
    }



# for initilizing the decoder-only
def init_decoder(
        rng: random.PRNGKey, 
        config: Dict, 
        vocab_size: int,
        seq_len: int
        ) -> Dict:
    """  
    config: dictionary with keys: d_model, dropout_rate, num_heads, etc. 
    vocab_size: vocabulary size 
    seq_len: longest sequence 
    """
    # refer to embedding size 
    d_model = config["d_model"]
    # arrays for stacking layers
    decoders = []
    # Initialize embeddings
    rng, embed_rng_tgt = random.split(rng, 2)
    embedding_tgt = init_embeddings(embed_rng_tgt, vocab_size, d_model)
    # Compute fixed positional encodings
    pe_tgt = compute_positional_encoding(d_model, seq_len)

    # Stacking decoder layers
    for i in range( config["stacks"] ):
        # initilizing multihead attention layer
        rng, dec_self_attn_rng, cross_attn_rng, ff_dec_rng = random.split(rng, 4)
        dec_self_attn_params = init_multihead_params(dec_self_attn_rng, d_model, config["h"])
        # initilizing feed forward layer
        ff_dec_params = init_feedforward_params(ff_dec_rng, d_model, config["d_ff"])
        # stacking decoder layers
        decoders.append( {
            "self_attn": dec_self_attn_params,
            "ff": ff_dec_params,
        } )

    # Final projection: from d_model to tgt_vocab_size
    rng, proj_rng = random.split(rng)
    W_proj = random.normal(proj_rng, (d_model, vocab_size)) / jnp.sqrt(d_model)
    b_proj = jnp.zeros((vocab_size,))
    # stacking all layers into a final decoder-only model
    return {
        "embedding": embedding_tgt,
        "pe": pe_tgt,
        "decoders": decoders,
        "final_proj": {"W": W_proj, "b": b_proj}
    }






