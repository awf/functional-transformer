"""
Pure-from-the-ground-up transformer, based on https://github.com/vpj/jax_transformer/blob/master/transformer.py

"""

from timer import timer

import jax
from jax import vmap
import jax.numpy as jnp

import matplotlib.pyplot as plt

import jax.experimental.host_callback

from jaxutils.Arg import Arg
from jaxutils.ParamsDict import ParamsDict

def rand(rng, f, shape, **kwargs):
    """
    Wrap jax.random.foo function to split the incoming rng, and return the new rng as well as the new RNG

    rng,vals = rand(rng, jax.random.uniform, (9,3), minval=-2.0, maxval=2.0)
    """
    rng,rng1 = jax.random.split(rng)
    return rng,f(rng1, shape, **kwargs)

def split(rng, *args):
    return jax.random.split(rng, *args)

def crossentropy(output: jnp.ndarray, target: int):
    return -jax.nn.log_softmax(output)[target]

def seq_crossentropy(output: jnp.ndarray, target: jnp.ndarray):
    loss_vmap = vmap(crossentropy, in_axes=(0, 0,))
    return loss_vmap(output, target).mean()

# Linear layer Wx + b
def Linear_init0(rng: jax.random.KeyArray, in_features: int, out_features: int):
    params = ParamsDict()
    rnd_range = 1 / in_features ** 0.5
    params.weight = jax.random.uniform(rng, (in_features, out_features),  
                                    minval=-rnd_range, maxval=rnd_range)

    params.bias = jnp.zeros((out_features,))
    return params

def linear_init_uniform(rng: jax.random.KeyArray, in_features: int, out_features: int):
    params = ParamsDict()
    rnd_range = 1 / in_features ** 0.5
    rng,params.weight = rand(rng, jax.random.uniform, (in_features, out_features),  
                                    minval=-rnd_range, maxval=rnd_range)

    params.bias = jnp.zeros((out_features,))
    return rng,params

# Layer norm
def layernorm_init_identity(shape):
    """
    Initialize an elementwise_linear layer with unit gain, zero bias
    """
    return ParamsDict(
        gain = jnp.ones(shape),
        bias = jnp.zeros(shape)
    )

def linear_with_commented_shapes(params, x: jnp.ndarray):
    return jnp.matmul(x, params.weight) + params.bias[None,:]
    #                 L x N, N x M        |- 1 x M -|
    #      |----- L x M --------------|   |- ? x M ---------|

def center_commented(x, eps:float = 1e-5):
    """
    Return (x - mean(x))/std(x)

    """
    assert len(x.shape) == 1 # Used only on vectors in this example
    x_centered = x - x.mean()
    var = (x_centered ** 2).mean()
    return x_centered / jnp.sqrt(var + eps)

# Compact primitives for 'all on one slide'

def crossentropy(output: jnp.ndarray, target: int):
    return -jax.nn.log_softmax(output)[target]

def linear(params, x: jnp.ndarray):
    return x @ params.weight + params.bias[None,:]

def elementwise_linear(params, x: jnp.ndarray):
    return params.gain[None,:] * x + params.bias[None,:]

def center(x, eps = 1e-5):
    return (x - x.mean())/(x.std() + eps)


flip_pe_coef = Arg('flip-pe', False, "Scale token embedding, not position embedding")

def transformer_init(rng: jax.random.KeyArray, n_vocab: int, d_model: int, n_layers: int, n_heads: int, d_k : int, d_ff: int, max_len = 4096):
    # Build config struct for call
    config = ParamsDict()
    config.d_k = d_k
    config.heads = n_heads
    if flip_pe_coef():
        config.lambda_e = d_model ** -0.5
        config.lambda_pe = 1.0
    else:
        config.lambda_e = d_model ** -0.5
        config.lambda_pe = 1.0
    config.tau = 1 / d_k ** 0.5

    # Build initializers for params
    params = ParamsDict()

    # Create embedding layer
    rng,params.embeddings = rand(rng, jax.random.normal, (n_vocab, d_model))

    # Positional encodings initialized to zeros
    params.positional_encodings = jnp.zeros((max_len, d_model))

    # For transformer layers
    params.layers = []
    for _ in range(n_layers):
        layer = ParamsDict()
        layer.norm_self_attn = layernorm_init_identity(d_model)

        layer.heads = []
        for _ in range(n_heads):
            head = ParamsDict()
            rng,head.query = linear_init_uniform(rng, d_model, d_k)
            rng,head.key = linear_init_uniform(rng, d_model, d_k)
            rng,head.value = linear_init_uniform(rng, d_model, d_model)
            
            layer.heads.append(head)

        layer.norm_ff = layernorm_init_identity(d_model)

        rng,layer.ffn1 = linear_init_uniform(rng, d_model, d_ff)
        rng,layer.ffn2 = linear_init_uniform(rng, d_ff, d_model)

        params.layers.append(layer)

    # Final normalization and output layer
    params.pre_output_norm = layernorm_init_identity(d_model)
    rng,params.output = linear_init_uniform(rng, d_model, n_vocab)

    return rng,config,params

def transformer(cfg, params, x: jnp.ndarray):
    """
    cfg: Config, from transformer_init, holds hyperparameters
    params: Current transformer parameters, initialized in init
    x: 1D array of L integers, representing the input sequence
    output: L x n_vocab logits
    """

    L = len(x)

    # Create mask: 0 to attend, -Inf to ignore
    mask = jnp.log(jnp.tril(jnp.ones((L, L))))

    # Start with token embeddings
    embeddings = cfg.lambda_e * params.embeddings[x, :]     # L x Dm

    # Add positional encodings 
    embeddings += cfg.lambda_pe * params.positional_encodings[:L, :]

    # Apply the transformer layers
    for layer in params.layers:

        # Layer-normalize embeddings
        t1 = vmap(center)(embeddings)
        t1 = elementwise_linear(layer.norm_self_attn, t1)   # L x Dm

        # Multi-head self-attention
        for head in layer.heads:

            # Project into this head's query/key space 
            query = linear(head.query, t1)                  # L x Dk
            key = linear(head.key, t1)                      # L x Dk

            score = query @ key.T + mask                    # L x L
            attn = jax.nn.softmax(cfg.tau * score, axis=1)  # L x L

            value = linear(head.value, t1)                  # L x Dm
            self_attn = attn @ value                        # L x Dm

            # Add this head's contribution into embeddings
            embeddings += self_attn                         # L x Dm

        # Layer-normalize embeddings
        t2 = vmap(center)(embeddings)
        t2 = elementwise_linear(layer.norm_ff, t2)          # L x Dm

        # Feedforward fully connected 
        t2 = linear(layer.ffn1, t2)                         # L x Dff
        t2 = jax.nn.relu(t2)
        t2 = linear(layer.ffn2, t2)                         # L x Dm

        # Add this layer's contribution into embeddings
        embeddings += t2

    # Layer-normalize embeddings
    embeddings = vmap(center)(embeddings)
    embeddings = elementwise_linear(params.pre_output_norm, embeddings)

    # And linearly project to output dimension
    return linear(params.output, embeddings)

def transformer_loss(cfg, params, x):
    """
    # Transformer loss for one example

    cfg: Config, from init
    params: Current transformer parameters, initialized in init
    x: 1D array of integers, representing the input sequence
    """
    output = transformer(cfg, params, x)

    return seq_crossentropy(output[:-1], x[1:])

def transformer_sample_unjit(cfg, params, seq: jnp.ndarray, length: int = 20):
    
    for i in range(length):
        output = transformer(cfg, params, seq)

        idx = jnp.argmax(output[-1])

        seq = jnp.concatenate((seq, idx[None]))

    return seq
