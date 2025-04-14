"""
Pure-from-the-ground-up transformer, based on https://github.com/vpj/jax_transformer/blob/master/transformer.py

"""

from timer import timer

import jax
from jax import vmap
import jax.numpy as jnp

from jaxtyping import Array, Int

from functools import partial

import jax.experimental.host_callback

from awfutils import Arg, typecheck
from jaxutils.ParamsDict import ParamsDict


def rand(rng, f, shape, **kwargs):
    """
    Wrap jax.random.foo function to split the incoming rng, and return the new rng beside the payload

    rng = ... from previous code ...

    rng, vals1 = rand(rng, jax.random.uniform, (9,3), minval=-2.0, maxval=2.0)
    # ^-- rng is now newly split
    rng, vals2 = rand(rng, jax.random.normal, (3,9))
    # ^-- rng is split again
    """
    rng, rng1 = jax.random.split(rng)
    return rng, f(rng1, shape, **kwargs)


def linear_init_uniform(rng: jax.random.PRNGKey, in_features: int, out_features: int):
    """
    Initialize a linear layer with uniform weights and zero bias
    """
    params = ParamsDict()
    rnd_range = 1 / in_features**0.5
    rng, params.weight = rand(
        rng,
        jax.random.uniform,
        (in_features, out_features),
        minval=-rnd_range,
        maxval=rnd_range,
    )

    params.bias = jnp.zeros((out_features,))
    return rng, params


# Layer norm
def elementwise_linear_init_identity(shape):
    """
    Initialize an elementwise_linear layer with unit gain, zero bias
    """
    return ParamsDict(gain=jnp.ones(shape), bias=jnp.zeros(shape))


def linear(params, x: jnp.ndarray):
    return x @ params.weight + params.bias[None, :]


def elementwise_linear(params, x: jnp.ndarray):
    return params.gain[None, :] * x + params.bias[None, :]


def standardize(x, eps=1e-5):
    return (x - x.mean()) / (x.std() + eps)


flip_pe_coef = Arg("flip-pe", False, "Scale token embedding, not position embedding")


def transformer_init(
    rng: jax.random.PRNGKey,
    n_vocab: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    d_k: int,
    d_ff: int,
    max_len=4096,
):
    assert d_k * n_heads == d_model

    # Build config struct for call
    config = ParamsDict()
    config.d_model = d_model
    config.d_ff = d_ff
    config.d_k = d_k
    config.heads = n_heads
    if flip_pe_coef():
        config.lambda_e = d_model**-0.5
        config.lambda_pe = 1.0
    else:
        config.lambda_e = d_model**-0.5
        config.lambda_pe = 1.0
    config.tau = 1 / d_k**0.5

    # Build initializers for params
    params = ParamsDict()

    # Create embedding layer
    rng, params.embeddings = rand(rng, jax.random.normal, (n_vocab, d_model))

    # Positional encodings initialized to zeros
    params.positional_encodings = jnp.zeros((max_len, d_model))

    # For transformer layers
    params.layers = []
    for _ in range(n_layers):
        layer = ParamsDict()
        layer.norm_self_attn = elementwise_linear_init_identity(d_model)

        layer.heads = []
        for _ in range(n_heads):
            head = ParamsDict()
            rng, head.query = linear_init_uniform(rng, d_model, d_k)
            rng, head.key = linear_init_uniform(rng, d_model, d_k)
            rng, head.value = linear_init_uniform(rng, d_model, d_k)

            layer.heads.append(head)

        layer.norm_ff = elementwise_linear_init_identity(d_model)

        rng, layer.ffn1 = linear_init_uniform(rng, d_model, d_ff)
        rng, layer.ffn2 = linear_init_uniform(rng, d_ff, d_model)

        params.layers.append(layer)

    # Final normalization and output layer
    params.pre_output_norm = elementwise_linear_init_identity(d_model)
    rng, params.output = linear_init_uniform(rng, d_model, n_vocab)

    return rng, config, params


# Format off for the size annotations
# fmt: off
@partial(jax.jit, static_argnums=0)
@typecheck
def transformer(cfg, params, x: Int[Array, "L"]):
    """
    cfg: Config, from transformer_init, holds hyperparameters
    params: Current transformer parameters, initialized in init
    x: 1D array of L integers, representing the input sequence
    output: L x n_vocab logits
    """
    print("Compiling for L=", x.shape)

    L, = x.shape # x is just 1D. Vmap/pmap will handle batching

    # Make shape checkers for awfutils.typecheck
    LxL = lambda x: x.shape == (L, L)
    LxDk = lambda x: x.shape == (L, cfg.d_k)
    LxDff = lambda x: x.shape == (L, cfg.d_ff)
    LxDm = lambda x: x.shape == (L, cfg.d_model)

    # Create mask: 0 to attend, -Inf to ignore
    mask : LxL = jnp.log(jnp.tril(jnp.ones((L, L))))

    # Start with token embeddings
    embeddings : LxDm = cfg.lambda_e * params.embeddings[x, :]

    # Add (learned) positional encodings
    embeddings += cfg.lambda_pe * params.positional_encodings[:L, :]

    # Apply the transformer layers
    for layer in params.layers:

        # Layer-normalize embeddings
        t1 : LxDm = vmap(standardize)(embeddings)
        t1 = elementwise_linear(layer.norm_self_attn, t1)

        # Multi-head self-attention
        self_attns = []
        for head in layer.heads:

            # Project into this head's query/key space
            query : LxDk = linear(head.query, t1)
            key : LxDk = linear(head.key, t1)

            # Compute L x L attention matrix
            score : LxL = query @ key.T + mask
            attn : LxL = jax.nn.softmax(cfg.tau * score, axis=1)

            value : LxDk = linear(head.value, t1)
            self_attn : LxDk = attn @ value

            # Add this head's contribution into embeddings
            self_attns += [self_attn]  # [LxDk for #heads]

        t2 : LxDm = t1 + jnp.hstack(self_attns)

        # Layer-normalize embeddings
        t2 : LxDm = vmap(standardize)(t2)
        t2 : LxDm = elementwise_linear(layer.norm_ff, t2)

        # Feedforward fully connected
        t2 : LxDff = linear(layer.ffn1, t2)
        t2 = jax.nn.relu(t2)
        t2 : LxDm = linear(layer.ffn2, t2)

        # Add this layer's contribution into embeddings
        embeddings += t2

    # Layer-normalize embeddings
    embeddings = vmap(standardize)(embeddings)
    embeddings = elementwise_linear(params.pre_output_norm, embeddings)

    # And linearly project to output dimension
    return linear(params.output, embeddings) # L x n_vocab 
# fmt: on


def crossentropy(output: jnp.ndarray, target: int):
    return -jax.nn.log_softmax(output)[target]


def seq_crossentropy(output: jnp.ndarray, targets: jnp.ndarray):
    return vmap(crossentropy)(output, targets).mean()


def transformer_loss(cfg, params, x):
    """
    # Transformer loss for one example

    cfg: Config, from init
    params: Current transformer parameters, initialized in init
    x: 1D array of integers, representing the input sequence
    """
    output = transformer(cfg, params, x)

    return seq_crossentropy(output[:-1], x[1:])


# We don't jit this, as the loop will unroll, and take a long time to compile
def transformer_sample(cfg, params, seq: jnp.ndarray, length: int = 20):

    for _i in range(length):
        output = transformer(cfg, params, seq)

        idx = jnp.argmax(output[-1])

        seq = jnp.concatenate((seq, idx[None]))

    return seq
