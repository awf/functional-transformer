A fully functional (pun intended) implementation of a machine learning transformer model in Python/JAX.  I do realize that 'pure functional' and 'Python' are not necessarily [mots quit vont très bien ensemble](https://forum.wordreference.com/threads/sont-les-mots-qui-vont-tr%C3%A8s-bien-ensemble.1832510/), but I'm sure you'll agree on reading the code that it has [una anima di pura programmazione funzionale](https://jax.readthedocs.io/en/latest/notebooks/Common_Gotchas_in_JAX.html).  And a little [macaronica](https://en.wikipedia.org/wiki/Macaronic_language) appeals to the peasant soul.  In other words, don't worry about the language... 

Given only a few simple BLAS-like functions:
```python
def linear(params, x: jnp.ndarray):
    return x @ params.weight + params.bias[None,:]

def elementwise_linear(params, x: jnp.ndarray):
    return params.gain[None,:] * x + params.bias[None,:]

def standardize(x, eps = 1e-5):
    return (x - x.mean())/(x.std() + eps)
```
then the entire transformer forward computation is 25 lines of code (excerpt from `transformer.py`):
```python
def transformer(cfg, params, x: Int[Array, "L"]):
    """
    cfg: Config, from transformer_init, holds hyperparameters
    params: Current transformer parameters, initialized in init
    x: 1D array of L integers, representing the input sequence
    output: L x n_vocab logits
    """
    L, = x.shape # x is just 1D. Vmap/pmap will handle batching

    # Make shape checkers (https://github.com/awf/awfutils?tab=readme-ov-file#typecheck)
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
```

The loss and its gradient needs a few more lines:
```python
def crossentropy(output: jnp.ndarray, target: int):
    return -jax.nn.log_softmax(output)[target]

def seq_crossentropy(output: jnp.ndarray, targets: jnp.ndarray):
    return vmap(crossentropy)(output, targets).mean()

def transformer_loss(cfg, params, x):
    output = transformer(cfg, params, x)

    return seq_crossentropy(output[:-1], x[1:])

# Gradient wrt 'params'
grad_loss = jax.grad(transformer_loss, argnums=1)
```

The random initialization is also short:
```python
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
```

Add an optimizer, and we are pronto a romblare.

## Running
```sh
$ export JAX_PLATFORM_NAME=gpu # or cpu
$ export JAX_LOG_COMPILES=1 # or 0
$ export XLA_FLAGS=--xla_dump_to=./xla-dumps/  # Also dumps jaxprs to this folder
$ python main.py -help
$ python main.py -layers 3 -dmodel 512 -heads 8 -dk 64 -dff 2048 
```

Results at https://wandb.ai/awfidius/pure-transformer

## Acknowledgements

The model is based on https://github.com/vpj/jax_transformer/blob/master/transformer.py, and the Adam and Dataset 
classes in jaxutils are almost direct copies from https://github.com/vpj/jax_transformer
