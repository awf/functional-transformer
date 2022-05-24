# functional-transformer
A pure-functional implementation of a machine learning transformer model in Python/JAX

Given only a few simple BLAS-like functions:
```python
def linear(params, x: jnp.ndarray):
    return x @ params.weight + params.bias[None,:]

def elementwise_linear(params, x: jnp.ndarray):
    return params.gain[None,:] * x + params.bias[None,:]

def center(x, eps = 1e-5):
    return (x - x.mean())/(x.std() + eps)
```
then the entire transformer forward computation is 22 lines of code:
```python
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
```
and the random initialization is even shorter:
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
