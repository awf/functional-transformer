"""
Pure-from-the-ground-up transformer, based on https://github.com/vpj/jax_transformer
"""

from transformer import *

import time
import re
import sys
import os
import logging

import jax
import jax.numpy as jnp
import numpy as np

from functools import partial
from itertools import islice

import wandb

from jaxutils.Arg import Arg
from jaxutils.dataset import TinyShakespeare
from jaxutils.Adam import Adam
from jaxutils.show_jaxpr import show_jaxpr_and_xla, show_xla, show_jaxpr

jnp.set_printoptions(threshold=20,edgeitems=3,linewidth=2048,precision=3)
np.set_printoptions(threshold=20, edgeitems=3,linewidth=2048,precision=3)

# Noisily fail when arrays are the wrong size
from jax.config import config
config.update("jax_numpy_rank_promotion", "raise") 

jit_sampler = Arg('jit-sampler', False)
sample = transformer_sample_unjit
if jit_sampler.peek():
    sample = jax.jit(sample,static_argnums=(0,))



LOGLEVEL = os.environ.get('LOGLEVEL', 'INFO').upper()
logger = logging.getLogger('pure-tranfomer')
logger.setLevel(level=LOGLEVEL)
timer = timer.get_timer(logging.WARNING)
db = logger.debug

def main():

    lr = Arg(flag="lr", doc="Learning rate", default=0.001)
    beta1 = Arg(flag="beta1", doc="Adam beta1", default=0.9)
    beta2 = Arg(flag="beta2", doc="Adam beta2", default=0.99)
    seq_len = Arg(flag="seq-len", doc="Sequence length", default=32)
    batch_size = Arg(flag="batch-size", doc="Batch size", default=128)
    epochs = Arg('epochs', 32)
    batches = Arg('batches', sys.maxsize, 'Max batches')

    # Init the model params
    heads = Arg('heads', 8, 'Number of attention heads')
    d_model = Arg('dmodel', 512, 'Embedding dimension')
    d_k = Arg('dk', 64, 'Attention head dimension')
    d_ff = Arg('dff', 512, 'Feedforward layer dimension')
    n_layers = Arg('layers', 3, 'Number of layers')

    save = Arg('save', False, 'Save mode.  Log run to wandb, lengthen epochs and batches')

    if save():
        wandb.init(project="pure-transformer", entity="awfidius", config=Arg.config())
    else:
        print('Quick mode, disabling wandb')
        wandb.init(mode='disabled')
        epochs.default = 2
        # Choose primes, to catch any size mismatches
        d_model.default = 93
        d_k.default = 13
        heads.default = 7
        d_ff.default = 111

    start = time.time()

    # Create PRNG key
    rnd_key = jax.random.PRNGKey(0)
    # Create dataset
    dataset = TinyShakespeare(rnd_key, seq_len=seq_len(), batch_size=batch_size())
    tostr = lambda x: ''.join([dataset.itos[i] for i in x]).replace('\n', '\\n')

    rnd_key,cfg,params = transformer_init(rnd_key, dataset.n_tokens, 
                                d_model=d_model(), d_k=d_k(), n_layers=n_layers(), n_heads=heads(), d_ff=d_ff())

    names = [k for (k,_) in params.items()]
    print(names)
    assert len(names) == len(jax.tree_flatten(params)[0])

    #gnorms_table = wandb.Table(columns=names)
    #wandb.log({'gnorms_table': gnorms_table})

    sizes = jax.tree_map(lambda v:np.prod(v.shape), params)
    sizes.print('sizes:')
    print('Total parameter count:', np.sum(jax.tree_flatten(sizes)[0]))
    # sizes_table = wandb.Table(columns=['param','size'])

    @partial(jax.jit, static_argnums=0)
    def loss_batch(cfg, params, seq): # seq: B x S
        batched = vmap(transformer_loss, in_axes=(None, None, 0), out_axes=0)
        return jnp.mean(batched(cfg, params, seq))


    # show_jaxpr(get_loss_batch, (params, *islice(dataset,1)))
    grad_loss_batch_unjit = jax.grad(loss_batch, argnums=1)
    grad_loss_batch = jax.jit(grad_loss_batch_unjit, static_argnums=0)

    matches = re.search('--xla_dump_to=([^ ]+)', os.environ.get('XLA_FLAGS') or '')
    if matches:
        fn = matches[1] + "/grad_loss_batch.jaxpr.py"
        with open(fn, "w") as file:
            # xla = jax.xla_computation(loss_batch, static_argnums=0)(cfg, params, *islice(dataset,1))
            # print("XLA=", xla.as_hlo_text())
            show_jaxpr(grad_loss_batch, (cfg, params, *islice(dataset,1)), file=file, static_argnums=0)
        print('Saved jaxpr to', fn)

    # grad_loss_batch = jax.pjit(grad_loss_batch_unjit, static_argnums=0)

    optimizer = Adam(params, lr=lr(), betas=(beta1(),beta2()))


    for _epoch in range(epochs()):

        # Iterate through batches
        for i,data in enumerate(islice(dataset, sys.maxsize if save() else 10)):
            # Compute and log the loss
            loss = loss_batch(cfg, params, data)

            # Get the gradients
            grads = grad_loss_batch(cfg, params, data)

            # gnorms = jax.tree_map(lambda v:np.round(np.log((jnp.linalg.norm(v)))), grads)
            # if i == 0:
            #     gnorms.print('gnorms:')       

            # gnorms,_ = jax.tree_flatten(gnorms)
            # gnorms_table.add_data(*gnorms)
            # plt.plot(gnorms)
            # plt.ylabel("gnorms")

            print(wandb.run.name, 'loss',loss, 'sample', tostr(data[np.random.randint(0,len(data))])) # , 'gnorms', gnorms)
            total_time = time.time() - start

            wandb.log({"time": total_time, "batch": i, "loss": loss}) # 'gnorms': plt,  'gnorms_table': gnorms_table})

            # Update parameters
            params = optimizer.step(params, grads)

        # Log a sample after each epoch
        prompt = [dataset.stoi[c] for c in 'Au']
        with timer('sample'):
            sampled = sample(cfg, params, jnp.array(prompt))[len(prompt):]
            print(loss, tostr(prompt) + '|' + tostr(sampled))

    # Grab Current Time After Running the Code
    end = time.time()
    total_time = end - start
    print("TIME: "+ str(total_time))


if __name__ == '__main__':
    main()
