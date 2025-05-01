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

from awfutils import Arg
from jaxutils.datasets import TinyShakespeare
from jaxutils.Adam import Adam
from jaxutils.show_jaxpr import show_jaxpr_and_xla, show_xla, show_jaxpr

jnp.set_printoptions(threshold=20, edgeitems=3, linewidth=2048, precision=3)
np.set_printoptions(threshold=20, edgeitems=3, linewidth=2048, precision=3)

# Noisily fail when arrays are the wrong size
jax.config.update("jax_numpy_rank_promotion", "raise")

jax.config.update(
    "jax_log_compiles", Arg("log-compiles", False, "Log JAX recompilations").peek()
)
jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")

LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logger = logging.getLogger("pure-transformer")
logger.setLevel(level=LOGLEVEL)
timer = timer.get_timer(logging.WARNING)
db = logger.debug


def tree_axpy(a, x, y):
    return jax.tree.map(lambda x, y: a * x + y, x, y)


def main():

    lr = Arg(flag="lr", doc="Learning rate", default=0.001)
    beta1 = Arg(flag="beta1", doc="Adam beta1", default=0.9)
    beta2 = Arg(flag="beta2", doc="Adam beta2", default=0.99)
    seq_len = Arg(flag="seq-len", doc="Sequence length", default=32)
    batch_size = Arg(flag="batch-size", doc="Batch size", default=128)
    epochs = Arg("epochs", 32)
    batches = Arg("batches", sys.maxsize, "Max batches")

    # Init the model params
    heads = Arg("heads", 8, "Number of attention heads")
    d_model = Arg("dmodel", 512, "Embedding dimension")
    d_k = Arg("dk", 64, "Attention head dimension")
    d_ff = Arg("dff", 512, "Feedforward layer dimension")
    n_layers = Arg("layers", 3, "Number of layers")

    save = Arg("save", "", "Save mode.  Log run to wandb, lengthen epochs and batches")

    if save():
        wandb.init(
            project="pure-transformer",
            entity="awfidius",
            name=save() if len(save()) else None,
            config=Arg.config(),
        )
    else:
        print("Quick mode, disabling wandb, using small prime sizes")
        wandb.init(mode="disabled")
        epochs.default = 5
        batches.default = 10
        # Sizes are prime numbers, to catch any mismatches
        d_model.default = 13 * 7
        d_k.default = 13
        heads.default = 7
        d_ff.default = 111

    start = time.time()

    # Create PRNG key
    rnd_key = jax.random.PRNGKey(42)

    # Create dataset
    dataset = TinyShakespeare(rnd_key, seq_len=seq_len(), batch_size=batch_size())
    tostr = lambda x: "".join([dataset.itos[i] for i in x]).replace("\n", "\\n")

    rnd_key, cfg, params = transformer_init(
        rnd_key,
        dataset.n_tokens,
        d_model=d_model(),
        n_layers=n_layers(),
        n_heads=heads(),
        d_k=d_k(),
        d_ff=d_ff(),
    )

    names = [k for (k, _) in params.items()]
    print(names)
    assert len(names) == len(jax.tree.flatten(params)[0])

    # gnorms_table = wandb.Table(columns=names)
    # wandb.log({"gnorms_table": gnorms_table})

    sizes = jax.tree.map(lambda v: np.prod(v.shape), params)
    sizes.print("sizes:")
    print("Total parameter count:", np.sum(jax.tree.flatten(sizes)[0]))
    # sizes_table = wandb.Table(columns=['param','size'])

    @partial(jax.jit, static_argnums=0)
    def loss_batch(cfg, params, seq):
        batched = vmap(transformer_loss, in_axes=(None, None, 0), out_axes=0)
        return jnp.mean(batched(cfg, params, seq))

    # show_jaxpr(get_loss_batch, (params, *islice(dataset,1)))
    grad_loss_batch_unjit = jax.grad(loss_batch, argnums=1)
    grad_loss_batch = jax.jit(grad_loss_batch_unjit, static_argnums=0)

    value_and_grad_loss_batch_unjit = jax.value_and_grad(loss_batch, argnums=1)
    value_and_grad_loss_batch = jax.jit(
        value_and_grad_loss_batch_unjit, static_argnums=0
    )

    matches = re.search("--xla_dump_to=([^ ]+)", os.environ.get("XLA_FLAGS") or "")
    if matches:
        fn = matches[1] + "/grad_loss_batch.jaxpr.py"
        with open(fn, "w") as file:
            # xla = jax.xla_computation(loss_batch, static_argnums=0)(cfg, params, *islice(dataset,1))
            # print("XLA=", xla.as_hlo_text())
            show_jaxpr(
                grad_loss_batch,
                (cfg, params, *islice(dataset, 1)),
                file=file,
                static_argnums=0,
            )
        print("Saved jaxpr to", fn)

    sgd = Arg("sgd", False, "Pure sgd")

    # grad_loss_batch = jax.pjit(grad_loss_batch_unjit, static_argnums=0)

    optimizer = Adam(params, lr=lr(), betas=(beta1(), beta2()))

    gnorms_all = np.zeros((len(names), 0))
    for epoch in range(epochs()):

        if epoch == 0:
            # epoch zero is straight through to sample
            pass
        else:
            # Iterate through batches
            for i, data in enumerate(islice(dataset, batches())):
                # Get loss and gradients
                loss, grads = value_and_grad_loss_batch(cfg, params, data)

                # print(f"{wandb.run.name} loss {loss.item()} data {tostr(data[0])}")
                total_time = time.time() - start

                wandb.log({"time": total_time, "batch": i, "loss": loss})

                # Update parameters
                if sgd():
                    params = tree_axpy(-lr(), grads, params)
                else:
                    params = optimizer.step(params, grads)

        # Log a sample after each epoch
        with timer("sample and loss"):
            test_data = jnp.hstack([*islice(dataset, 4)])
            loss = loss_batch(cfg, params, test_data)

            prompt = [dataset.stoi[c] for c in "Au"]
            sampled = transformer_sample(
                cfg, params, jnp.array(prompt), length=20 + epoch
            )
            print(
                f"E{epoch} L{loss:.3f}",
                f"Sample [{tostr(prompt)}|{tostr(sampled[len(prompt):])}]",
            )

    # Grab Current Time After Running the Code
    end = time.time()
    total_time = end - start
    print("TIME: " + str(total_time))


if __name__ == "__main__":
    main()
