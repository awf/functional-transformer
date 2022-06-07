"""
Pure-from-the-ground-up mnist convnet, based on
"""

from timer import timer

import time
import re
import sys
import os
import logging

import jax
import jax.numpy as jnp
import jax.nn as jnn
import numpy as np
from jax import vmap

from functools import partial
from itertools import islice

import wandb

from icecream import ic

from jaxutils.ParamsDict import ParamsDict
from jaxutils.Arg import Arg
from jaxutils.rand import rand
from jaxutils.Adam import Adam

# Noisily fail when arrays are the wrong size
from jax.config import config

config.update("jax_numpy_rank_promotion", "raise")

LOGLEVEL = os.environ.get("LOGLEVEL", "INFO").upper()
logger = logging.getLogger("pure-tranfomer")
logger.setLevel(level=LOGLEVEL)
timer = timer.get_timer(logging.WARNING)
db = logger.debug

# ---------------------------------------------
def maxpool(input, window_shape):
    """
    Input:
        x             2D array of size (XH x XW)
        window_shape  Pair (KH,KW)
    Output:
        2D array of size (H x W)
    """
    KH, KW = window_shape
    XH, XW = input.shape
    H, W = XH // KH, XW // KW
    # Drop last columns/rows if shape doesn't divide size
    x = input[: H * KH, : W * KW]
    rs = jnp.reshape(x, (H, KH, W, KW), order="C")
    return jnp.max(rs, axis=(1, 3))


def dropout(rng, keep_prob, x):
    mask = jax.random.bernoulli(rng, p=keep_prob, shape=x.shape)
    return mask * x


def conv(x, k):
    """
    Simple N-D conv, no batching
    """
    return jax.scipy.signal.convolve(x, k, mode="valid")


# fmt: off
def cnn(cfg, rng, params, x):  # x : W x H
    maxpool_axis2 = vmap(maxpool, in_axes=(2, None), out_axes=2)

    x = vmap(conv, in_axes=(None, 2), out_axes=2)(x, params.layer1)  # W1 x H1 x C

    x = jnn.relu(x)
    x = maxpool_axis2(x, cfg.maxpool_window)        # W2 x H2 x C       W2 = W1 // 2
    
    ic(x.shape)
    x = vmap(conv, in_axes=(None, 3), out_axes=2)(x, params.layer2)                # W3 x H3

    x = jnp.squeeze(x, axis=3)

    x = jnn.relu(x)

    ic(x.shape)
    x = maxpool_axis2(x, cfg.maxpool_window)
    ic(x.shape)
    
    x = dropout(rng, cfg.dropout_keep_prob, x)
    x = x.flatten() @ params.dense
    return x
# fmt:on


def cnn_loss(cfg, rng, params, x, l):
    y = cnn(cfg, rng, params, x)
    return -jax.nn.log_softmax(y)[l]


def cnn_init(rng):
    params = ParamsDict()
    rng, params.layer1 = rand(rng, jax.random.uniform, (3, 3, 32))
    rng, params.layer2 = rand(rng, jax.random.uniform, (3, 3, 32, 64))
    rng, params.dense = rand(rng, jax.random.uniform, (5 * 5 * 64, 10))
    # How to automate this size computation?

    cfg = ParamsDict()
    cfg.maxpool_window = (2, 2)
    cfg.dropout_keep_prob = 0.5
    cfg.tau = 1.0
    return rng, cfg, params


def main():

    lr = Arg(flag="lr", doc="Learning rate", default=0.001)
    beta1 = Arg(flag="beta1", doc="Adam beta1", default=0.9)
    beta2 = Arg(flag="beta2", doc="Adam beta2", default=0.99)
    batch_size = Arg(flag="batch-size", doc="Batch size", default=128)
    epochs = Arg("epochs", 32)

    # Init the model params
    dropout_keep_prob = Arg("dropout", 8, "Dropout keep prob")

    # save = Arg("save", "", "Save mode.  Log run to wandb, lengthen epochs and batches")

    # if save():
    #     wandb.init(
    #         project="pure-transformer",
    #         entity="awfidius",
    #         name=save() if len(save()) else None,
    #         config=Arg.config(),
    #     )
    # else:
    #     print("Quick mode, disabling wandb, using small prime sizes")
    #     wandb.init(mode="disabled")
    #     epochs.default = 2
    #     batches.default = 10
    #     # Sizes are prime numbers, to catch any mismatches
    #     d_model.default = 93
    #     d_k.default = 13
    #     heads.default = 7
    #     d_ff.default = 111

    # Create PRNG key
    rnd_key = jax.random.PRNGKey(42)

    # Create dataset
    with np.load("mnist.npz") as f:
        train_x = f["train_x"]
        train_l = f["train_l"]
        val_x = f["val_x"]
        val_l = f["val_l"]
        test_x = f["test_x"]
        test_l = f["test_l"]

    rnd_key, cfg, params = cnn_init(rnd_key)

    @partial(jax.jit, static_argnums=0)
    def loss_batch(cfg, rng, params, seq_x, seq_l):
        batched = vmap(cnn_loss, in_axes=(None, None, None, 0, 0), out_axes=0)
        return jnp.mean(batched(cfg, rng, params, seq_x, seq_l))

    value_and_grad_loss_batch_unjit = jax.value_and_grad(loss_batch, argnums=2)
    value_and_grad_loss_batch = jax.jit(
        value_and_grad_loss_batch_unjit, static_argnums=0
    )

    optimizer = Adam(params, lr=lr(), betas=(beta1(), beta2()))

    for epoch in range(epochs()):

        # Iterate through batches
        for i in range(0, len(train_x), batch_size()):
            data_x = train_x[i : i + batch_size()]
            data_l = train_l[i : i + batch_size()]

            # Get loss and gradients
            rnd_key, rng = jax.random.split(rnd_key)
            loss, grads = value_and_grad_loss_batch(cfg, rng, params, data_x, data_l)

            print(
                wandb.run.name,
                "loss",
                loss,
            )

            # wandb.log(
            #     {
            #         "time": total_time,
            #         "batch": i,
            #         "loss": loss,
            #         # "gnorms": wandb.Image(gnorms_all, caption="Parameter norm"),
            #     }
            # )  # 'gnorms': plt,  'gnorms_table': gnorms_table})

            params = optimizer.step(params, grads)


def test_maxpool():
    H, W = 11, 7
    KH, KW = 3, 5
    x = np.cumsum(np.ones((KH * H, KW * W)), axis=0)
    x = np.rint(np.cumsum(x, axis=1))

    test_out = np.zeros((H, W))
    for iout in range(H):
        for jout in range(W):
            window = x[iout * KH : iout * KH + KH, jout * KW : jout * KW + KW]
            test_out[iout, jout] = window.flatten().max()
    # print(x[0:3,0:5])
    # print(x[0:3,5:10])
    # print(x[3:6,0:5])
    y = maxpool(x, window_shape=(KH, KW))

    print(test_out)

    print(y)

    assert jnp.all(test_out == y)


if __name__ == "__main__":
    main()