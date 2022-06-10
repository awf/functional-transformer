"""
Pure-from-the-ground-up mnist convnet, based on
"""
import matplotlib.pylab as plt
import numpy as np

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

import copy
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
timer = timer.get_timer(logging.INFO)
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

    x = vmap(conv, in_axes=(None, 2), out_axes=2)(x, params.layer1)

    x = jnn.relu(x)
    x = maxpool_axis2(x, cfg.maxpool_window)
    
    x = vmap(conv, in_axes=(None, 3), out_axes=2)(x, params.layer2)
    x = jnn.relu(x)

    x = jnp.squeeze(x, axis=3)
    x = maxpool_axis2(x, cfg.maxpool_window)
    x = dropout(rng, cfg.dropout_keep_prob, x)

    x = x.flatten()
    x = x @ params.dense1
    x = jnn.relu(x)
    x = x @ params.dense2
    assert x.shape == (10,)
    return x * cfg.tau
# fmt:on


def cnn_loss(cfg, rng, params, x, l):
    y = cnn(cfg, rng, params, x)
    return -jax.nn.log_softmax(y)[l]


dropout_keep_prob = Arg("dropout", 0.5, "Dropout keep prob")


def cnn_init(rng):
    params = ParamsDict()
    rng, params.layer1 = rand(
        rng, jax.random.uniform, (3, 3, 32), minval=-0.5, maxval=0.5
    )
    rng, params.layer2 = rand(
        rng, jax.random.uniform, (3, 3, 32, 64), minval=-0.5, maxval=0.5
    )
    s = 1 / (5 * 5 * 64)
    H = 128
    rng, params.dense1 = rand(
        rng, jax.random.uniform, (5 * 5 * 64, H), minval=-s, maxval=s
    )
    rng, params.dense2 = rand(rng, jax.random.uniform, (H, 10), minval=-s, maxval=s)

    cfg = ParamsDict()
    cfg.maxpool_window = (2, 2)
    cfg.dropout_keep_prob = dropout_keep_prob()
    cfg.tau = 0.01
    return rng, cfg, params


@partial(jax.jit, static_argnames="cfg")
def cnn_batched(cfg, rng, params, x):
    f = partial(cnn, cfg, rng, params)
    return vmap(f)(x)


print("Loaded")
# %%


def main():

    lr = Arg(flag="lr", doc="Learning rate", default=0.001)
    beta1 = Arg(flag="beta1", doc="Adam beta1", default=0.9)
    beta2 = Arg(flag="beta2", doc="Adam beta2", default=0.99)
    batch_size = Arg(flag="batch-size", doc="Batch size", default=128)
    epochs = Arg("epochs", 32)

    # Init the model params

    save = Arg("save", "", "Save mode.  Log run to wandb, lengthen epochs and batches")

    if save():
        wandb.init(
            project="pure-convnet",
            entity="awfidius",
            name=save() if len(save()) else None,
            config=Arg.config(),
        )
    else:
        print("Quick mode, disabling wandb, using small prime sizes")
        wandb.init(mode="disabled")
        epochs.default = 2

    # Create PRNG key
    rnd_key = jax.random.PRNGKey(42)

    # Create dataset
    # wget https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz
    with np.load("mnist.npz") as f:
        scale = 1 / 256
        train_and_val_x = f["x_train"] * scale
        train_and_val_l = f["y_train"]

        # test_x = f["x_test"] * scale
        # test_l = f["y_test"]

    train_x = train_and_val_x[:50000]
    train_l = train_and_val_l[:50000]

    val_x = train_and_val_x[50000:]
    val_l = train_and_val_l[50000:]
    val_l_inds = np.argsort(val_l)

    rnd_key, cfg, params = cnn_init(rnd_key)

    sizes = jax.tree_map(lambda v: np.prod(v.shape), params)
    sizes.print("sizes:")
    print("Total parameter count:", np.sum(jax.tree_flatten(sizes)[0]))

    cfg_inference = copy.deepcopy(cfg)
    cfg_inference.dropout_keep_prob = 1.0

    ic(cfg, cfg_inference)

    @partial(jax.jit, static_argnums=0)
    def loss_batch(cfg, rng, params, seq_x, seq_l):
        f = partial(cnn_loss, cfg, rng, params)
        y = vmap(f)(seq_x, seq_l)
        return jnp.mean(y)

    value_and_grad_loss_batch_unjit = jax.value_and_grad(loss_batch, argnums=2)
    value_and_grad_loss_batch = jax.jit(
        value_and_grad_loss_batch_unjit, static_argnames="cfg"
    )

    np.set_printoptions(precision=3)

    def amap(f, xs):
        return np.array(list(map(f, xs)))

    optimizer = Adam(params, lr=lr(), betas=(beta1(), beta2()))

    for epoch in range(epochs()):

        # Iterate through batches
        for i in range(0, len(train_x), batch_size()):
            data_x = train_x[i : i + batch_size()]
            data_l = train_l[i : i + batch_size()]

            # Get loss and gradients
            rnd_key, rng = jax.random.split(rnd_key)
            loss, grads = value_and_grad_loss_batch(cfg, rng, params, data_x, data_l)

            if i % 10 == 0:
                pred_y = cnn_batched(cfg_inference, rng, params, val_x)
                pred_l = jnp.argmax(pred_y, axis=1)

                val_num_correct = np.count_nonzero(pred_l == val_l)
                val_error = 100 - val_num_correct / len(val_l) * 100
                val_loss = loss_batch(cfg_inference, rng, params, val_x, val_l)
                print(
                    f"Validation loss/acc: {val_loss:6.4f}/{val_error:.2f}"
                    + f", N={amap(jnp.linalg.norm, jax.tree_leaves(params))}"
                    + f", logits={amap(jnp.linalg.norm, pred_y.T).shape}"
                    + f", logits={amap(jnp.linalg.norm, pred_y.T)}"
                )

                preds_inds = range(0, len(val_l), len(val_l) // 300)
                wandb.log(
                    {
                        "batch": i,
                        "loss": loss,
                        "val_error": val_error,
                        # "gradnorms": list(map(jnp.linalg.norm, jax.tree_leaves(params))),
                        # "preds": wandb.Image(
                        #     -np.array(pred_y[val_l_inds[preds_inds], :].T)
                        # ),
                    }
                )
            # else:
            # wandb.log({"batch": i,"loss": loss}}

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

# %%
