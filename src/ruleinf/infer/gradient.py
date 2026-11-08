"""Gradient-based meta-learning of rule coefficients (backprop through the plastic dynamics).

Full BPTT with per-recording-bin rematerialisation (see simulate.py), global-norm gradient
clipping and NaN-zeroing. Several restarts from prior samples are optimised in parallel
with vmap; the restart with the lowest final loss is returned.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from ..rules import N_THETA
from ..simulate import SimConfig
from .objective import loss, sample_prior


def infer_gradient(net, stim, s_obs, scale, cfg: SimConfig, key, bound: float = 1.5,
                   n_restarts: int = 4, n_steps: int = 150, lr: float = 0.05,
                   n_seeds: int = 2, n_windows: int = 4) -> dict:
    k_init, k_noise = jax.random.split(key)
    theta0 = sample_prior(k_init, n_restarts, N_THETA, bound)
    noise_keys = jax.random.split(k_noise, n_seeds)
    opt = optax.chain(optax.zero_nans(), optax.clip_by_global_norm(1.0), optax.adam(lr))

    vg = jax.vmap(jax.value_and_grad(
        lambda th: loss(th, noise_keys, net, stim, s_obs, scale, cfg, n_windows)))

    @jax.jit
    def step(theta, state):
        l, g = vg(theta)
        upd, state = jax.vmap(opt.update)(g, state, theta)
        theta = jnp.clip(optax.apply_updates(theta, upd), -bound, bound)
        return theta, state, l

    theta, state = theta0, jax.vmap(opt.init)(theta0)
    hist = []
    for _ in range(n_steps):
        theta, state, l = step(theta, state)
        hist.append(np.asarray(l))
    final = np.asarray(jax.vmap(
        lambda th: loss(th, noise_keys, net, stim, s_obs, scale, cfg, n_windows))(theta))
    best = int(np.nanargmin(final))
    return {"theta": np.asarray(theta[best]), "loss": float(final[best]),
            "history": np.stack(hist)[:, best], "all_theta": np.asarray(theta),
            "all_loss": final}
