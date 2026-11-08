"""Shared objective: simulate on an *assumed* network, compare summary stats with observed ones.

All functions are module-level and jitted with the network/stimulus as traced arguments,
so switching between assumed networks does not trigger recompilation.
"""
from __future__ import annotations

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from ..simulate import SimConfig, recorded_stimulus, simulate
from ..summary import summary_stats
from ..topologies import Network


@partial(jax.jit, static_argnames=("cfg", "n_windows"))
def sim_stats(theta, key, net: Network, stim, cfg: SimConfig, n_windows: int = 4):
    R = simulate(theta, net, stim, key, cfg)
    return summary_stats(R, recorded_stimulus(stim, cfg.rec_every), n_windows)


@partial(jax.jit, static_argnames=("cfg", "n_windows"))
def batch_sim_stats(thetas, keys, net: Network, stim, cfg: SimConfig, n_windows: int = 4):
    """thetas (B, D), keys (B, 2) -> (B, S)."""
    return jax.vmap(lambda th, k: sim_stats(th, k, net, stim, cfg, n_windows))(thetas, keys)


def discrepancy(stats_sim, s_obs, scale):
    """Normalised squared distance (mean over stats). NaN/inf -> large finite value."""
    d = jnp.mean(((stats_sim - s_obs) / scale) ** 2, axis=-1)
    return jnp.where(jnp.isfinite(d), d, 1e3)


@partial(jax.jit, static_argnames=("cfg", "n_windows"))
def loss(theta, keys, net: Network, stim, s_obs, scale, cfg: SimConfig, n_windows: int = 4):
    """Mean discrepancy over a fixed set of noise seeds (common random numbers)."""
    s = jax.vmap(lambda k: sim_stats(theta, k, net, stim, cfg, n_windows))(keys)
    return discrepancy(s, s_obs, scale).mean()


def sample_prior(key, n: int, dim: int, bound: float) -> jnp.ndarray:
    return jax.random.uniform(key, (n, dim), minval=-bound, maxval=bound)


def simulate_many(thetas, net, stim, cfg, key, n_windows=4, batch=128) -> np.ndarray:
    """Batched simulation of summary stats for many thetas, returned as numpy."""
    out = []
    keys = jax.random.split(key, thetas.shape[0])
    for i in range(0, thetas.shape[0], batch):
        th, ks = thetas[i:i + batch], keys[i:i + batch]
        pad = batch - th.shape[0]           # pad last batch to avoid recompilation
        if pad:
            th = jnp.concatenate([th, jnp.repeat(th[-1:], pad, 0)])
            ks = jnp.concatenate([ks, jnp.repeat(ks[-1:], pad, 0)])
        out.append(np.asarray(batch_sim_stats(th, ks, net, stim, cfg, n_windows))[: batch - pad])
    return np.concatenate(out)


def stat_scale(net, stim, cfg, key, dim, bound, n=256, n_windows=4) -> jnp.ndarray:
    """Per-statistic scale = std across prior simulations on the assumed network."""
    k1, k2 = jax.random.split(key)
    s = simulate_many(sample_prior(k1, n, dim, bound), net, stim, cfg, k2, n_windows)
    s = s[np.all(np.isfinite(s), axis=1)]
    sd = s.std(0)
    return jnp.asarray(sd + 1e-2 * np.median(sd) + 1e-6)
