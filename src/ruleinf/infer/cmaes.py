"""CMA-ES over rule coefficients; each generation is evaluated as one vmapped batch."""
from __future__ import annotations

import cma
import jax
import jax.numpy as jnp
import numpy as np

from ..rules import N_THETA
from ..simulate import SimConfig
from .objective import loss


def infer_cmaes(net, stim, s_obs, scale, cfg: SimConfig, key, bound: float = 1.5,
                popsize: int = 24, max_iter: int = 60, sigma0: float = 0.5,
                n_seeds: int = 2, n_windows: int = 4, seed: int = 0) -> dict:
    noise_keys = jax.random.split(key, n_seeds)
    batched = jax.jit(jax.vmap(
        lambda th: loss(th, noise_keys, net, stim, s_obs, scale, cfg, n_windows)))
    es = cma.CMAEvolutionStrategy(np.zeros(N_THETA), sigma0,
                                  {"bounds": [-bound, bound], "popsize": popsize,
                                   "maxiter": max_iter, "seed": seed + 1, "verbose": -9})
    hist = []
    while not es.stop():
        X = es.ask()
        f = np.asarray(batched(jnp.asarray(np.stack(X), dtype=jnp.float32)), dtype=float)
        es.tell(X, f.tolist())
        hist.append(f.min())
    theta = np.asarray(es.result.xbest, dtype=np.float32)
    return {"theta": theta, "loss": float(es.result.fbest), "history": np.array(hist),
            "theta_mean": np.asarray(es.result.xfavorite, dtype=np.float32)}
