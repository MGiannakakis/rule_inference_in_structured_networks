"""Recovery metrics for an inferred rule."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .infer.objective import discrepancy, sim_stats
from .rules import classify


def theta_metrics(theta_hat, theta_true, bound: float = 1.5) -> dict:
    th, tt = np.asarray(theta_hat), np.asarray(theta_true)
    cos = float(th @ tt / (np.linalg.norm(th) * np.linalg.norm(tt) + 1e-12))
    return {"theta_rmse": float(np.sqrt(np.mean((th - tt) ** 2)) / bound),
            "theta_cosine": cos,
            "pred_rule": classify(th)}


def functional_error(theta_hat, theta_true, true_net, stim_test, scale, cfg, key,
                     n_seeds: int = 4, n_windows: int = 4) -> float:
    """Discrepancy on the TRUE network with a held-out stimulus between the inferred and
    true rules. This measures whether the inferred rule *behaves* like the true one, even
    when the coefficients differ (degeneracy)."""
    keys = jax.random.split(key, n_seeds)
    f = jax.vmap(lambda th, k: sim_stats(th, k, true_net, stim_test, cfg, n_windows),
                 in_axes=(None, 0))
    s_true = f(jnp.asarray(theta_true), keys).mean(0)
    s_hat = f(jnp.asarray(theta_hat), keys)
    return float(discrepancy(s_hat, s_true, scale).mean())
