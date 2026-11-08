"""Permutation-invariant summary statistics of recorded activity.

The same statistics are used by every inference method. They don't depend on neuron
identity, so a trajectory from one network can be compared with simulations on a
different (assumed) network, which the cross-topology tests need. All statistics are
smooth, so they are safe to differentiate.
"""
from __future__ import annotations

import jax.numpy as jnp

STAT_NAMES = ["mean_rate", "rate_heterogeneity", "temporal_std", "participation_ratio",
              "mean_corr", "autocorr_lag1", "stim_r2"]
EPS = 1e-6


def _window_stats(R: jnp.ndarray, U: jnp.ndarray, ridge: float = 1e-2) -> jnp.ndarray:
    T, N = R.shape
    Rc = R - R.mean(0)
    C = Rc.T @ Rc / (T - 1)
    trC = jnp.trace(C)
    pr = trC ** 2 / (jnp.sum(C ** 2) + EPS) / N
    sd = jnp.sqrt(jnp.diag(C) + EPS)
    Cn = C / (sd[:, None] * sd[None, :])
    mean_corr = (Cn.sum() - jnp.trace(Cn)) / (N * (N - 1))
    ac1 = jnp.sum(Rc[:-1] * Rc[1:]) / (jnp.sum(Rc ** 2) + EPS)
    Uc = U - U.mean(0)
    beta = jnp.linalg.solve(Uc.T @ Uc + ridge * T * jnp.eye(U.shape[1]), Uc.T @ Rc)
    r2 = 1.0 - jnp.sum((Rc - Uc @ beta) ** 2) / (jnp.sum(Rc ** 2) + EPS)
    return jnp.stack([R.mean(), R.mean(0).std(), jnp.sqrt(jnp.mean(Rc ** 2) + EPS),
                      pr, mean_corr, ac1, r2])


def summary_stats(R: jnp.ndarray, U: jnp.ndarray, n_windows: int = 4) -> jnp.ndarray:
    """R: (T_rec, N) rates, U: (T_rec, M) binned stimulus -> (n_windows * 7,).

    Splitting into time windows captures how the dynamics change as plasticity acts."""
    L = R.shape[0] // n_windows
    Rw = R[: L * n_windows].reshape(n_windows, L, -1)
    Uw = U[: L * n_windows].reshape(n_windows, L, -1)
    return jnp.concatenate([_window_stats(Rw[w], Uw[w]) for w in range(n_windows)])


def stat_names(n_windows: int = 4) -> list[str]:
    return [f"{s}@w{w}" for w in range(n_windows) for s in STAT_NAMES]
