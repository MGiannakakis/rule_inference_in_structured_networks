"""Weights-visible baseline: when per-step rates *and* weights are observed, the rule is a
linear regression of dG on the monomial features. Used to show that rule inference is
trivial there, which motivates the activity-only setting."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from ..rules import rule_features
from ..simulate import SimConfig, simulate


def infer_from_weights(theta_true, net, stim, cfg: SimConfig, key, n_steps: int = 150,
                       max_samples: int = 200_000, seed: int = 0) -> dict:
    cfg1 = SimConfig(**{**cfg.__dict__, "rec_every": 1, "obs_noise": 0.0})
    R, G = simulate(jnp.asarray(theta_true), net, jnp.asarray(stim[:n_steps]), key, cfg1,
                    return_weights=True)
    R, G = np.asarray(R), np.asarray(G)
    mask = np.asarray(net.mask) > 0
    if not cfg.plastic_inhibitory:
        mask &= np.asarray(net.sign) > 0
    gmax = float(net.gmax)
    dG = (G[1:] - G[:-1]) / cfg.eta
    valid = mask[None] & (G[1:] > 0) & (G[1:] < gmax * 0.999)      # drop clipped synapses
    F = rule_features(R[:-1], R[:-1], G[:-1])                      # (T-1, N, N, D)
    X, y = F[valid], dG[valid]
    if X.shape[0] > max_samples:
        idx = np.random.default_rng(seed).choice(X.shape[0], max_samples, replace=False)
        X, y = X[idx], y[idx]
    theta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return {"theta": theta.astype(np.float32), "n_samples": int(X.shape[0])}
