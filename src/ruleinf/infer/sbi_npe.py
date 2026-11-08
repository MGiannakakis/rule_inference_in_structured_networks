"""Amortised neural posterior estimation (sbi / PyTorch) on summary statistics.

One posterior is trained per (assumed network, stimulus) and reused for every observation,
including observations from every true rule and true topology. The posterior width and
contraction then measure identifiability directly.
"""
from __future__ import annotations

import jax
import numpy as np

from ..rules import N_THETA
from ..simulate import SimConfig
from .objective import sample_prior, simulate_many


def simulate_training_set(net, stim, cfg: SimConfig, key, n_sims: int = 10_000,
                          bound: float = 1.5, n_windows: int = 4, batch: int = 128):
    k1, k2 = jax.random.split(key)
    theta = np.asarray(sample_prior(k1, n_sims, N_THETA, bound))
    x = simulate_many(theta, net, stim, cfg, k2, n_windows, batch)
    ok = np.all(np.isfinite(x), axis=1)
    return theta[ok], x[ok]


def train_npe(theta: np.ndarray, x: np.ndarray, bound: float = 1.5, seed: int = 0,
              density_estimator: str = "nsf", **train_kwargs):
    import torch
    from sbi.inference import NPE
    from sbi.utils import BoxUniform

    torch.manual_seed(seed)
    prior = BoxUniform(-bound * torch.ones(N_THETA), bound * torch.ones(N_THETA))
    inference = NPE(prior=prior, density_estimator=density_estimator)
    inference.append_simulations(torch.as_tensor(theta, dtype=torch.float32),
                                 torch.as_tensor(x, dtype=torch.float32))
    estimator = inference.train(show_train_summary=False, **train_kwargs)
    return inference.build_posterior(estimator)


def posterior_summary(posterior, x_obs: np.ndarray, theta_true: np.ndarray | None = None,
                      bound: float = 1.5, n_samples: int = 2000) -> dict:
    import torch

    xo = torch.as_tensor(np.asarray(x_obs), dtype=torch.float32)
    samples = posterior.sample((n_samples,), x=xo, show_progress_bars=False).numpy()
    prior_var = (2 * bound) ** 2 / 12
    out = {"theta": samples.mean(0), "theta_median": np.median(samples, 0),
           "std": samples.std(0), "contraction": 1 - samples.var(0) / prior_var,
           "samples": samples}
    if theta_true is not None:
        tt = torch.as_tensor(np.asarray(theta_true)[None], dtype=torch.float32)
        lp_true = posterior.log_prob(tt, x=xo).item()
        lp_s = posterior.log_prob(torch.as_tensor(samples[:500]), x=xo).numpy()
        out["true_logprob"] = lp_true
        out["hpd_rank"] = float((lp_s > lp_true).mean())   # ~U(0,1) if calibrated
    return out
