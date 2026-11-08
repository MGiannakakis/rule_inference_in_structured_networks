"""Positive control: matched (exact) network, compare inferred-rule loss with the noise floor.

The noise floor is the loss of theta_true itself under fresh noise seeds. If an optimiser
reaches the floor with theta far from theta_true, the rule is not identifiable from these
observations (degeneracy), not merely badly optimised.

    python experiments/positive_control.py --topology modular --rule oja
"""
from __future__ import annotations

import argparse
import time

import jax
import jax.numpy as jnp
import numpy as np

from ruleinf.infer.cmaes import infer_cmaes
from ruleinf.infer.gradient import infer_gradient
from ruleinf.infer.objective import loss, sim_stats, stat_scale
from ruleinf.metrics import functional_error, theta_metrics
from ruleinf.rules import N_THETA, PRESETS, TERM_NAMES
from ruleinf.simulate import SimConfig
from ruleinf.stimuli import make_stimulus
from ruleinf.topologies import make_network


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--topology", default="modular")
    p.add_argument("--rule", default="oja")
    p.add_argument("--stimulus", default="ou")
    p.add_argument("--grad-steps", type=int, default=150)
    p.add_argument("--cmaes-iter", type=int, default=60)
    p.add_argument("--popsize", type=int, default=24)
    p.add_argument("--methods", nargs="+", default=["gradient", "cmaes"])
    a = p.parse_args()

    cfg, bound = SimConfig(), 1.5
    net = make_network(a.topology, seed=0)
    stim = jnp.asarray(make_stimulus(a.stimulus, 2000, 10, seed=0))
    stim_test = jnp.asarray(make_stimulus(a.stimulus, 2000, 10, seed=500))
    theta_true = jnp.asarray(PRESETS[a.rule])
    scale = stat_scale(net, stim, cfg, jax.random.PRNGKey(1), N_THETA, bound)
    s_obs = sim_stats(theta_true, jax.random.PRNGKey(2), net, stim, cfg)

    eval_keys = jax.random.split(jax.random.PRNGKey(99), 8)   # held-out noise seeds
    L = lambda th: float(loss(jnp.asarray(th), eval_keys, net, stim, s_obs, scale, cfg))
    rng = np.random.default_rng(0)
    rand = [L(rng.uniform(-bound, bound, N_THETA).astype(np.float32)) for _ in range(16)]
    print(f"loss(theta_true) = {L(theta_true):.3f}  (noise floor)", flush=True)
    print(f"loss(theta=0)    = {L(np.zeros(N_THETA, np.float32)):.3f}")
    print(f"loss(prior)      = {np.median(rand):.3f} (median of 16 prior draws)")

    for name, fn in [("gradient", lambda k: infer_gradient(net, stim, s_obs, scale, cfg, k, bound,
                                                           n_steps=a.grad_steps)),
                     ("cmaes", lambda k: infer_cmaes(net, stim, s_obs, scale, cfg, k, bound,
                                                     max_iter=a.cmaes_iter, popsize=a.popsize))]:
        if name not in a.methods:
            continue
        t = time.time()
        res = fn(jax.random.PRNGKey(3))
        th = res["theta"]
        m = theta_metrics(th, theta_true, bound)
        fe = functional_error(th, theta_true, net, stim_test, scale, cfg, jax.random.PRNGKey(4))
        print(f"\n{name}: {time.time()-t:.0f}s  loss={L(th):.3f}  rmse={m['theta_rmse']:.3f} "
              f"cos={m['theta_cosine']:+.2f} pred={m['pred_rule']}  func_err(test stim)={fe:.3f}", flush=True)
        top = np.argsort(-np.abs(th))[:5]
        print("  largest inferred terms:", ", ".join(f"{TERM_NAMES[i]}={th[i]:+.2f}" for i in top))
    tt = np.asarray(theta_true)
    print("  true terms:            ", ", ".join(f"{TERM_NAMES[i]}={tt[i]:+.2f}"
                                                 for i in np.flatnonzero(tt)))


if __name__ == "__main__":
    main()
