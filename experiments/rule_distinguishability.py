"""Stage-1 sanity check: does each preset rule evolve the weights gradually, and are the rules
distinguishable from summary statistics on a given topology?

Prints the mean excitatory weight over time per rule, then a matrix of between-rule distances
in normalised summary-stat space divided by the within-rule (noise-seed) distance. Values
around 1 mean the two rules can't be told apart from activity.

    python experiments/rule_distinguishability.py            # eta = 1e-3 (default)
    python experiments/rule_distinguishability.py --eta 5e-4 --topologies er modular layered
"""
from __future__ import annotations

import argparse

import jax
import jax.numpy as jnp
import numpy as np

from ruleinf.rules import PRESETS
from ruleinf.simulate import SimConfig, recorded_stimulus, simulate
from ruleinf.stimuli import make_stimulus
from ruleinf.summary import summary_stats
from ruleinf.topologies import make_network


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eta", type=float, default=1e-3)
    p.add_argument("--topologies", nargs="+", default=["er", "modular", "layered"])
    p.add_argument("--stimulus", default="ou")
    p.add_argument("--n-seeds", type=int, default=4)
    a = p.parse_args()

    np.set_printoptions(precision=2, suppress=True, linewidth=220)
    cfg = SimConfig(eta=a.eta)
    stim = jnp.asarray(make_stimulus(a.stimulus, 2000, 10, seed=1))
    U = recorded_stimulus(stim, cfg.rec_every)
    names = list(PRESETS)
    for topo in a.topologies:
        net = make_network(topo, seed=0)
        E = net.mask * (net.sign > 0)
        S = []
        for name in names:
            th = jnp.asarray(PRESETS[name])
            _, G = simulate(th, net, stim, jax.random.PRNGKey(0), cfg, return_weights=True)
            s = np.stack([np.asarray(summary_stats(simulate(th, net, stim, jax.random.PRNGKey(k), cfg), U))
                          for k in range(a.n_seeds)])
            S.append(s)
            gE = [float((G[t] * E).sum() / E.sum()) for t in np.linspace(0, G.shape[0] - 1, 5).astype(int)]
            at_max = float(((G[-1] >= net.gmax * 0.999) * E).sum() / E.sum())
            print(f"{topo:8s} {name:12s} G_E over time {np.array(gE)}  rate per window "
                  f"{np.asarray(s.mean(0)[0::7])}  frac_at_gmax {at_max:.2f}", flush=True)
        S = np.stack(S)                                  # rules x seeds x stats
        scale = S.reshape(-1, S.shape[-1]).std(0) + 1e-6
        mu = S.mean(1) / scale
        noise = np.mean([np.linalg.norm((S[i, x] - S[i, y]) / scale)
                         for i in range(len(names)) for x in range(a.n_seeds) for y in range(x)])
        D = np.linalg.norm(mu[:, None] - mu[None], axis=-1) / noise
        print(f"\n{topo}: between-rule distance / within-rule seed noise ({noise:.2f})")
        print("             " + " ".join(f"{n[:6]:>6s}" for n in names))
        for i, n in enumerate(names):
            print(f"  {n:11s}" + " ".join(f"{D[i, j]:6.1f}" for j in range(len(names))))
        print()


if __name__ == "__main__":
    main()
