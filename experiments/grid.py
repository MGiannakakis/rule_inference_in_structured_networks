"""Cross-topology rule-inference grid.

For every true topology x rule x stimulus x seed, an activity-only observation is generated
on the TRUE network. The rule is then inferred while assuming a (possibly wrong) network:

    exact       the true network itself (weights + wiring known)
    resampled   a fresh draw from the same topology family
    rewired     degree-preserving rewiring of the true wiring (same degrees, weights, E/I)
    er          density-matched Erdos-Renyi network
    <family>    a draw from another topology family (e.g. "modular")

Example:
    python experiments/grid.py --preset smoke
    python experiments/grid.py --preset small --methods gradient cmaes sbi
"""
from __future__ import annotations

import argparse
import json
import time
import zlib
from dataclasses import asdict
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

from ruleinf.infer.baseline import infer_from_weights
from ruleinf.infer.cmaes import infer_cmaes
from ruleinf.infer.gradient import infer_gradient
from ruleinf.infer.objective import sim_stats, stat_scale
from ruleinf.metrics import functional_error, theta_metrics
from ruleinf.rules import N_THETA, PRESETS
from ruleinf.simulate import SimConfig
from ruleinf.stimuli import make_stimulus
from ruleinf.topologies import TOPOLOGIES, make_network, network_stats, rewired_network

PRESET_CONFIGS = {
    "smoke": dict(topologies=["er", "modular"], conditions=["exact", "er"], rules=["hebb", "oja"],
                  stimuli=["ou"], methods=["weights", "gradient"], seeds=[0], T=1000,
                  n=60, grad_steps=30, cmaes_iter=10, n_sims=1000),
    "small": dict(topologies=["er", "modular", "layered"],
                  conditions=["exact", "resampled", "rewired", "er"],
                  rules=["hebb", "oja", "bcm", "homeostatic"], stimuli=["ou"],
                  methods=["weights", "gradient", "cmaes"], seeds=[0, 1], T=2000, n=100,
                  grad_steps=150, cmaes_iter=60, n_sims=10_000),
    "full": dict(topologies=list(TOPOLOGIES),
                 conditions=["exact", "resampled", "rewired", "er"] + list(TOPOLOGIES),
                 rules=[k for k in PRESETS if k != "none"],
                 stimuli=["white_noise", "ou", "patterns", "sequence"],
                 methods=["weights", "gradient", "cmaes", "sbi"], seeds=[0, 1, 2], T=2000,
                 n=100, grad_steps=200, cmaes_iter=80, n_sims=20_000),
}


def assumed_network(true_topo: str, true_net, condition: str, seed: int, n: int):
    if condition == "exact":
        return true_net
    if condition == "resampled":
        return make_network(true_topo, n=n, seed=seed + 1000)
    if condition == "rewired":
        return rewired_network(true_net, seed=seed + 2000)
    if condition == "er":
        return make_network("er", n=n, seed=seed + 3000,
                            density=network_stats(true_net)["density"])
    if condition in TOPOLOGIES:
        return make_network(condition, n=n, seed=seed + 4000)
    raise ValueError(condition)


def run(cfg_run: dict, out: Path, bound: float = 1.5) -> pd.DataFrame:
    sim_cfg = SimConfig()
    n, T = cfg_run["n"], cfg_run["T"]
    rows, thetas = [], {}
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for seed in cfg_run["seeds"]:
        for stim_name in cfg_run["stimuli"]:
            stim = jnp.asarray(make_stimulus(stim_name, T, 10, seed=seed))
            stim_test = jnp.asarray(make_stimulus(stim_name, T, 10, seed=seed + 500))
            for true_topo in cfg_run["topologies"]:
                true_net = make_network(true_topo, n=n, seed=seed)
                true_scale = stat_scale(true_net, stim, sim_cfg, jax.random.PRNGKey(seed),
                                        N_THETA, bound)
                for cond in cfg_run["conditions"]:
                    if cond == true_topo:
                        continue                         # identical to "resampled"
                    net_a = assumed_network(true_topo, true_net, cond, seed, n)
                    key = jax.random.PRNGKey(
                        zlib.crc32(f"{seed}|{stim_name}|{true_topo}|{cond}".encode()) % 2**31)
                    k_scale, k_sbi, key = jax.random.split(key, 3)
                    scale = stat_scale(net_a, stim, sim_cfg, k_scale, N_THETA, bound)
                    posterior = None
                    if "sbi" in cfg_run["methods"]:
                        from ruleinf.infer.sbi_npe import simulate_training_set, train_npe
                        th_tr, x_tr = simulate_training_set(net_a, stim, sim_cfg, k_sbi,
                                                            cfg_run["n_sims"], bound)
                        posterior = train_npe(th_tr, x_tr, bound, seed)
                    for rule in cfg_run["rules"]:
                        theta_true = jnp.asarray(PRESETS[rule])
                        k_obs, k_inf, k_test, key = jax.random.split(key, 4)
                        s_obs = sim_stats(theta_true, k_obs, true_net, stim, sim_cfg)
                        for method in cfg_run["methods"]:
                            tm = time.time()
                            extra = {}
                            if method == "weights":
                                if cond != "exact":
                                    continue             # baseline only makes sense on the true net
                                res = infer_from_weights(theta_true, true_net, stim, sim_cfg, k_inf)
                            elif method == "gradient":
                                res = infer_gradient(net_a, stim, s_obs, scale, sim_cfg, k_inf,
                                                     bound, n_steps=cfg_run["grad_steps"])
                            elif method == "cmaes":
                                res = infer_cmaes(net_a, stim, s_obs, scale, sim_cfg, k_inf, bound,
                                                  max_iter=cfg_run["cmaes_iter"], seed=seed)
                            elif method == "sbi":
                                from ruleinf.infer.sbi_npe import posterior_summary
                                res = posterior_summary(posterior, s_obs, theta_true, bound)
                                extra = {"post_std_mean": float(res["std"].mean()),
                                         "contraction_mean": float(res["contraction"].mean()),
                                         "hpd_rank": res["hpd_rank"]}
                            else:
                                raise ValueError(method)
                            th = np.asarray(res["theta"])
                            row = dict(seed=seed, stimulus=stim_name, true_topology=true_topo,
                                       condition=cond, rule=rule, method=method,
                                       loss=res.get("loss", np.nan),
                                       **theta_metrics(th, theta_true, bound),
                                       functional_error=functional_error(
                                           th, theta_true, true_net, stim_test, true_scale,
                                           sim_cfg, k_test),
                                       seconds=time.time() - tm, **extra)
                            rows.append(row)
                            thetas["|".join(map(str, (seed, stim_name, true_topo, cond, rule,
                                                      method)))] = th
                            print(f"[{time.time()-t0:7.1f}s] {stim_name:10s} {true_topo:11s} "
                                  f"{cond:11s} {rule:11s} {method:8s} "
                                  f"rmse={row['theta_rmse']:.3f} cos={row['theta_cosine']:+.2f} "
                                  f"pred={row['pred_rule']:11s} func={row['functional_error']:.3f}",
                                  flush=True)
                    pd.DataFrame(rows).to_csv(out / "results.csv", index=False)
    np.savez(out / "thetas.npz", **thetas)
    (out / "config.json").write_text(json.dumps({**cfg_run, "sim": asdict(sim_cfg),
                                                 "bound": bound}, indent=2))
    return pd.DataFrame(rows)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--preset", default="smoke", choices=list(PRESET_CONFIGS))
    p.add_argument("--out", default=None)
    for k in ["topologies", "conditions", "rules", "stimuli", "methods"]:
        p.add_argument(f"--{k}", nargs="+")
    p.add_argument("--seeds", nargs="+", type=int)
    for k in ["T", "n", "grad_steps", "cmaes_iter", "n_sims"]:
        p.add_argument(f"--{k.replace('_', '-')}", dest=k, type=int)
    args = vars(p.parse_args())
    cfg_run = dict(PRESET_CONFIGS[args.pop("preset")])
    out = Path(args.pop("out") or f"results/{time.strftime('%Y%m%d-%H%M%S')}")
    cfg_run.update({k: v for k, v in args.items() if v is not None})
    df = run(cfg_run, out)
    print(df.groupby(["method", "condition"])[["theta_rmse", "theta_cosine",
                                                "functional_error"]].mean().round(3))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
