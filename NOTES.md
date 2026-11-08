# Progress notes

Last updated 2026-11-08. Development so far was on a laptop (RTX 2050, 4 GB); continuing on the lab machine.

## Setting up on a new machine
```bash
git clone https://github.com/MGiannakakis/rule_inference_in_structured_networks.git
cd rule_inference_in_structured_networks
conda env create -f environment.yml        # creates env "rule_inference" and installs ruleinf (-e .)
conda activate rule_inference
python -c "import jax; print(jax.devices())"   # check that the GPU is visible
pytest -q                                      # expect 19 passed (~1 min)
```
`environment.yml` installs `jax[cuda12]`. If the lab GPU driver needs CUDA 13, change it to `jax[cuda13]`.
Torch is CPU-only on purpose: it is used only for SBI density estimation, while JAX does the simulations.
`results/` is gitignored, so the key numbers from earlier runs are copied below.

## Status
Done:
- Package `ruleinf`: 5 topologies (E/I, density 0.1, spectral radius 0.9) and controls
  (degree-preserving rewiring, density-matched ER); 18-term polynomial rule space with 6
  presets; 4 stimulus types; JAX simulator (only E synapses plastic, eta = 1e-3); 28 permutation-invariant summary stats;
  inference with weights-visible least squares, gradient (BPTT), CMA-ES and SBI (NPE).
- `experiments/grid.py`: full cross-topology grid (presets smoke / small / full).
- `tests/test_core.py`: 19 tests, all passing.

Bugs found and fixed along the way:
- eta = 0.02 saturated all Hebbian-like rules in the first window, which made them trivially indistinguishable. Now 1e-3.
- The clip to [0, gmax] was also truncating inhibitory magnitudes. Now only plastic synapses are clipped,
  and gmax is computed from E weights.

## Results so far
**Rule distinguishability** (`experiments/rule_distinguishability.py`; between-rule distance ÷ seed noise):
most rule pairs are 3–15× the noise. Exception: homeostatic vs none is about 1× on ER/modular but 3.5× on
layered. This is a first sign that topology changes identifiability.

**Smoke grid** (n=60, T=1000, 30 gradient steps): the weights-visible baseline recovers θ exactly
(RMSE 0.000, cosine 1.00). Gradient inference from activity gives cosine ≈ 0, but on the exact network
the functional error is sometimes low (0.02–0.08), which points to degeneracy.

**Positive control: FAILED** (`positive_control.py --topology modular --rule oja`, n=100, T=2000):

| | loss | θ RMSE | cosine | functional error (test stimulus) |
|---|---|---|---|---|
| θ_true (noise floor) | 0.013 | 0 | 1 | – |
| prior draws (median) | 2.25 | | | |
| gradient, 150 steps × 4 restarts (142 s) | 0.283 | 0.64 | −0.16 | 0.79 |
| CMA-ES, 60 generations × 24 (24 s) | 0.203 | 0.44 | +0.33 | 0.61 |

Both methods stay 15–20× above the noise floor, and many coefficients hit the ±1.5 bound.
**Fix this before running the grid**, otherwise the mismatched-topology results are meaningless.

## Next steps (in order)
1. Diagnose the failed positive control: run CMA-ES with a large budget
   `python experiments/positive_control.py --methods cmaes --cmaes-iter 400 --popsize 48`
   (stopped on the laptop, not yet run).
   - It reaches the floor (~0.013) with a wrong θ → degeneracy. Shrink the rule basis and/or enrich the observations.
   - It stays above the floor → optimisation problem. Add restarts (IPOP), lower sigma0, use a smaller basis.
2. Candidate fixes: a reduced basis (drop the pre²/post² cross terms, ~8 terms); more summary-stat
   windows (8) or extra stats (E/I population rates); several stimuli per inference.
3. Once the positive control passes for hebb/oja/bcm/homeostatic: `grid.py --preset small`
   (a few hours on the laptop), then add SBI (`--methods ... sbi`, 10k–20k simulations per assumed network).
4. Analysis/plots: heatmap of true topology × assumed condition (θ error, functional error, posterior contraction).

## Design caveats to keep in mind
- In the "exact" condition the inferrer is given the true initial weights, which is privileged information;
  "resampled" is the realistic version of a "known topology".
- BCM uses a fixed threshold, because a sliding threshold can't be expressed in the polynomial family.
- Input B drives all neurons in every topology (including layered), to keep the comparison clean.
- Gradient uses full BPTT with per-bin checkpointing, not truncated BPTT.
