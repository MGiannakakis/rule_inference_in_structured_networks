# Rule inference across network topologies

Does connectivity structure determine whether a synaptic plasticity rule can be identified
from neural activity? Plastic E/I rate networks with different topologies are driven by
different stimuli under several plasticity rules. The rule is then inferred from
**activity only** while assuming the true network, a resampled one, a degree-preserving
rewiring, a density-matched ER network, or another topology family.

## Setup
```bash
conda env create -f environment.yml      # or: conda activate rule_inference (already created)
pytest -q
```

## Components (`src/ruleinf/`)
| module | content |
|---|---|
| `topologies.py` | ER, small-world, modular (SBM), scale-free (Chung–Lu), layered; E/I (Dale), matched density 0.1 and spectral radius 0.9; degree-preserving rewiring |
| `rules.py` | 18-term polynomial rule `dG = Σ θ_abc pre^a post^b w^c`; presets: hebb, anti_hebb, oja, covariance, bcm (fixed threshold), homeostatic |
| `stimuli.py` | white noise, OU, pattern blocks, repeating sequence |
| `simulate.py` | JAX rate network with plasticity; only E synapses are plastic by default; checkpointed for BPTT |
| `summary.py` | permutation-invariant, differentiable summary stats in 4 time windows (28 dims) |
| `infer/` | `baseline` (weights visible, least squares), `gradient` (Adam through BPTT), `cmaes`, `sbi_npe` (amortised NPE) |
| `metrics.py` | θ RMSE, cosine, nearest-preset rule, functional error on the true net with a held-out stimulus |

All methods share the same rule space, summary statistics and normalised discrepancy.

## Experiments
```bash
python experiments/grid.py --preset smoke             # ~2 min
python experiments/grid.py --preset small             # 3 topologies x 4 conditions x 4 rules
python experiments/grid.py --preset full              # everything, incl. SBI
python experiments/positive_control.py --topology modular --rule oja
python experiments/rule_distinguishability.py         # are rules separable per topology?
```
See [NOTES.md](NOTES.md) for current status, results so far and next steps.
Results go to `results/<run>/results.csv` (one row per true-topology × condition × rule ×
method × seed) and `thetas.npz`.

## Notes / caveats
- When weights are visible, the rule is recovered exactly (RMSE ≈ 0). Activity-only inference is
  the regime that is actually being tested.
- Coefficient error and *functional* error can disagree, because distinct θ can produce
  indistinguishable activity (cf. Ramesh et al. 2023). Report both.
- The inferrer is given the assumed network's initial weights. In the "exact" condition this
  is privileged information; "resampled" is the realistic version of a "known topology".
