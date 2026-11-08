"""Polynomial local plasticity rules.

    dG_ij = eta * sum_{a,b,c} theta_abc * pre_j^a * post_i^b * G_ij^c

with a, b in {0,1,2} and c in {0,1} (18 coefficients). Every named rule is a point in this
shared theta-space, so all inference methods search the same space and recovery errors are
directly comparable.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

PRE_POW = (0, 1, 2)
POST_POW = (0, 1, 2)
W_POW = (0, 1)
TERMS: list[tuple[int, int, int]] = [(a, b, c) for c in W_POW for b in POST_POW for a in PRE_POW]
N_THETA = len(TERMS)
TERM_NAMES = [f"pre^{a}·post^{b}·w^{c}" for a, b, c in TERMS]


def theta_from_dict(coefs: dict[tuple[int, int, int], float]) -> np.ndarray:
    theta = np.zeros(N_THETA, dtype=np.float32)
    for term, v in coefs.items():
        theta[TERMS.index(term)] = v
    return theta


# (pre, post, w) exponents. Magnitudes chosen so the rules act on a comparable timescale.
PRESETS: dict[str, np.ndarray] = {
    "none": theta_from_dict({}),
    "hebb": theta_from_dict({(1, 1, 0): 1.0, (0, 0, 1): -0.2}),
    "anti_hebb": theta_from_dict({(1, 1, 0): -1.0, (0, 0, 0): 0.05}),
    "oja": theta_from_dict({(1, 1, 0): 1.0, (0, 2, 1): -1.0}),
    "covariance": theta_from_dict({(1, 1, 0): 1.0, (1, 0, 0): -0.3, (0, 1, 0): -0.3,
                                   (0, 0, 0): 0.09, (0, 0, 1): -0.1}),
    # BCM with a fixed threshold (0.3); a sliding threshold is not expressible in this family.
    "bcm": theta_from_dict({(1, 2, 0): 1.0, (1, 1, 0): -0.3}),
    # synaptic scaling towards a target postsynaptic rate of 0.2
    "homeostatic": theta_from_dict({(0, 0, 1): 0.2, (0, 1, 1): -1.0}),
}


def theta_to_tensor(theta: jnp.ndarray) -> jnp.ndarray:
    """(18,) -> (c, b, a) = (2, 3, 3)."""
    return theta.reshape(len(W_POW), len(POST_POW), len(PRE_POW))


def _powers(r: jnp.ndarray, pows) -> jnp.ndarray:
    return jnp.stack([r ** p for p in pows], axis=-1)            # (N, P)


def rule_update(theta: jnp.ndarray, r_pre: jnp.ndarray, r_post: jnp.ndarray,
                G: jnp.ndarray) -> jnp.ndarray:
    """Unscaled dG (N, N) for a single time step; entry [i, j] = post i, pre j."""
    Th = theta_to_tensor(theta)
    P = _powers(r_pre, PRE_POW)                                    # (N, 3)
    Q = _powers(r_post, POST_POW)                                  # (N, 3)
    dG = jnp.zeros_like(G)
    for ci, c in enumerate(W_POW):
        dG = dG + (Q @ Th[ci] @ P.T) * (G ** c)
    return dG


def rule_features(r_pre: np.ndarray, r_post: np.ndarray, G: np.ndarray) -> np.ndarray:
    """Monomial features (..., N_THETA) per synapse, for the weights-visible baseline."""
    feats = [(r_pre[..., None, :] ** a) * (r_post[..., :, None] ** b) * (G ** c)
             for a, b, c in TERMS]
    return np.stack(feats, axis=-1)


def classify(theta_hat: np.ndarray, candidates: list[str] | None = None) -> str:
    """Nearest named rule by cosine similarity (scale-free)."""
    candidates = candidates or [k for k in PRESETS if k != "none"]
    t = np.asarray(theta_hat)
    sims = {k: float(t @ PRESETS[k] / (np.linalg.norm(t) * np.linalg.norm(PRESETS[k]) + 1e-12))
            for k in candidates}
    return max(sims, key=sims.get)
