"""Connectivity topologies and network construction (pure numpy).

Convention: ``mask[i, j]`` / ``W[i, j]`` is the synapse from presynaptic neuron j
onto postsynaptic neuron i. No self-connections.

Every network is E/I (Dale's law): the sign of a synapse is set by the presynaptic
neuron, and plasticity acts on non-negative magnitudes ``G`` with ``W = sign * G``.
All topologies are generated at a common target density and the initial weights are
rescaled to a common spectral radius, so differences between families come from the
wiring pattern rather than from trivial statistics.
"""
from __future__ import annotations

from typing import Callable, NamedTuple

import jax.numpy as jnp
import numpy as np


class Network(NamedTuple):
    """JAX pytree holding everything the simulator needs."""

    mask: jnp.ndarray   # (N, N) float {0,1}
    sign: jnp.ndarray   # (N, N) float {+1,-1}, column-wise (presynaptic) sign
    G0: jnp.ndarray     # (N, N) initial non-negative magnitudes
    B: jnp.ndarray      # (N, M) input weights
    gmax: jnp.ndarray   # scalar, upper bound on magnitudes


# --------------------------------------------------------------------------- masks
def _no_self(m: np.ndarray) -> np.ndarray:
    np.fill_diagonal(m, False)
    return m


def erdos_renyi(n: int, density: float, rng: np.random.Generator) -> np.ndarray:
    return _no_self(rng.random((n, n)) < density)


def small_world(n: int, density: float, rng: np.random.Generator, beta: float = 0.1) -> np.ndarray:
    """Directed Watts-Strogatz: ring lattice (k neighbours each side), presynaptic rewiring."""
    k = max(1, int(round(density * (n - 1) / 2)))
    m = np.zeros((n, n), bool)
    idx = np.arange(n)
    for off in range(1, k + 1):
        m[idx, (idx + off) % n] = True
        m[idx, (idx - off) % n] = True
    for i, j in zip(*np.nonzero(m)):
        if rng.random() < beta:
            free = np.flatnonzero(~m[i])
            free = free[free != i]
            if free.size:
                m[i, j] = False
                m[i, rng.choice(free)] = True
    return m


def modular(n: int, density: float, rng: np.random.Generator,
            n_modules: int = 4, ratio: float = 10.0) -> np.ndarray:
    """Stochastic block model with p_in = ratio * p_out, matched overall density."""
    labels = np.arange(n) % n_modules
    rng.shuffle(labels)
    f_in = (n / n_modules - 1) / (n - 1)
    p_in = min(1.0, density / (f_in + (1 - f_in) / ratio))
    p_out = p_in / ratio
    same = labels[:, None] == labels[None, :]
    p = np.where(same, p_in, p_out)
    return _no_self(rng.random((n, n)) < p)


def scale_free(n: int, density: float, rng: np.random.Generator, gamma: float = 2.5) -> np.ndarray:
    """Directed Chung-Lu graph with power-law expected in/out degrees, matched density."""
    w = (np.arange(n) + 1.0) ** (-1.0 / (gamma - 1.0))
    w_in, w_out = rng.permutation(w), rng.permutation(w)
    outer = w_in[:, None] * w_out[None, :]
    np.fill_diagonal(outer, 0.0)
    lo, hi = 0.0, 1e6
    for _ in range(100):  # bisection on the scale so that mean edge prob = density
        c = 0.5 * (lo + hi)
        if np.minimum(1.0, c * outer).sum() / (n * (n - 1)) > density:
            hi = c
        else:
            lo = c
    p = np.minimum(1.0, lo * outer)
    return _no_self(rng.random((n, n)) < p)


def layered(n: int, density: float, rng: np.random.Generator,
            n_layers: int = 4, ff_to_rec: float = 3.0) -> np.ndarray:
    """Feedforward chain of layers (l -> l+1) plus sparse within-layer recurrence."""
    layer = np.repeat(np.arange(n_layers), int(np.ceil(n / n_layers)))[:n]
    ff = layer[:, None] == layer[None, :] + 1     # post in l+1, pre in l
    rec = layer[:, None] == layer[None, :]
    np.fill_diagonal(rec, False)
    p_rec = density * n * (n - 1) / (ff_to_rec * ff.sum() + rec.sum())
    p = np.where(ff, min(1.0, ff_to_rec * p_rec), np.where(rec, min(1.0, p_rec), 0.0))
    return _no_self(rng.random((n, n)) < p)


TOPOLOGIES: dict[str, Callable[..., np.ndarray]] = {
    "er": erdos_renyi,
    "small_world": small_world,
    "modular": modular,
    "scale_free": scale_free,
    "layered": layered,
}


def rewire_degree_preserving(mask: np.ndarray, rng: np.random.Generator,
                             swaps_per_edge: int = 10) -> np.ndarray:
    """Double-edge swaps (a<-b, c<-d) -> (a<-d, c<-b); preserves every in- and out-degree."""
    m = mask.copy()
    post, pre = (a.copy() for a in np.nonzero(m))
    n_edges = post.size
    for _ in range(swaps_per_edge * n_edges):
        e1, e2 = rng.integers(n_edges, size=2)
        a, b, c, d = post[e1], pre[e1], post[e2], pre[e2]
        if e1 == e2 or a == d or c == b or m[a, d] or m[c, b]:
            continue
        m[a, b] = m[c, d] = False
        m[a, d] = m[c, b] = True
        pre[e1], pre[e2] = d, b
    return m


# ------------------------------------------------------------------------ networks
def _assemble(mask: np.ndarray, exc: np.ndarray, mags: np.ndarray, B: np.ndarray,
              rho: float, gmax_factor: float) -> Network:
    sign = np.where(exc[None, :], 1.0, -1.0) * np.ones_like(mags)
    G = mags * mask
    radius = np.max(np.abs(np.linalg.eigvals(sign * G)))
    G = G * (rho / max(radius, 1e-8))
    exc_syn = mask & exc[None, :]
    gmax = gmax_factor * G[exc_syn].mean()      # bound for (plastic) excitatory magnitudes
    f = lambda a: jnp.asarray(a, dtype=jnp.float32)
    return Network(f(mask), f(sign), f(G), f(B), f(gmax))


def make_network(topology: str, n: int = 100, density: float = 0.1, seed: int = 0,
                 frac_exc: float = 0.8, rho: float = 0.9, n_in: int = 10,
                 input_gain: float = 1.0, gmax_factor: float = 5.0,
                 mask: np.ndarray | None = None, **topo_kwargs) -> Network:
    """Sample a network of the given topology family (or use a supplied mask)."""
    rng = np.random.default_rng(seed)
    if mask is None:
        mask = TOPOLOGIES[topology](n, density, rng, **topo_kwargs)
    exc = rng.permutation(n) < int(round(frac_exc * n))
    mags = rng.lognormal(0.0, 0.5, size=(n, n))
    mags[:, ~exc] *= frac_exc / (1 - frac_exc)          # balance E and I input
    B = rng.normal(0.0, input_gain / np.sqrt(n_in), size=(n, n_in))
    return _assemble(mask, exc, mags, B, rho, gmax_factor)


def rewired_network(net: Network, seed: int = 0, rho: float = 0.9,
                    gmax_factor: float = 5.0) -> Network:
    """Degree-preserving rewiring of ``net``; keeps E/I identities, input weights and the
    multiset of weight magnitudes (shuffled onto the new edges)."""
    rng = np.random.default_rng(seed)
    mask = np.asarray(net.mask) > 0
    new_mask = rewire_degree_preserving(mask, rng)
    exc = np.asarray(net.sign)[0] > 0
    mags = np.zeros(mask.shape)
    mags[new_mask] = rng.permutation(np.asarray(net.G0)[mask])
    return _assemble(new_mask, exc, mags, np.asarray(net.B), rho, gmax_factor)


def network_stats(net: Network) -> dict[str, float]:
    m = np.asarray(net.mask) > 0
    n = m.shape[0]
    W = np.asarray(net.sign * net.G0)
    return {
        "density": m.sum() / (n * (n - 1)),
        "spectral_radius": float(np.max(np.abs(np.linalg.eigvals(W)))),
        "in_degree_cv": float(m.sum(1).std() / m.sum(1).mean()),
        "out_degree_cv": float(m.sum(0).std() / m.sum(0).mean()),
        "reciprocity": float((m & m.T).sum() / m.sum()),
    }
