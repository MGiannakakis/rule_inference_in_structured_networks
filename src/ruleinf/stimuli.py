"""Input stimuli: each generator returns a float32 array of shape (T, M)."""
from __future__ import annotations

from typing import Callable

import numpy as np


def white_noise(T: int, M: int, rng: np.random.Generator, amp: float = 1.0) -> np.ndarray:
    return (amp * rng.standard_normal((T, M))).astype(np.float32)


def ou(T: int, M: int, rng: np.random.Generator, amp: float = 1.0, tau: float = 50.0) -> np.ndarray:
    """Ornstein-Uhlenbeck noise with correlation time ``tau`` steps and stationary std ``amp``."""
    a = np.exp(-1.0 / tau)
    u = np.zeros((T, M))
    u[0] = rng.standard_normal(M)
    xi = rng.standard_normal((T, M)) * np.sqrt(1 - a ** 2)
    for t in range(1, T):
        u[t] = a * u[t - 1] + xi[t]
    return (amp * u).astype(np.float32)


def patterns(T: int, M: int, rng: np.random.Generator, amp: float = 1.5,
             n_patterns: int = 4, dur: int = 100, noise: float = 0.2) -> np.ndarray:
    """Low-dimensional structure: blocks of a few fixed patterns in random order, plus noise."""
    P = rng.standard_normal((n_patterns, M))
    order = rng.integers(n_patterns, size=int(np.ceil(T / dur)))
    u = np.repeat(P[order], dur, axis=0)[:T]
    return (amp * u + noise * rng.standard_normal((T, M))).astype(np.float32)


def sequence(T: int, M: int, rng: np.random.Generator, amp: float = 1.5,
             n_patterns: int = 4, dur: int = 50, noise: float = 0.2) -> np.ndarray:
    """A fixed sequence of patterns that repeats (temporal structure for STDP-like effects)."""
    P = rng.standard_normal((n_patterns, M))
    order = np.arange(int(np.ceil(T / dur))) % n_patterns
    u = np.repeat(P[order], dur, axis=0)[:T]
    return (amp * u + noise * rng.standard_normal((T, M))).astype(np.float32)


STIMULI: dict[str, Callable[..., np.ndarray]] = {
    "white_noise": white_noise,
    "ou": ou,
    "patterns": patterns,
    "sequence": sequence,
}


def make_stimulus(name: str, T: int, M: int, seed: int = 0, **kwargs) -> np.ndarray:
    return STIMULI[name](T, M, np.random.default_rng(seed), **kwargs)
