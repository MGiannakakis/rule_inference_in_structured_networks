"""Plastic rate network simulator (JAX).

    x_{t+1} = x_t + dt/tau * (-x_t + (sign*G_t) r_t + B u_t + bias) + noise
    r_t     = tanh(relu(x_t))                       (rates in [0, 1))
    G_{t+1} = clip(G_t + eta * mask_E * dG(theta, r_t, G_t), 0, gmax)

By default only excitatory (E-presynaptic) synapses are plastic; inhibition is fixed.

Activity is recorded every ``rec_every`` steps. The outer scan step is rematerialised
(``jax.checkpoint``) so backprop through long trajectories fits in memory.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp

from .rules import rule_update
from .topologies import Network


@dataclass(frozen=True)
class SimConfig:
    dt_tau: float = 0.1       # dt / tau
    eta: float = 1e-3         # plasticity rate per step
    plastic_inhibitory: bool = False  # if False only E-presynaptic synapses are plastic
    noise: float = 0.05       # process noise std (per sqrt(step))
    obs_noise: float = 0.01   # observation noise on recorded rates
    bias: float = 0.3         # tonic input
    rec_every: int = 10       # record every k steps
    plastic: bool = True


def rate(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.tanh(jax.nn.relu(x))


@partial(jax.jit, static_argnames=("cfg", "return_weights"))
def simulate(theta: jnp.ndarray, net: Network, stim: jnp.ndarray, key: jax.Array,
             cfg: SimConfig = SimConfig(), return_weights: bool = False):
    """Returns recorded rates (T // rec_every, N), plus magnitudes G if ``return_weights``."""
    T, N, k = stim.shape[0], net.G0.shape[0], cfg.rec_every
    n_rec = T // k
    k_proc, k_obs = jax.random.split(key)
    xi = jax.random.normal(k_proc, (n_rec, k, N)) * cfg.noise * jnp.sqrt(cfg.dt_tau)
    U = stim[: n_rec * k].reshape(n_rec, k, -1)
    pmask = net.mask if cfg.plastic_inhibitory else net.mask * (net.sign > 0)

    def micro(carry, inp):
        x, G = carry
        u, z = inp
        r = rate(x)
        x_new = x + cfg.dt_tau * (-x + (net.sign * G) @ r + net.B @ u + cfg.bias) + z
        if cfg.plastic:
            G_new = jnp.clip(G + cfg.eta * rule_update(theta, r, r, G), 0.0, net.gmax)
            G = jnp.where(pmask > 0, G_new, G)
        return (x_new, G), None

    @jax.checkpoint
    def macro(carry, inp):
        carry, _ = jax.lax.scan(micro, carry, inp)
        x, G = carry
        return carry, ((rate(x), G) if return_weights else rate(x))

    carry0 = (jnp.zeros(N), net.G0)
    _, out = jax.lax.scan(macro, carry0, (U, xi))
    R = out[0] if return_weights else out
    R = R + cfg.obs_noise * jax.random.normal(k_obs, R.shape)
    return (R, out[1]) if return_weights else R


def recorded_stimulus(stim: jnp.ndarray, rec_every: int) -> jnp.ndarray:
    """Stimulus averaged over each recording bin, aligned with simulate()'s output."""
    n_rec = stim.shape[0] // rec_every
    return stim[: n_rec * rec_every].reshape(n_rec, rec_every, -1).mean(1)
