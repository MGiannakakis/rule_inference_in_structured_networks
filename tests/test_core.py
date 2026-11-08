import jax
import jax.numpy as jnp
import numpy as np
import pytest

from ruleinf.infer.baseline import infer_from_weights
from ruleinf.rules import N_THETA, PRESETS, classify, rule_update
from ruleinf.simulate import SimConfig, recorded_stimulus, simulate
from ruleinf.stimuli import STIMULI, make_stimulus
from ruleinf.summary import summary_stats
from ruleinf.topologies import (TOPOLOGIES, make_network, network_stats,
                                rewire_degree_preserving, rewired_network)


@pytest.mark.parametrize("topo", list(TOPOLOGIES))
def test_topology_matched_stats(topo):
    s = network_stats(make_network(topo, n=100, density=0.1, seed=0))
    assert abs(s["density"] - 0.1) < 0.02
    assert abs(s["spectral_radius"] - 0.9) < 1e-3


def test_rewiring_preserves_degrees():
    rng = np.random.default_rng(0)
    m = TOPOLOGIES["modular"](80, 0.1, rng)
    r = rewire_degree_preserving(m, rng)
    assert (r.sum(0) == m.sum(0)).all() and (r.sum(1) == m.sum(1)).all()
    assert not np.diag(r).any() and (r != m).mean() > 0.05


def test_rewired_network_keeps_ei_and_weights():
    net = make_network("small_world", n=60, seed=1)
    rw = rewired_network(net)
    assert np.array_equal(np.asarray(net.sign), np.asarray(rw.sign))
    a = np.sort(np.asarray(net.G0)[np.asarray(net.mask) > 0])
    b = np.sort(np.asarray(rw.G0)[np.asarray(rw.mask) > 0])
    np.testing.assert_allclose(a / a.mean(), b / b.mean(), rtol=1e-5)  # same up to rho rescale


def test_rule_update_matches_analytic():
    rng = np.random.default_rng(0)
    r = jnp.asarray(rng.random(7))
    G = jnp.asarray(rng.random((7, 7)))
    oja = rule_update(jnp.asarray(PRESETS["oja"]), r, r, G)
    np.testing.assert_allclose(oja, np.outer(r, r) - (r ** 2)[:, None] * G, rtol=1e-5, atol=1e-6)
    hebb = rule_update(jnp.asarray(PRESETS["hebb"]), r, r, G)
    np.testing.assert_allclose(hebb, np.outer(r, r) - 0.2 * G, rtol=1e-5, atol=1e-6)


def test_classify_presets():
    for k, v in PRESETS.items():
        if k != "none":
            assert classify(v) == k


@pytest.mark.parametrize("stim", list(STIMULI))
def test_simulation_bounded_and_finite(stim):
    net = make_network("er", n=50, seed=0)
    s = jnp.asarray(make_stimulus(stim, 500, 10, seed=0))
    key = jax.random.PRNGKey(0)
    for th in [PRESETS["hebb"], np.full(N_THETA, 1.5, np.float32), np.full(N_THETA, -1.5, np.float32)]:
        R, G = simulate(jnp.asarray(th), net, s, key, SimConfig(), return_weights=True)
        assert np.isfinite(np.asarray(R)).all()
        exc = np.asarray(net.sign) > 0
        assert float(np.asarray(G)[:, exc].max()) <= float(net.gmax) + 1e-6
        assert float(G.min()) >= 0
        st = summary_stats(R, recorded_stimulus(s, 10))
        assert np.isfinite(np.asarray(st)).all()


def test_inhibitory_synapses_fixed_by_default():
    net = make_network("er", n=50, seed=0)
    s = jnp.asarray(make_stimulus("ou", 500, 10))
    _, G = simulate(jnp.asarray(PRESETS["hebb"]), net, s, jax.random.PRNGKey(0), SimConfig(),
                    return_weights=True)
    inh = np.asarray(net.sign) < 0
    np.testing.assert_allclose(np.asarray(G[-1])[inh], np.asarray(net.G0)[inh])


def test_gradients_finite():
    from ruleinf.infer.objective import loss
    net = make_network("er", n=40, seed=0)
    s = jnp.asarray(make_stimulus("ou", 400, 10))
    cfg = SimConfig()
    keys = jax.random.split(jax.random.PRNGKey(0), 2)
    s_obs = jnp.zeros(28)
    g = jax.grad(loss)(jnp.asarray(PRESETS["oja"]), keys, net, s, s_obs, jnp.ones(28), cfg)
    assert np.isfinite(np.asarray(g)).all() and np.abs(np.asarray(g)).sum() > 0


@pytest.mark.parametrize("rule", ["hebb", "oja", "bcm", "homeostatic"])
def test_weights_visible_baseline_recovers_rule(rule):
    net = make_network("modular", n=60, seed=0)
    s = make_stimulus("ou", 200, 10, seed=0)
    cfg = SimConfig(noise=0.05)
    res = infer_from_weights(PRESETS[rule], net, s, cfg, jax.random.PRNGKey(0), n_steps=100)
    np.testing.assert_allclose(res["theta"], PRESETS[rule], atol=1e-2)
