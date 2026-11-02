import jax
import jax.numpy as jnp
import numpy as np

from jax_sbgeom.jax_utils.stationary import polish_stationary


def _f(x, a):
    return a * jnp.cos(x[0]) + jnp.cos(2 * x[0])


def test_polish_finds_maximum_and_gradient_is_envelope():
    f = _f
    x, ok = polish_stationary(f, jnp.array([[0.3]]), 2.0)
    assert ok[0] and abs(float(x[0, 0])) < 1e-9
    g = jax.grad(lambda a: f(polish_stationary(f, jnp.array([[0.3]]), a)[0][0], a))(2.0)
    np.testing.assert_allclose(g, 1.0, rtol=1e-8)          # d/da of the max = cos(0)


def _wave(s, a):
    return 1e6 * (a * jnp.cos(2 * jnp.pi * (s - 0.3)) + 0.5 * jnp.cos(4 * jnp.pi * (s - 0.3)))


def test_maximise_periodic_polishes_to_the_true_maximum():
    from jax_sbgeom.jax_utils.stationary import Polish, maximise_periodic
    sampled = maximise_periodic(_wave, 2.0, 7, None)
    polished = maximise_periodic(_wave, 2.0, 7, Polish(1e-10, 12))
    np.testing.assert_allclose(polished, 2.5e6, rtol=1e-12)       # f(0.3) = a + 1/2, independent of the scale 1e6
    assert float(sampled) < float(polished)


def test_maximise_periodic_gradient_is_envelope():
    from jax_sbgeom.jax_utils.stationary import Polish, maximise_periodic
    g = jax.grad(lambda a: maximise_periodic(_wave, a, 7, Polish(1e-10, 12)))(2.0)
    np.testing.assert_allclose(g, 1e6, rtol=1e-8)                 # d/da of the maximum = 1e6 cos(0)


def test_maximise_periodic_falls_back_to_the_sample_when_newton_does_not_converge():
    from jax_sbgeom.jax_utils.stationary import Polish, maximise_periodic
    sampled = maximise_periodic(_wave, 2.0, 7, None)
    np.testing.assert_allclose(maximise_periodic(_wave, 2.0, 7, Polish(1e-10, 1)), sampled)
