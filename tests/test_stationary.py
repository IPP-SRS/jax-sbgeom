import jax
import jax.numpy as jnp
import numpy as np

from jax_sbgeom.jax_utils.stationary import polish_stationary


def test_polish_finds_maximum_and_gradient_is_envelope():
    f = lambda x, a: a * jnp.cos(x[0]) + jnp.cos(2 * x[0])
    x, ok = polish_stationary(f, jnp.array([[0.3]]), 2.0)
    assert ok[0] and abs(float(x[0, 0])) < 1e-9
    g = jax.grad(lambda a: f(polish_stationary(f, jnp.array([[0.3]]), a)[0][0], a))(2.0)
    np.testing.assert_allclose(g, 1.0, rtol=1e-8)          # d/da of the max = cos(0)
