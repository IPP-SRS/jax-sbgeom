"""speed / curvature / length of coils: analytic curves, gradients, discrete coils."""
import jax
import jax.numpy as jnp
import numpy as onp
import pytest
from scipy.special import ellipe

import jax_sbgeom as jsb
from jax_sbgeom.coils import FourierCoil
from jax_sbgeom.coils.discrete_coil import DiscreteCoil

jax.config.update("jax_enable_x64", True)


def _circle(radius):
    return FourierCoil(fourier_cos=jnp.array([[radius, 0.0, 0.0]]), fourier_sin=jnp.array([[0.0, radius, 0.0]]),
                       centre_i=jnp.zeros(3))


def _ellipse(a, b):
    return FourierCoil(fourier_cos=jnp.array([[a, 0.0, 0.0]]), fourier_sin=jnp.array([[0.0, b, 0.0]]),
                       centre_i=jnp.zeros(3))


def _wavy(scale=1.0):
    k = jnp.arange(1, 6)
    cos = scale * jnp.stack([3.0 / k ** 2 * (k == 1) + 0.2 / k ** 3, 0.3 / k ** 2 * (k == 2), 0.5 / k ** 3], axis=-1)
    sin = scale * jnp.stack([0.1 / k ** 3, 3.0 / k ** 2 * (k == 1) + 0.15 / k ** 3, 0.4 / k ** 2 * (k == 1)], axis=-1)
    return FourierCoil(fourier_cos=cos, fourier_sin=sin, centre_i=jnp.array([0.3, -0.2, 0.1]))


@pytest.mark.parametrize("radius", [0.7, 5.0])
def test_circle(radius):
    c = _circle(radius)
    s = jnp.linspace(0, 1, 7, endpoint=False)
    assert jnp.allclose(c.speed(s), 2 * jnp.pi * radius, rtol=1e-13)
    assert jnp.allclose(c.curvature(s), 1.0 / radius, rtol=1e-12)
    assert jnp.allclose(c.length(), 2 * jnp.pi * radius, rtol=1e-13)


def test_ellipse_length_and_curvature():
    a, b = 3.0, 1.2
    c = _ellipse(a, b)
    exact = 4 * a * ellipe(1 - b ** 2 / a ** 2)
    assert jnp.allclose(c.length(), exact, rtol=1e-12)
    assert jnp.allclose(c.curvature(0.0), a / b ** 2, rtol=1e-12)        # at (a, 0)
    assert jnp.allclose(c.curvature(0.25), b / a ** 2, rtol=1e-12)       # at (0, b)


def test_shapes_and_vectorisation():
    c = _wavy()
    for s in (0.3, jnp.linspace(0, 1, 6), jnp.linspace(0, 1, 6).reshape(2, 3)):
        assert c.speed(s).shape == jnp.shape(s)
        assert c.curvature(s).shape == jnp.shape(s)


def test_length_converges_and_matches_fine_polyline():
    c = _wavy()
    ref = c.length(4096)
    errs = [abs(c.length(n) - ref) / ref for n in (8, 12, 16)]
    assert errs[0] > errs[1] > errs[2] > 0 and errs[0] > 1e-6 and abs(c.length(64) - ref) < 1e-12 * ref
    l3 = ref
    s = jnp.arange(200000) / 200000
    x = c.position(s)
    chord = jnp.sum(jnp.linalg.norm(jnp.roll(x, -1, 0) - x, axis=-1))
    assert jnp.allclose(chord, l3, rtol=1e-9)           # chord error ~ h^2


def test_curvature_matches_frenet_normal_direction():
    # kappa * n_hat = dT/ds_arc  =>  |dT/ds| / speed with T the unit tangent
    c = _wavy()
    s = jnp.array([0.07, 0.33, 0.81])
    dT = jax.vmap(jax.jacfwd(c.tangent))(s)
    assert jnp.allclose(jnp.linalg.norm(dT, axis=-1) / c.speed(s), c.curvature(s), rtol=1e-10)


def test_length_invariances():
    c = _wavy()
    assert jnp.allclose(c.length(), c.reverse_parametrisation().length(), rtol=1e-13)
    shifted = FourierCoil(c.fourier_cos, c.fourier_sin, c.centre_i + 10.0)
    assert jnp.allclose(shifted.length(), c.length(), rtol=1e-13)


def test_length_scales_linearly_and_gradient():
    c = _wavy()
    scaled = FourierCoil(2.5 * c.fourier_cos, 2.5 * c.fourier_sin, c.centre_i)
    assert jnp.allclose(scaled.length(), 2.5 * c.length(), rtol=1e-13)
    f = lambda cs: FourierCoil(cs, c.fourier_sin, c.centre_i).length()
    g = jax.grad(f)(c.fourier_cos)
    d = jax.random.normal(jax.random.PRNGKey(0), c.fourier_cos.shape)
    h = 1e-6
    fd = (f(c.fourier_cos + h * d) - f(c.fourier_cos - h * d)) / (2 * h)
    assert jnp.allclose(jnp.sum(g * d), fd, rtol=1e-7)


def test_fourier_fit_of_samples_reproduces_circle():
    theta = jnp.arange(64) * 2 * jnp.pi / 64
    pts = jnp.stack([2.0 * jnp.cos(theta), 2.0 * jnp.sin(theta), 0.0 * theta], axis=-1)
    f = jsb.coils.convert_to_fourier_coil(DiscreteCoil.from_positions(pts))
    assert jnp.allclose(f.length(), 4 * jnp.pi, rtol=1e-12)
    assert jnp.allclose(f.curvature(0.37), 0.5, rtol=1e-10)


def test_discrete_coil():
    sq = DiscreteCoil.from_positions(jnp.array([[0, 0, 0], [2, 0, 0], [2, 3, 0], [0, 3, 0]], dtype=float))
    assert jnp.allclose(sq.length(), 10.0)
    with pytest.raises(NotImplementedError):
        sq.curvature(0.1)
    # speed of the piecewise-linear interpolant: segment length x number of segments
    assert jnp.allclose(sq.speed(0.1), 4 * 2.0)
