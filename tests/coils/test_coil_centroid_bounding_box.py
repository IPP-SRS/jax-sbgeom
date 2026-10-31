"""Coil.centroid / Coil.bounding_box: analytic curves, brute force, gradients, discrete coils."""
import jax
import jax.numpy as jnp
import numpy as onp
import pytest

from jax_sbgeom.coils import FourierCoil
from jax_sbgeom.coils.discrete_coil import DiscreteCoil

jax.config.update("jax_enable_x64", True)


def _ellipse(a, b, centre=(0.0, 0.0, 0.0)):
    return FourierCoil(fourier_cos=jnp.array([[a, 0.0, 0.0]]), fourier_sin=jnp.array([[0.0, b, 0.0]]),
                       centre_i=jnp.array(centre))


def _wavy(scale=1.0):
    k = jnp.arange(1, 6)
    cos = scale * jnp.stack([3.0 / k ** 2 * (k == 1) + 0.2 / k ** 3, 0.3 / k ** 2 * (k == 2), 0.5 / k ** 3], axis=-1)
    sin = scale * jnp.stack([0.1 / k ** 3, 3.0 / k ** 2 * (k == 1) + 0.15 / k ** 3, 0.4 / k ** 2 * (k == 1)], axis=-1)
    return FourierCoil(fourier_cos=cos, fourier_sin=sin, centre_i=jnp.array([0.3, -0.2, 0.1]))


def test_circle_and_ellipse_off_centre():
    c = _ellipse(2.0, 2.0, (1.0, -2.0, 3.0))
    assert jnp.allclose(c.centroid(), jnp.array([1.0, -2.0, 3.0]), atol=1e-13)
    lo, hi = c.bounding_box()
    assert jnp.allclose(lo, jnp.array([-1.0, -4.0, 3.0]), atol=1e-12)
    assert jnp.allclose(hi, jnp.array([3.0, 0.0, 3.0]), atol=1e-12)
    e = _ellipse(3.0, 1.0, (0.5, 0.0, 0.0))
    lo, hi = e.bounding_box()
    assert jnp.allclose(lo, jnp.array([-2.5, -1.0, 0.0]), atol=1e-12)
    assert jnp.allclose(hi, jnp.array([3.5, 1.0, 0.0]), atol=1e-12)


def test_tilted_ring():
    # circle of radius r in a plane with unit normal n, centre c: half-extent along axis i is r sqrt(1 - n_i^2)
    r, c = 1.7, onp.array([0.4, -0.3, 2.0])
    n = onp.array([1.0, 2.0, 2.0]) / 3.0
    u = onp.cross(n, [0.0, 0.0, 1.0]); u /= onp.linalg.norm(u)
    v = onp.cross(n, u)
    coil = FourierCoil(fourier_cos=jnp.asarray((r * u)[None]), fourier_sin=jnp.asarray((r * v)[None]), centre_i=jnp.asarray(c))
    half = r * onp.sqrt(1 - n ** 2)
    lo, hi = coil.bounding_box()
    assert jnp.allclose(lo, c - half, atol=1e-12) and jnp.allclose(hi, c + half, atol=1e-12)
    assert jnp.allclose(coil.centroid(), c, atol=1e-13)
    assert jnp.allclose(hi[2] - lo[2], 2 * half[2], atol=1e-12)


def test_wavy_vs_dense_brute_force_and_polish_helps():
    c = _wavy()
    s = jnp.arange(2_000_000) / 2_000_000
    x = onp.asarray(c.position(s))
    w = onp.asarray(c.speed(s))
    ref_centroid = (x * w[:, None]).sum(0) / w.sum()
    assert jnp.allclose(c.centroid(), ref_centroid, atol=1e-12)
    # dense sampling error ~ (h^2 / 2) |x''| ~ 1e-11 -> refine the reference with the same polish at high resolution
    lo, hi = c.bounding_box()
    assert onp.all(onp.asarray(lo) <= x.min(0) + 1e-13) and onp.all(onp.asarray(hi) >= x.max(0) - 1e-13)
    assert onp.allclose(lo, x.min(0), atol=1e-10) and onp.allclose(hi, x.max(0), atol=1e-10)
    lo2, hi2 = c.bounding_box(n_samples=4096)
    assert onp.allclose(lo, lo2, atol=1e-13) and onp.allclose(hi, hi2, atol=1e-13)
    # coarse sampling is rescued by the polish
    lo_c, hi_c = c.bounding_box(n_samples=16)
    xs = onp.asarray(c.position(jnp.arange(16) / 16))                               # the unpolished sampled box
    lo_n = jnp.asarray(xs.min(0))
    assert onp.allclose(lo_c, lo, atol=1e-9) and onp.allclose(hi_c, hi, atol=1e-9)
    assert jnp.max(jnp.abs(lo_n - lo)) > 1e-4


def _perturbed(x):
    c = _wavy()
    d = jax.random.normal(jax.random.PRNGKey(3), c.fourier_cos.shape) * 1e-1
    e = jax.random.normal(jax.random.PRNGKey(4), c.fourier_sin.shape) * 1e-1
    return FourierCoil(c.fourier_cos + x * d, c.fourier_sin + x * e, c.centre_i)


def test_gradients_vs_finite_differences():
    f_box = lambda x: jnp.concatenate(_perturbed(x).bounding_box())
    f_cen = lambda x: _perturbed(x).centroid()
    for f in (f_box, f_cen):
        g = jax.jacfwd(f)(0.0)
        eps = 1e-6
        fd = (f(eps) - f(-eps)) / (2 * eps)
        assert jnp.allclose(g, fd, rtol=1e-6, atol=1e-8)
    assert jnp.all(jnp.isfinite(jax.jacrev(f_box)(0.0)))


def test_jit_vmap():
    coils = jax.vmap(lambda x: _perturbed(x))(jnp.array([0.0, 0.5, 1.0]))
    lo, hi = jax.jit(jax.vmap(lambda c: c.bounding_box()))(coils)
    cen = jax.vmap(lambda c: c.centroid())(coils)
    assert lo.shape == hi.shape == cen.shape == (3, 3)
    lo1, hi1 = _perturbed(0.5).bounding_box()
    assert jnp.allclose(lo[1], lo1, atol=1e-13) and jnp.allclose(hi[1], hi1, atol=1e-13)


def test_discrete_square():
    sq = DiscreteCoil.from_positions(jnp.array([[0, 0, 0], [2, 0, 0], [2, 3, 1], [0, 3, 1]], dtype=float))
    lo, hi = sq.bounding_box()
    assert jnp.allclose(lo, jnp.array([0.0, 0.0, 0.0])) and jnp.allclose(hi, jnp.array([2.0, 3.0, 1.0]))
    # centroid of the perimeter of a rectangle (symmetric) is its centre
    assert jnp.allclose(sq.centroid(), jnp.array([1.0, 1.5, 0.5]))
    # unequal segment lengths: weights matter
    tri = DiscreteCoil.from_positions(jnp.array([[0, 0, 0], [4, 0, 0], [0, 3, 0]], dtype=float))
    mids = jnp.array([[2, 0, 0], [2, 1.5, 0], [0, 1.5, 0]], dtype=float)
    w = jnp.array([4.0, 5.0, 3.0])
    assert jnp.allclose(tri.centroid(), (mids * w[:, None]).sum(0) / 12.0)
